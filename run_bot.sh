#!/bin/bash

IMAGE_NAME="discordbot"
ENV_FILE=".env"

# Check if user can run Docker without sudo
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not accessible. Ensure you are in the 'docker' group or have permissions."
    echo "Run 'sudo usermod -aG docker $USER', log out/in, or use 'newgrp docker'."
    exit 1
fi

create_env_file() {
    echo "Setting up environment variables for the first time..."

    # --- Interactive Setup ---
    echo "--- Core Application ---"
    read -s -p "Enter your Discord Token (required): " discord_token; echo
    if [ -z "$discord_token" ]; then
        echo "Error: DISCORD_TOKEN is required."
        exit 1
    fi
    read -p "Enter your Timezone (default: Europe/Amsterdam): " timezone
    [ -z "$timezone" ] && timezone="Europe/Amsterdam"

    echo -e "\n--- Database & Memory ---"
    read -s -p "Enter Neo4j Password (required): " neo4j_password; echo
    if [ -z "$neo4j_password" ]; then
        echo "Error: NEO4J_PASSWORD is required."
        exit 1
    fi
    read -p "Reset Milvus DB on next start? (true/false, default: false): " reset_mdb
    [ -z "$reset_mdb" ] && reset_mdb="false"

    echo -e "\n--- LLM & AI Services ---"
    read -p "Enter your Primary LocalAI URL (default: http://localhost:8080): " localai_url
    [ -z "$localai_url" ] && localai_url="http://localhost:8080"

    read -s -p "Enter your Primary LocalAI API Key (optional, press Enter to skip): " localai_api_key; echo
    read -p "Enter your LLM Model Name (default: Gen): " model_name
    [ -z "$model_name" ] && model_name="Gen"
    
    # Secondary LocalAI is optional
    read -p "Enter your Secondary LocalAI URL (optional, press Enter to skip): " localai_2_url
    if [ -n "$localai_2_url" ]; then
        read -s -p "Enter your Secondary LocalAI API Key (optional): " localai_2_api_key; echo
        read -p "Enter your Secondary Embeddings Model (default: all-MiniLM-L6-v2): " embeddings_2_model
        [ -z "$embeddings_2_model" ] && embeddings_2_model="all-MiniLM-L6-v2"
    else
        # Ensure secondary variables are blank if the URL is not provided
        localai_2_api_key=""
        embeddings_2_model=""
    fi

    read -p "Enter your TinyGen API URL (optional, press Enter to skip): " tinygen_api_url

    echo -e "\n--- Behavior & Performance ---"
    read -p "Enter LLM Temperature (default: 0.8): " llm_temperature
    [ -z "$llm_temperature" ] && llm_temperature="0.8"

    echo -e "\n--- Image Generation ---"
    read -p "Enter your ComfyUI URL (optional, enables ComfyUI mode): " comfy_url
    if [ -n "$comfy_url" ]; then
        read -p "Enter default ComfyUI workflow filename (e.g., flux-api.json): " wflow
        read -p "Enter anime/illustrious ComfyUI workflow filename (e.g., il-api.json): " wflow_il
    else
        wflow=""
        wflow_il=""
    fi

    # Write all variables to the .env file
    cat <<EOL > "$ENV_FILE"
# Core Application
DISCORD_TOKEN=$discord_token
TIMEZONE=$timezone
BOT_USER_ID=

# Database & Memory
NEO4J_USER=neo4j
NEO4J_PASSWORD=$neo4j_password
RESET_MDB=$reset_mdb

# LLM & AI Services
LOCALAI_URL=$localai_url
LOCALAI_API_KEY=$localai_api_key
MODEL_NAME=$model_name
EMBEDDINGS_MODEL=all-MiniLM-L6-v2
RERANK_MODEL=jina-reranker-v1-base-en
VL_MODEL=moondream2
LOCALAI_2_URL=$localai_2_url
LOCALAI_2_API_KEY=$localai_2_api_key
EMBEDDINGS_2_MODEL=$embeddings_2_model
TINYGEN_API_URL=$tinygen_api_url

# Behavior & Performance
LLM_TEMPERATURE=$llm_temperature

# Image Generation
SD_MODELL=stablediffusion
IMAGE_SIZE=960x640
MAX_IMAGE_SIZE=1280
IMAGE_N=1
IMAGE_STEPS=40
IMAGE_CFG_SCALE=6.5
COMFY_URL=$comfy_url
WFLOW=$wflow
WFLOW_IL=$wflow_il
EOL

    echo "Generated $ENV_FILE with your inputs."
}

build_and_run() {
    # Create directories as user
    echo "Creating directories..."
    mkdir -p ./data ./data/mdb/etcd ./data/mdb/minio ./data/mdb/milvus ./data/neo4j/data ./data/neo4j/logs ./data/neo4j/import ./data/neo4j/plugins ./data/logs

    echo "Building Docker image: $IMAGE_NAME..."
    docker build --no-cache -t "$IMAGE_NAME" \
        --build-arg USER_UID=$(id -u) \
        --build-arg USER_GID=$(id -g) .
    if [ $? -ne 0 ]; then
        echo "Error: Docker build failed."
        exit 1
    fi

    echo "Starting services with Docker Compose..."
    docker compose down --volumes --remove-orphans
    docker compose up -d
    if [ $? -eq 0 ]; then
        echo "Services are running. View logs with: docker compose logs -f dbot"
        # Ensure directories are user-owned
        sleep 5 # Give services a moment to create initial files
        if ! chown -R $(id -u):$(id -g) ./data ./web 2>/dev/null; then
            echo "Attempting to change ownership with sudo..."
            sudo chown -R $(id -u):$(id -g) ./data ./web
            if [ $? -ne 0 ]; then
                echo "Warning: Failed to change ownership of some files. Check permissions if you encounter issues."
            fi
        fi
    else
        echo "Error: Failed to start services. Check logs with 'docker compose logs'."
        exit 1
    fi
}

if [ ! -f "$ENV_FILE" ]; then
    echo "No $ENV_FILE found. Starting interactive setup..."
    create_env_file
    build_and_run
else
    echo "$ENV_FILE exists. Skipping setup and running the services..."
    build_and_run
fi
