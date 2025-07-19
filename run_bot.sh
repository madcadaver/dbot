#!/bin/bash

IMAGE_NAME="discordbot"
ENV_FILE=".env"
ENV_SAMPLE=".env.sample"

# Check if user can run Docker without sudo
if ! docker info >/dev/null 2>&1; then
    echo "Error: Docker is not accessible. Ensure you are in the 'docker' group or have permissions."
    echo "Run 'sudo usermod -aG docker $USER', log out/in, or use 'newgrp docker'."
    exit 1
fi

create_env_file() {
    echo "Setting up environment variables for the first time..."
    if [ ! -f "$ENV_SAMPLE" ]; then
        cat <<EOL > "$ENV_SAMPLE"
DISCORD_TOKEN=
CLEAR_COMMANDS=
TIMEZONE=
RESET_MDB=
REBASE=
ENABLE_MEM=
USE_HISTORY=
LOCALAI_URL=
LOCALAI_API_KEY=
MODEL_NAME=
EMBEDDINGS_MODEL=
RERANK_MODEL=
VL_MODEL=
SD_MODELL=
IMAGE_SIZE=
MAX_IMAGE_SIZE=
IMAGE_N=
IMAGE_STEPS=
IMAGE_CFG_SCALE=
LOCALAI_2_URL=
LOCALAI_2_API_KEY=
EMBEDDINGS_2_MODEL=
EOL
        echo "Created $ENV_SAMPLE as a template."
    fi

    # Init Section
    read -s -p "Enter your Discord Token (required): " discord_token; echo
    if [ -z "$discord_token" ]; then
        echo "Error: DISCORD_TOKEN is required."
        exit 1
    fi

    read -p "Clear any leftover commands? (true/false, default: false): " clear_commands
    [ -z "$clear_commands" ] && clear_commands="false"

    read -p "Enter your Timezone (default: Europe/Amsterdam): " timezone
    [ -z "$timezone" ] && timezone="Europe/Amsterdam"

    read -p "Reset Milvus DB? (true/false, default: false): " reset_mdb
    [ -z "$reset_mdb" ] && reset_mdb="false"

    read -p "Rebase Milvus DB? (true/false, default: false): " rebase
    [ -z "$rebase" ] && rebase="false"

    read -p "Enable Memory functionality? (true/false, default: false): " enable_mem
    [ -z "$enable_mem" ] && enable_mem="false"

    read -p "Enable History functionality? (true/false, default: true): " use_history
    [ -z "$use_history" ] && use_history="true"

    # LocalAI_1 Section
    read -p "Enter your LocalAI URL (default: http://localhost:8080): " localai_url
    [ -z "$localai_url" ] && localai_url="http://localhost:8080"

    read -s -p "Enter your LocalAI API Key (optional, press Enter if none): " localai_api_key; echo

    read -p "Enter your Model Name (default: gpt-4): " model_name
    [ -z "$model_name" ] && model_name="gpt-4"

    read -p "Enter your Embeddings Model (default: all-MiniLM-L6-v2): " embeddings_model
    [ -z "$embeddings_model" ] && embeddings_model="all-MiniLM-L6-v2"

    read -p "Enter your Rerank Model (default: jina-reranker-v1-base-en): " rerank_model
    [ -z "$rerank_model" ] && rerank_model="jina-reranker-v1-base-en"

    read -p "Enter your Vision Model (e.g., moondream2, no default): " vl_model

    # Image Generation Section
    read -p "Enter your Stable Diffusion Model (default: stablediffusion): " sd_modell
    [ -z "$sd_modell" ] && sd_modell="stablediffusion"

    read -p "Enter default Image Size (default: 768x768): " image_size
    [ -z "$image_size" ] && image_size="768x768"

    read -p "Enter max Image Size (default: 1280): " max_image_size
    [ -z "$max_image_size" ] && max_image_size="1280"

    read -p "Enter default number of Images (default: 2): " image_n
    [ -z "$image_n" ] && image_n="2"

    read -p "Enter default Image Steps (default: 40): " image_steps
    [ -z "$image_steps" ] && image_steps="40"

    read -p "Enter default Image CFG Scale (default: 6.5): " image_cfg_scale
    [ -z "$image_cfg_scale" ] && image_cfg_scale="6.5"

    # LocalAI_2 Section (optional)
    read -p "Enter your Secondary LocalAI URL (optional, press Enter if none): " localai_2_url
    if [ -n "$localai_2_url" ]; then
        read -s -p "Enter your Secondary LocalAI API Key (optional, press Enter if none): " localai_2_api_key; echo
        read -p "Enter your Secondary Embeddings Model (default: all-MiniLM-L6-v2): " embeddings_2_model
        [ -z "$embeddings_2_model" ] && embeddings_2_model="all-MiniLM-L6-v2"
    else
        localai_2_api_key=""
        embeddings_2_model=""
    fi

    cat <<EOL > "$ENV_FILE"
DISCORD_TOKEN=$discord_token
CLEAR_COMMANDS=$clear_commands
TIMEZONE=$timezone
RESET_MDB=$reset_mdb
REBASE=$rebase
ENABLE_MEM=$enable_mem
USE_HISTORY=$use_history
LOCALAI_URL=$localai_url
LOCALAI_API_KEY=$localai_api_key
MODEL_NAME=$model_name
EMBEDDINGS_MODEL=$embeddings_model
RERANK_MODEL=$rerank_model
VL_MODEL=$vl_model
SD_MODELL=$sd_modell
IMAGE_SIZE=$image_size
MAX_IMAGE_SIZE=$max_image_size
IMAGE_N=$image_n
IMAGE_STEPS=$image_steps
IMAGE_CFG_SCALE=$image_cfg_scale
LOCALAI_2_URL=$localai_2_url
LOCALAI_2_API_KEY=$localai_2_api_key
EMBEDDINGS_2_MODEL=$embeddings_2_model
EOL

    echo "Generated $ENV_FILE with your inputs."
}

build_and_run() {
    # Create directories as user
    echo "Creating directories as current user..."
    mkdir -p ./data
    mkdir -p ./data/mdb/etcd
    mkdir -p ./data/mdb/minio
    mkdir -p ./data/mdb/milvus
    mkdir -p ./data/neo4j/data
    mkdir -p ./data/neo4j/logs
    mkdir -p ./data/neo4j/import
    mkdir -p ./data/neo4j/plugins
    mkdir -p ./data/logs
    mkdir -p ./data/backups
    mkdir -p ./web

    echo "Building Docker image: $IMAGE_NAME..."
    docker build --no-cache -t "$IMAGE_NAME" \
        --build-arg USER_UID=$(id -u) \
        --build-arg USER_GID=$(id -g) .
    if [ $? -ne 0 ]; then
        echo "Error: Docker build failed."
        exit 1
    fi

    echo "Starting services with Docker Compose..."
    # Force remove any existing containers to avoid conflicts
    echo "Force removing any existing containers..."
    for container in dbot milvus-standalone milvus-etcd milvus-minio dbot-web; do
        if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
            echo "Removing container $container..."
            docker rm -f $container
        fi
    done
    # Stop any remaining Docker Compose services and remove volumes/networks
    docker compose down --volumes --remove-orphans
    docker rmi dbot-web
    docker system prune --volumes -f
    # Start all services
    docker compose up -d
#    docker stop dbot-web
    if [ $? -eq 0 ]; then
        echo "Services are running. Check logs with 'docker logs dbot -f', 'docker logs milvus-standalone', or 'docker logs dbot-web -f'."
        # Ensure directories are user-owned, using sudo if necessary
        for dir in ./data ./web; do
            if ! chown -R $(id -u):$(id -g) $dir 2>/dev/null; then
                echo "Some files in $dir require sudo to change ownership. You may be prompted for your password."
                sudo chown -R $(id -u):$(id -g) $dir
                if [ $? -ne 0 ]; then
                    echo "Warning: Failed to change ownership of some files in $dir. Services are running, but check permissions."
                else
                    echo "Successfully changed ownership of $dir to $(id -u):$(id -g)."
                fi
            fi
        done
    else
        echo "Error: Failed to start services with Docker Compose."
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
