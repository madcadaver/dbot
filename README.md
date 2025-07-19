# Gen AI - A Multi-Modal Discord Bot

Gen is a sophisticated, multi-modal AI assistant for Discord, designed with a persistent memory, advanced tool usage, and a customizable personality. It leverages a powerful backend of graph and vector databases to create a rich, context-aware conversational experience.

---

## How It Works 🧠

This project is more than a simple chatbot; it's a full RAG (Retrieval-Augmented Generation) system. When you interact with Gen, a multi-step process begins:

1.  **Ingestion**: Your message is logged into a **Neo4j** graph database, creating a permanent record of the interaction linked to your user profile.
2.  **Context Retrieval**: The bot queries **Neo4j** for recent conversation history (short-term memory) and searches a **Milvus** vector database for semantically relevant past conversations or facts (long-term memory).
3.  **Thought & Decision**: This contextual information is compiled with a list of available "tools" and sent to a Large Language Model (LLM). The LLM then decides whether to respond directly or to use a tool (like searching the web or generating an image).
4.  **Action & Response**: If a tool is chosen, the bot executes it, processes the result, and loops the information back into the thought process. Once a final decision is made, the response is sent back to Discord.

---

## Features ✨

* **Persistent Memory**: Using Neo4j and Milvus, Gen never forgets a conversation, fact, or user preference.
* **Tool Use**: Gen can perform actions beyond simple chat, including:
    * `perform_web_search`: Search the web to answer questions about current events.
    * `generate_image`: Create new images from a text prompt using Stable Diffusion or ComfyUI.
    * `store_knowledge`: Permanently learn new facts you tell it.
* **User Profiles**: Remembers user-specific aliases and preferences.
* **Multi-Modal**: Can understand text and analyze uploaded images.
* **Containerized**: The entire application stack is managed with Docker and Docker Compose for easy setup and deployment.

---

## Setup & Installation ⚙️

The project is designed to be run with Docker and Docker Compose.

### 1. Prerequisites

* **Docker** and **Docker Compose** installed.
* The current user must have permissions to run Docker commands. If not, add them to the `docker` group:
    ```bash
    sudo usermod -aG docker $USER
    ```
    You will need to log out and back in for this change to take effect.

### 2. Clone the Repository

```bash
git clone https://github.com/madcadaver/dbot.git
cd dbot
```

### 3. Run the Setup Script

The `run_bot.sh` script will guide you through the initial setup.

```bash
bash run_bot.sh
```

The first time you run this, it will detect that there is no `.env` file and will start an interactive setup process. It will ask for:

* Your **Discord Bot Token**.
* The URL for your **LocalAI** instance.
* Other configuration settings for models and features.

This will create a `.env` file in the root directory with all your settings.

### 4. Starting the Bot

After the initial setup, running the script again will start all the services.

```bash
bash run_bot.sh
```

This command will:
* Build the necessary Docker images.
* Start all services defined in `docker-compose.yml` in detached mode (`-d`).
* Set the correct file permissions on the `data` and `web` directories.

You can view the logs for the bot, web, or database services with:
```bash
docker logs dbot -f
docker logs neo4j -f
docker logs milvus-standalone -f
```

### 5. Stopping the Bot

To stop all running services and remove the containers, use:
```bash
docker compose down
```

---

## Services 🛠️

The `docker-compose.yml` file orchestrates the following services:

* **`dbot`**: The main Python application that runs the Discord bot.
* **`web`**: A container for a future web interface.
* **`neo4j`**: The graph database that stores the structured memory of the bot. You can access the Neo4j Browser at `http://localhost:7474`.
* **`milvus-standalone`**, **`etcd`**, **`minio`**: The Milvus vector database and its dependencies, used for fast semantic search of memories.

---

## Customization 🎨

* **Personality**: Edit `data/gen_profile.json` to change Gen's name, appearance, likes/dislikes, and core personality traits.
* **Tools**: Add or modify the tool definitions in `capabilities.py` to expand or change what Gen can do.
* **Models**: Change the model names for generation, embeddings, and vision in your `.env` file.
