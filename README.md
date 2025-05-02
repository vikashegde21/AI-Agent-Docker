# 🤖 AI Agents Docker Project

Welcome to the **AI Agents Docker Project**! 🚀

This project provides a ready-to-use environment for building, running, and experimenting with AI agent applications using Docker. It supports Python (FastAPI, Flask), Go, and more, and is designed for rapid development, testing, and deployment of intelligent agents and microservices.

---

## 🛠️ Quick Start

1. **(Optional) Create and activate a Python virtual environment**
   ```sh
   python -m venv cont
   # On Windows:
   cont\Scripts\activate
   # On Linux/macOS:
   source cont/bin/activate
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   ```

3. **Clone or create your project directory**
   - Place your `Dockerfile` and `requirements.txt` at the project root.
   - Place your application code in the `app/` directory (e.g., `app/main.py`).

4. **Build the Docker image**
   ```sh
   docker build -t ai-agents-docker .
   ```

5. **Run the container with live code reload**
   ```sh
   docker run --rm -it -p 8000:8000 -v $(pwd)/app:/app/app ai-agents-docker
   ```
   _This mounts your local `app/` directory into the container for instant code reloads._

6. **Test your AI agent**
   - 🌐 Visit [http://localhost:8000](http://localhost:8000) for the main endpoint.
   - ❤️ Healthcheck: [http://localhost:8000/health](http://localhost:8000/health)

7. **Check container health**
   ```sh
   docker ps --format '{{.Names}}: {{.Status}}'
   ```
   _The container is healthy if `/health` returns status 200._

---

## 🤖 AI Agent Features & Usage

This project includes a classic LangChain-based AI agent with the following capabilities:

- **Task-based Agent Execution**
  - Run a single agent task using:
    ```sh
    python docker_agent.py --agent-task "List files in my_output"
    ```
  - The agent can reason, plan, and use tools (file listing, shell commands, Docker interaction).

- **Conversational Chatbot Mode**
  - Start an interactive chat session with memory:
    ```sh
    python docker_agent.py --agent-chat
    ```
  - The agent will remember context across turns and can answer questions, run shell commands, or list Docker containers interactively.

- **Dockerized Application Generation**
  - Generate, build, and run Dockerized applications (Python, Go, etc.) using:
    ```sh
    python docker_agent.py "create a python application with FastAPI and Redis" --run-and-report --output-dir my_output --s3-bucket my-bucket --s3-object my-report.json
    ```
  - This will:
    - Generate all necessary Docker and app files in `my_output/`
    - Build and run the container
    - Save a detailed container report to S3

---

## 🤩 Features
- 🧠 **AI Agent Ready**: Easily run FastAPI, Flask, or Go-based AI agents.
- 🔄 **Live Reload**: Mount your code for instant updates.
- 🔒 **Secure by Default**: Runs as a non-root user, all Python packages installed in `/app/.local`.
- 🐳 **Docker Best Practices**: Healthchecks, volume mounts, and custom networks supported.
- 📦 **Easy Dependency Management**: Just update `requirements.txt` and rebuild.
- 🧑‍💻 **Debug Friendly**: `docker exec -it <container_id> /bin/bash` for shell access.
- ☁️ **S3 Integration**: Upload container reports to AWS S3 for sharing or automation.

---

## 🛠️ Tools Available to the Agent
- **File Management**: List files in any directory.
- **Shell Commands**: Run shell commands and return their output.
- **Docker Management**: List and stop running Docker containers.
- (You can extend the agent with more tools as needed.)

---

## 📚 Example: Adding a New AI Agent
1. Add your agent code to `app/` (e.g., `app/my_agent.py`).
2. Update `requirements.txt` if you need more packages.
3. Rebuild and rerun the container as above.

---

## 🌟 Contributing
Pull requests and suggestions are welcome! Feel free to open an issue or submit a PR.

---

## 📄 License
MIT License

---

Made with ❤️ by the AI Agents Docker Project Team
