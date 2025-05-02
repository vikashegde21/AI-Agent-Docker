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

## 🤖 Running the AI Agent Generator

You can use the built-in agent to generate new Dockerized applications and reports:

```sh
python docker_agent.py "create a python application with FastAPI and Redis" --run-and-report --output-dir my_output --s3-bucket my-bucket --s3-object my-report.json
```

**Example:**
```sh
python docker_agent.py "create a golang REST API with a health endpoint" --run-and-report --output-dir my_output --s3-bucket test-container-report --s3-object my-report16.json
```

This will:
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

---

## 🤖 About the Project
This project is part of an AI agent automation suite. It enables:
- Rapid prototyping of AI-powered microservices
- Seamless integration with other containers via Docker Compose
- Experimentation with different agent architectures (Python, Go, etc.)

> **Tip:** You can extend this setup to include Redis, databases, or other services by editing `docker-compose.yml`.

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
