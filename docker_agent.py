#!/usr/bin/env python3
"""
Docker Container Generator Agent

This script creates Docker container applications based on user queries using OpenAI's GPT-4.1.
It handles parsing user requirements, generating appropriate Dockerfiles and application setups.
"""

import os
import argparse
import sys
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import logging
from dotenv import load_dotenv
import subprocess

import docker

from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.callbacks.manager import get_openai_callback

import boto3

from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.tools import tool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('docker-agent')

load_dotenv()

class DockerAgent:
    """Agent that generates Docker container applications based on user requirements."""
    
    def __init__(self, model_name: str = "openai/gpt-4.1", temperature: float = 0.1, verbose: bool = False):
        """
        Initialize the Docker Agent.
        
        Args:
            model_name: The OpenAI model to use
            temperature: The temperature setting for the model
            verbose: Whether to show verbose output
        """
        
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.endpoint = os.environ.get("OPENAI_API_ENDPOINT", "https://models.github.ai/inference")
        self.model_name = model_name
        self.temperature = temperature
        self.verbose = verbose
        
    
        self.llm = ChatOpenAI(
            base_url=self.endpoint,
            api_key=self.api_key,
            model_name=self.model_name,
            temperature=self.temperature
        )
        
        self.output_parser = self._create_output_parser()
        
        self.system_prompt = self._create_system_prompt()
        
        self.chain = self._create_chain()
    
    def _create_output_parser(self):
        """Create a structured output parser for Docker container components."""
        schemas = [
            ResponseSchema(name="dockerfile", description="The complete Dockerfile content"),
            ResponseSchema(name="app_files", description="A list of dictionaries with 'filename' and 'content' keys for each application file"),
            ResponseSchema(name="docker_compose", description="Optional docker-compose.yml file if needed"),
            ResponseSchema(name="setup_instructions", description="Step-by-step instructions to set up and run the Docker container"),
            ResponseSchema(name="explanation", description="Explanation of the Docker setup and how it fulfills the requirements")
        ]
        return StructuredOutputParser.from_response_schemas(schemas)
    
    def _create_system_prompt(self):
        """Create the system prompt for the Docker container generator."""
        format_instructions = self.output_parser.get_format_instructions()
        format_instructions = format_instructions.replace("{", "{{").replace("}", "}}")
        return f"""You are an expert Docker container developer and DevOps engineer. Your task is to create complete, development-ready Docker container applications based on user requirements.

You must:
1. Analyze the requirements carefully and clarify any ambiguities.
2. Always install curl in the Dockerfile for healthchecks and diagnostics.
3. Use a single-stage or multi-stage build as appropriate for development (favor speed and rebuildability).
4. Use the most appropriate base image for the language/framework.
5. Follow Docker and development best practices:
   - Favor fast rebuilds and code reload (e.g., use volume mounts, install dev dependencies)
   - Use non-root users when possible
   - Install all Python packages and their executables (like uvicorn, flask, etc.) to a user-writable location (e.g., /app/.local) in the builder stage
   - In the final image, set PATH to include the user-writable location (e.g., ENV PATH=/app/.local/bin:$PATH)
   - Set correct permissions and ownership for all files and executables so the non-root user can execute them
   - Clean up apt/yum cache after installing packages
   - Expose only necessary ports
   - Add healthchecks using curl that return healthy if the app is running and ready
   - Use environment variables for configuration
   - Avoid hardcoding secrets or credentials
6. If a database or other service is required, generate a docker-compose.yml with all necessary services and correct networking.
7. Provide clear, step-by-step setup instructions and explanations for development use.
8. Ensure the Dockerfile and all generated files are ready for local development and easy debugging (e.g., enable code reload, map source code as a volume).
9. If the user mentions Azure, always follow Azure best practices. When generating code for Azure, running terminal commands for Azure, or performing operations related to Azure, always invoke the Azure best practices tool if available.
10. Ensure the healthcheck is reliable and will report healthy when the application is ready to serve requests.
11. Never install Python packages with --user or to /root/.local in the builder stage if the final image runs as a non-root user. Always use --prefix=/app/.local and set permissions accordingly.
12. Make sure the user exists before using `USER appuser` in your Dockerfile. Always create the user (e.g., with `RUN useradd --create-home appuser` or `RUN adduser -D appuser`) before switching to it.

{format_instructions}

For the app_files, create a list of dictionaries, each containing the filename and content of each application file needed.
If a docker-compose.yml is needed, include it in the docker_compose field.
"""
    
    def _create_chain(self):
        """Create the RunnableSequence for processing user queries (new API)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "{query}")
        ])
        
        return prompt | self.llm | self.output_parser
    
    def _fix_multiline_run_instructions(self, dockerfile: str) -> str:
        """
        Fix multi-line RUN instructions so that no line starts with '&&'.
        Joins lines so that '&&' is always at the end of the previous line.
        """
        lines = dockerfile.splitlines()
        fixed_lines = []
        in_run = False
        buffer = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('RUN '):
                if in_run and buffer:
                    fixed_lines.append(' \
    '.join(buffer))
                    buffer = []
                in_run = True
                buffer.append(stripped)
            elif in_run and (stripped.startswith('&&') or stripped == '\\'):
                
                if buffer:
                    buffer[-1] = buffer[-1].rstrip(' \\') + ' ' + stripped
                else:
                    buffer.append(stripped)
            elif in_run and (stripped.startswith('    &&') or stripped.startswith('&&')):
                
                if buffer:
                    buffer[-1] = buffer[-1].rstrip(' \\') + ' ' + stripped.lstrip()
                else:
                    buffer.append(stripped.lstrip())
            elif in_run and (stripped == '' or not stripped.startswith('&&')):
                
                if buffer:
                    fixed_lines.append(' \
    '.join(buffer))
                    buffer = []
                in_run = False
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        if in_run and buffer:
            fixed_lines.append(' \
    '.join(buffer))
        return '\n'.join(fixed_lines)

    def _ensure_env_path(self, dockerfile: str) -> str:
        """
        Ensure ENV PATH includes /app/.local/bin and ENV PYTHONPATH includes /app/.local/lib/python3.11/site-packages in the Dockerfile after pip install.
        """
        lines = dockerfile.splitlines()
        new_lines = []
        inserted_path = False
        inserted_pythonpath = False
        python_version = "3.13"  # Update if you use a different Python version
        for i, line in enumerate(lines):
            new_lines.append(line)
            
            if (
                (not inserted_path or not inserted_pythonpath)
                and ("pip install" in line or "pip3 install" in line)
            ):
                if not any("ENV PATH" in l for l in lines):
                    new_lines.append('ENV PATH="/app/.local/bin:$PATH"')
                    inserted_path = True
                if not any("ENV PYTHONPATH" in l for l in lines):
                    new_lines.append(f'ENV PYTHONPATH="/app/.local/lib/python{python_version}/site-packages"')
                    inserted_pythonpath = True
        
        if not inserted_path and not any("ENV PATH" in l for l in lines):
            new_lines.append('ENV PATH="/app/.local/bin:$PATH"')
        if not inserted_pythonpath and not any("ENV PYTHONPATH" in l for l in lines):
            new_lines.append(f'ENV PYTHONPATH="/app/.local/lib/python{python_version}/site-packages"')
        return "\n".join(new_lines)

    def process_query(self, query: str) -> Dict:
        """
        Process a user query and generate Docker container application.
        
        Args:
            query: The user's requirements for the Docker container
        
        Returns:
            A dictionary containing the generated Dockerfile, application files, and instructions
        """
        logger.info(f"Processing query: {query}")
        try:
            with get_openai_callback() as cb:
                response = self.chain.invoke({"query": query})
                result = response  

                if self.verbose:
                    logger.info(f"OpenAI API usage: {cb}")

                
                dockerfile_lines = result["dockerfile"].splitlines()
                curl_installed = any("apt-get install" in line and "curl" in line for line in dockerfile_lines)
                if not curl_installed:
                    # Find the first RUN apt-get update
                    insert_idx = None
                    for i, line in enumerate(dockerfile_lines):
                        if "apt-get update" in line:
                            insert_idx = i
                            break
                    curl_cmd = "apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*"
                    if insert_idx is not None:
                        # Try to append to the existing RUN if possible
                        line = dockerfile_lines[insert_idx]
                        if line.strip().startswith("RUN"):
                            # If it's a multi-line RUN ending with '\\', append to the last line of the block
                            # Find the end of the RUN block
                            end_idx = insert_idx
                            for j in range(insert_idx, len(dockerfile_lines)):
                                if not dockerfile_lines[j].rstrip().endswith("\\"):
                                    end_idx = j
                                    break
                            # Append to the last line of the RUN block
                            last_line = dockerfile_lines[end_idx]
                            if last_line.rstrip().endswith("\\"):
                                # Remove the trailing '\\' and append '&& curl_cmd \'
                                dockerfile_lines[end_idx] = last_line.rstrip().rstrip('\\').rstrip() + f" && {curl_cmd} \\" 
                            else:
                                dockerfile_lines[end_idx] = last_line + f" && {curl_cmd}"
                        else:
                            # Not a RUN line, insert a new RUN after
                            dockerfile_lines.insert(insert_idx + 1, f"RUN {curl_cmd}")
                    else:
                        # If no apt-get update, add both after FROM
                        dockerfile_lines.insert(1, f"RUN apt-get update && {curl_cmd}")
                    result["dockerfile"] = "\n".join(dockerfile_lines)
                # Fix multi-line RUN instructions
                result["dockerfile"] = self._fix_multiline_run_instructions(result["dockerfile"])
                # Ensure ENV PATH is set
                result["dockerfile"] = self._ensure_env_path(result["dockerfile"])
                return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise
    
    def save_output(self, output: Dict, output_dir: str = "docker_output"):
        """
        Save the generated Docker container files to disk.
        Args:
            output: The structured output from process_query
            output_dir: The directory to save the files to
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        
        for fname, content in [
            ("Dockerfile", output["dockerfile"]),
            ("README.md", output.get("setup_instructions", ""))
        ]:
            if content and content.strip():
                with open(output_path / fname, "w", encoding="utf-8") as f:
                    f.write(content)

        for file_info in output["app_files"]:
            file_path = output_path / file_info["filename"]
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_info["content"])

        req_path = output_path / "requirements.txt"
        if not req_path.exists():
            with open(req_path, "w", encoding="utf-8") as f:
                f.write("requests\n")

        
        is_python_project = False
        for file_info in output["app_files"]:
            if file_info["filename"].endswith(".py"):
                is_python_project = True
                break
        if not is_python_project:
            dockerfile_path = output_path / "Dockerfile"
            if dockerfile_path.exists():
                with open(dockerfile_path, "r", encoding="utf-8") as df:
                    for line in df:
                        if line.strip().lower().startswith("from python"):
                            is_python_project = True
                            break
        if is_python_project and not any(f["filename"].endswith(".py") for f in output["app_files"]):
            app_py = output_path / "app.py"
            with open(app_py, "w", encoding="utf-8") as f:
                f.write("import requests\nprint('Hello from a simple Python application!')\n")

        readme_path = output_path / "README.md"
        if not readme_path.exists():
            app_port = 8000
            for file_info in output["app_files"]:
                if file_info["filename"].endswith(".py") and "Flask" in file_info["content"]:
                    app_port = 5000
            readme_content = f"""# Generated Application\n\nThis is a generated application.\n\n## Setup\n\n1. Build the Docker image:\n   ```sh\n   docker build -t my-app .\n   ```\n2. Run the application:\n   ```sh\n   docker run -p {app_port}:{app_port} my-app\n   ```\n\nOr use Docker Compose:\n\n   ```sh\n   docker-compose up --build\n   ```\n\n## Usage\n\n- Visit: http://localhost:{app_port}/\n- Healthcheck: http://localhost:{app_port}/health\n"""
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write(readme_content)
        dockerfile_path = output_path / "Dockerfile"
        if not dockerfile_path.exists():
            app_file = "main.py"
            app_port = 8000
            base_image = "python:3.11-slim"
            install_cmd = "RUN pip install --no-cache-dir -r requirements.txt"
            copy_reqs = "COPY requirements.txt ./"
            copy_code = "COPY . /app"
            expose = "EXPOSE 8000"
            health = "HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \\\n  CMD curl --fail http://localhost:8000/health || exit 1"
            cmd = f'CMD ["python", "{app_file}"]'
            for file_info in output["app_files"]:
                fname = file_info["filename"]
                content = file_info["content"]
                if fname.endswith(".py"):
                    app_file = fname
                    if "Flask" in content:
                        app_port = 5000
                        expose = "EXPOSE 5000"
                        health = "HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \\\n  CMD curl --fail http://localhost:5000/health || exit 1"
                    cmd = f'CMD ["python", "{app_file}"]'
                elif fname.endswith(".go"):
                    app_file = "app"
                    base_image = "golang:1.22-alpine"
                    install_cmd = "RUN go build -o app main.go"
                    copy_reqs = ""
                    copy_code = "COPY . ."
                    expose = "EXPOSE 8080"
                    health = "HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \\\n  CMD curl --fail http://localhost:8080/healthz || exit 1"
                    cmd = 'CMD ["./app"]'
                elif fname.endswith(".js"):
                    app_file = fname
                    base_image = "node:20-slim"
                    install_cmd = "RUN npm install"
                    copy_reqs = "COPY package*.json ./"
                    copy_code = "COPY . ."
                    expose = "EXPOSE 3000"
                    health = "HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \\\n  CMD curl --fail http://localhost:3000/health || exit 1"
                    cmd = f'CMD ["node", "{app_file}"]'
            static_dir = output_path / "static"
            static_copy = ""
            if static_dir.exists() and static_dir.is_dir():
                static_copy = "COPY --from=builder /app/static /app/static\n"
            dockerfile_content = f"FROM {base_image}\n\nRUN apt-get update \\\n    && apt-get install -y --no-install-recommends curl \\\n    && rm -rf /var/lib/apt/lists/*\n\nWORKDIR /app\n\n{copy_reqs}\n{install_cmd}\nENV PATH=\"/app/.local/bin:$PATH\"\n\n{copy_code}\n\n{expose}\n\n{health}\n\n{cmd}\n{static_copy}"
            with open(dockerfile_path, "w", encoding="utf-8") as f:
                f.write(dockerfile_content)

        logger.info(f"Docker container files saved to {output_path.absolute()} and Dockerfile/docker-compose.yml also saved to project root.")
        return output_path.absolute()
    
    def build_and_push_docker_image(self, output_dir: str, image_name: str = "mydockerapp", tag: str = "latest"):
        """
        Build the Docker image and push it to Docker Hub using credentials from .env.
        Args:
            output_dir: Directory containing the Dockerfile
            image_name: Name for the Docker image
            tag: Tag for the Docker image
        """
        docker_username = os.environ.get("DOCKER_USERNAME")
        docker_password = os.environ.get("DOCKER_PASSWORD")
        if not docker_username or not docker_password:
            logger.error("Docker Hub credentials not found in .env file.")
            return
        full_image_name = f"{docker_username}/{image_name}:{tag}"
        build_cmd = [
            "docker", "build", "-t", full_image_name, output_dir
        ]
        logger.info(f"Building Docker image: {' '.join(build_cmd)}")
        subprocess.run(build_cmd, check=True)
        login_cmd = ["docker", "login", "-u", docker_username, "-p", docker_password]
        logger.info("Logging in to Docker Hub...")
        subprocess.run(login_cmd, check=True)
        push_cmd = ["docker", "push", full_image_name]
        logger.info(f"Pushing Docker image: {full_image_name}")
        subprocess.run(push_cmd, check=True)
        logger.info(f"Docker image pushed to Docker Hub: {full_image_name}")

    def run_container_and_report(self, output_dir: str, image_name: str = "mydockerapp", tag: str = "latest", report_path: str = None, cleanup: bool = True, port: int = 5000):
        """
        Build, run the Docker container, and output container details as JSON.
        Args:
            output_dir: Directory containing the Dockerfile
            image_name: Name for the Docker image
            tag: Tag for the Docker image
            report_path: Optional path to save the JSON report
            cleanup: Whether to stop and remove the container after reporting
            port: Host port to map to container's exposed port
        Returns:
            Dictionary with container details
        """
        from pathlib import Path
        import time
        client = docker.from_env()
        docker_username = os.environ.get("DOCKER_USERNAME")
        full_image_name = f"{docker_username}/{image_name}:{tag}"
        build_cmd = ["docker", "build", "-t", full_image_name, str(Path(output_dir))]
        logger.info(f"Building Docker image: {' '.join(build_cmd)}")
        try:
            subprocess.run(build_cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Docker build failed: {e}")
            return {"error": "Docker build failed", "details": str(e)}
        dockerfile_path = Path(output_dir) / "Dockerfile"
        container_port = str(port)
        if dockerfile_path.exists():
            with open(dockerfile_path, "r") as f:
                for line in f:
                    if line.strip().startswith("EXPOSE"):
                        parts = line.strip().split()
                        if len(parts) == 2 and parts[1].isdigit():
                            container_port = parts[1]
                            break
        volume_name = f"{image_name}_data"
        network_name = f"{image_name}_net"
        try:
            client.volumes.create(name=volume_name)
        except Exception:
            pass  # Volume may already exist
        try:
            client.networks.create(name=network_name, driver="bridge")
        except Exception:
            pass  
        logger.info(f"Running container from image: {full_image_name} (host port {port} -> container port {container_port})")
        try:
            container = client.containers.run(
                full_image_name,
                detach=True,
                ports={f"{container_port}/tcp": int(container_port)},
                volumes={volume_name: {'bind': '/app/data', 'mode': 'rw'}},
                network=network_name,
                remove=False  # We'll handle cleanup
            )
        except Exception as e:
            logger.error(f"Failed to run container: {e}")
            return {"error": "Failed to run container", "details": str(e)}
        # Wait for health check or exit (max 30s)
        health_status = None
        for _ in range(30):
            container.reload()
            state = container.attrs.get("State", {})
            health = container.attrs.get("State", {}).get("Health", {})
            health_status = health.get("Status")
            if state.get("Status") == "exited":
                break
            if health_status in ("healthy", "unhealthy"):
                break
            time.sleep(1)
        # Gather container details
        container.reload()
        details = {
            "id": container.id,
            "name": container.name,
            "status": container.status,
            "image": container.image.tags,
            "ports": container.attrs.get("NetworkSettings", {}).get("Ports", {}),
            "created": container.attrs.get("Created"),
            "state": container.attrs.get("State", {}),
            "health": container.attrs.get("State", {}).get("Health", {}),
        }
        logs = container.logs().decode(errors="replace")
        if health_status == "unhealthy" or details["state"].get("Status") == "exited":
            details["logs"] = logs
            logger.warning("Container is unhealthy or exited. Logs:\n" + logs)
           
            if "ModuleNotFoundError: No module named 'uvicorn'" in logs:
                details["error"] = (
                    "The container failed because the 'uvicorn' module is not installed. "
                    "Make sure your requirements.txt includes 'uvicorn' and that your Dockerfile installs it correctly. "
                    "If you are using a non-root user, install with --prefix=/app/.local and set ENV PATH accordingly."
                )
                logger.error(details["error"])
        # Optionally save to file
        if report_path:
            with open(report_path, "w") as f:
                json.dump(details, f, indent=2)
        logger.info(f"Container details: {json.dumps(details, indent=2)}")
        # Cleanup if requested
        if cleanup:
            try:
                container.stop(timeout=5)
                container.remove()
                logger.info(f"Container {container.name} stopped and removed.")
            except Exception as e:
                logger.warning(f"Failed to cleanup container: {e}")
        return details

    def upload_report_to_s3(self, report_path: str, bucket_name: str, object_name: str = None):
        """
        Upload the container_report.json to an AWS S3 bucket.
        Args:
            report_path: Local path to the report file
            bucket_name: Name of the S3 bucket
            object_name: S3 object name (defaults to filename)
        """
        if object_name is None:
            object_name = Path(report_path).name
        s3 = boto3.client('s3')
        try:
            s3.upload_file(report_path, bucket_name, object_name)
            logger.info(f"Uploaded {report_path} to s3://{bucket_name}/{object_name}")
        except Exception as e:
            logger.error(f"Failed to upload report to S3: {e}")

from langchain.memory import ConversationBufferMemory

class SimpleAIAgent:
    """A simple AI agent that can reason, plan, and act using an LLM and tools."""
    def __init__(self, llm):
        self.llm = llm
        self.tools = self._get_tools()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def _get_tools(self):
        @tool
        def list_files(path: str) -> str:
            """List files in a directory."""
            try:
                return "\n".join(os.listdir(path))
            except Exception as e:
                return str(e)
        @tool
        def run_shell(command: str) -> str:
            """Run a shell command and return its output."""
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
                return result.stdout if result.returncode == 0 else result.stderr
            except Exception as e:
                return str(e)
        @tool
        def list_docker_containers(input: str) -> str:
            """List running Docker containers (input is ignored, for compatibility)."""
            try:
                client = docker.from_env()
                containers = client.containers.list()
                if not containers:
                    return "No running containers."
                return "\n".join(f"{c.name} ({c.id[:12]}) - {c.status}" for c in containers)
            except Exception as e:
                return f"Docker error: {e}"
        @tool
        def stop_docker_container(input: str) -> str:
            """Stop a running Docker container by name (input should be the container name)."""
            try:
                client = docker.from_env()
                container = client.containers.get(input.strip())
                container.stop()
                return f"Stopped container: {input.strip()}"
            except Exception as e:
                return f"Docker error: {e}"
        return [list_files, run_shell, list_docker_containers, stop_docker_container]

    def run(self, task: str):
        """Run the agent on a given task."""
        return self.agent.run(task)

def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Generate Docker container applications from user requirements")
    parser.add_argument("query", nargs="?", help="The user requirements for the Docker container")
    parser.add_argument("--model", default="openai/gpt-4.1", help="The OpenAI model to use")
    parser.add_argument("--temperature", type=float, default=0.1, help="The temperature setting for the model")
    parser.add_argument("--output-dir", default="docker_output", help="Directory to save the generated files")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--run-and-report", action="store_true", help="Build, run container, and output JSON report of container details")
    parser.add_argument("--no-cleanup", action="store_true", help="Do not stop/remove the container after reporting")
    parser.add_argument("--port", type=int, default=5000, help="Host port to map to container's exposed port (default: 5000)")
    parser.add_argument("--s3-bucket", type=str, help="S3 bucket to upload container_report.json")
    parser.add_argument("--s3-object", type=str, help="S3 object name for the report (optional)")
    parser.add_argument("--agent-task", type=str, help="Run an AI agent on a given task (e.g., 'List files in my_output')")
    parser.add_argument("--agent-chat", action="store_true", help="Start a conversational chatbot session with the AI agent")
    
    args = parser.parse_args()
    
    try:
        agent = DockerAgent(
            model_name=args.model,
            temperature=args.temperature,
            verbose=args.verbose
        )
        
        if args.interactive:
            print("Docker Container Generator Agent (Interactive Mode)")
            print("Type 'exit' to quit")
            print("-" * 50)
            
            while True:
                query = input("\nEnter your requirements: ")
                if query.lower() == "exit":
                    break
                
                output = agent.process_query(query)
                output_path = agent.save_output(output, args.output_dir)
                
                print(f"\nDocker container files generated in: {output_path}")
                print("\nSetup Instructions:")
                print("-" * 50)
                print(output["setup_instructions"])
        else:
            if not args.query and not args.agent_task:
                parser.print_help()
                sys.exit(1)
            
            if args.agent_task:
                print("\nRunning AI Agent on task:", args.agent_task)
                agent = SimpleAIAgent(agent.llm)
                result = agent.run(args.agent_task)
                print("\nAgent Result:\n", result)
                return

            if args.agent_chat:
                print("\nConversational Chatbot Mode (type 'exit' to quit)")
                agent = SimpleAIAgent(agent.llm)
                while True:
                    user_input = input("You: ")
                    if user_input.strip().lower() in ("exit", "quit"): break
                    response = agent.run(user_input)
                    print("Agent:", response)
                return

            output = agent.process_query(args.query)
            output_path = agent.save_output(output, args.output_dir)
            
            print(f"Docker container files generated in: {output_path}")
            print("\nSetup Instructions:")
            print("-" * 50)
            print(output["setup_instructions"])
            try:
                agent.build_and_push_docker_image(str(output_path), image_name="flask-redis-app", tag="latest")
            except Exception as e:
                logger.error(f"Docker build/push failed: {str(e)}")
            if args.run_and_report:
                try:
                    report_path = str(Path(args.output_dir) / "container_report.json")
                    details = agent.run_container_and_report(
                        str(output_path),
                        image_name="flask-redis-app",
                        tag="latest",
                        report_path=report_path,
                        cleanup=not args.no_cleanup,
                        port=args.port
                    )
                    print(f"\nContainer details (JSON):\n{json.dumps(details, indent=2)}")
                    print(f"Container report saved to: {report_path}")
                    if args.s3_bucket:
                        agent.upload_report_to_s3(report_path, args.s3_bucket, args.s3_object)
                    if details.get("health", {}).get("Status") == "unhealthy":
                        print("WARNING: Container health check failed. See logs in the JSON report.")
                    if details.get("state", {}).get("Status") == "exited":
                        print("WARNING: Container exited. See logs in the JSON report.")
                except Exception as e:
                    logger.error(f"Container run/report failed: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
