### Step-by-step Setup Instructions

1. **Clone or create your project directory** and place the following files in it:
   - `Dockerfile` (as above)
   - `requirements.txt` (as above)
   - `main.py` (as above)

2. **Build the Docker image:**
   ```sh
   docker build -t python-fastapi-dev .
   ```

3. **Run the container with source code mounted for live reload:**
   ```sh
   docker run --rm -it -p 8000:8000 \
     -v $(pwd):/app \
     python-fastapi-dev
   ```
   This mounts your current directory into the container, enabling code reload on changes.

4. **Test the application:**
   - Open [http://localhost:8000/](http://localhost:8000/) in your browser. You should see `{"message": "Hello, World!"}`.
   - Healthcheck endpoint: [http://localhost:8000/health](http://localhost:8000/health)

5. **Develop:**
   - Edit `main.py` or add more files. The server will reload automatically.

6. **Stop the container:**
   - Press `Ctrl+C` in the terminal where the container is running.
