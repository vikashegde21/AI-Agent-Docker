# Use the official Python base image for development
FROM python:3.11-slim as base

# Install curl for healthchecks and diagnostics
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Create a non-root user 'appuser' with home directory /app
RUN useradd --create-home --home-dir /app appuser

# Set working directory
WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt ./

# Install Python dependencies to /app/.local
RUN pip install --upgrade pip && pip install --prefix=/app/.local -r requirements.txt && chown -R appuser:appuser /app/.local
ENV PYTHONPATH="/app/.local/lib/python3.13/site-packages"

# Copy application code and set permissions
COPY . .
RUN chown -R appuser:appuser /app

# Set environment variables
ENV PATH="/app/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Expose the application port
EXPOSE 8000

# Healthcheck: check if the app is running and ready
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

# Switch to non-root user
USER appuser

# Run the app with code reload for development
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]