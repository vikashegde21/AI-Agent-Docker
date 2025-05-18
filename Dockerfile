FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefix=/app/.local -r requirements.txt \
    && rm -rf /root/.cache/pip
ENV PATH="/app/.local/bin:$PATH"
ENV PYTHONPATH="/app/.local/lib/python3.11/site-packages"
COPY . /app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1
CMD ["python", "main.py"]
