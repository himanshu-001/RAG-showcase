FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install system dependencies (curl for container healthchecks)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application files, configs, and initial data
COPY app ./app
COPY data ./data
COPY test ./test
COPY .streamlit ./.streamlit
COPY entrypoint.sh ./entrypoint.sh

# Ensure script is executable and ensure data/uploaded_sources exists
RUN chmod +x ./entrypoint.sh && \
    mkdir -p /app/data/uploaded_sources

# Expose FastAPI backend (8080) and Streamlit UI (8501) ports
EXPOSE 8080 8501

# Entrypoint script handles 'fastapi', 'streamlit', 'both', or custom CLI commands
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command runs both services concurrently
CMD ["both"]
