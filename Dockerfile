# WirelessAgent - Green Agent Docker Image
# UC Berkeley AgentX Competition Submission
# AgentBeats Compatible
# Author: Jingwen
# Date: 1/13/2026

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src:/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml .
COPY requirements.txt .

# Install Python dependencies
RUN uv pip install --system -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY benchmarks/ ./benchmarks/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs

# Expose port for A2A protocol (AgentBeats default: 9009)
EXPOSE 9009

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9009/health || exit 1

# Default command - run the green agent server
CMD ["python", "src/server.py"]
