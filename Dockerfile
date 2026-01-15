# WirelessAgent - Green Agent Docker Image
# UC Berkeley AgentX Competition Submission
# AgentBeats Compatible
# 
# Docker Image: ghcr.io/jwentong/wirelessagent-r2:latest
# Author: Jingwen
# Date: 1/15/2026
#
# AgentBeats Requirements:
# - ENTRYPOINT must accept: --host, --port, --card-url
# - Must be built for linux/amd64 architecture

FROM python:3.11-slim

# Metadata
LABEL org.opencontainers.image.title="WirelessAgent"
LABEL org.opencontainers.image.description="Green Agent for WCHW Benchmark - UC Berkeley AgentX Competition"
LABEL org.opencontainers.image.authors="Jingwen Tong <jwentong@foxmail.com>"
LABEL org.opencontainers.image.source="https://github.com/jwentong/WirelessAgent-R2"
LABEL org.opencontainers.image.licenses="MIT"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src:/app

# Assessment mode: "test" (100 problems) or "validate" (349 problems)
# Can be overridden at runtime: docker run -e ASSESSMENT_MODE=validate ...
ENV ASSESSMENT_MODE=test

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (for better layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/datasets/ ./data/datasets/
COPY benchmarks/ ./benchmarks/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p /app/logs

# Expose port for A2A protocol (AgentBeats default: 9009)
EXPOSE 9009

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:9009/health || exit 1

# AgentBeats ENTRYPOINT requirement:
# Must accept --host, --port, --card-url arguments
ENTRYPOINT ["python", "src/server.py"]

# Default arguments (can be overridden)
CMD ["--host", "0.0.0.0", "--port", "9009"]
