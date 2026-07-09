# ── Build stage ──────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies and gcc so pip can compile packages that have C extensions
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from build stage to runtime stage
COPY --from=builder /install /usr/local

# Copying source code and model
COPY src/ ./src/
COPY models/ ./models/

# Creating non-root user for security
RUN useradd --create-home --uid 1000 appuser \
 && chown -R appuser:appuser /app

 # Switch to non-root user for all subsequent commands and the CMD
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check — Kubernetes will also use /health endpoint
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=20s \
    --retries=3 \
    CMD python -c "\
import urllib.request, sys; \
res = urllib.request.urlopen('http://localhost:8000/health'); \
sys.exit(0 if res.status == 200 else 1)"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENV=production

# This command runs when the container starts and runs FastAPI with uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]