# ──────────────────────────────────────────────────────────────────────────────
# FieldOpsEnv — Autonomous Field Robotics Task Environment
# Docker image
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# Metadata
LABEL maintainer="FieldOpsEnv"
LABEL description="Autonomous Field Robotics Task Environment — OpenEnv submission"
LABEL version="1.0.0"

# Prevent .pyc files and enable unbuffered stdout (critical for streaming logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ── Defaults (override at runtime via -e flags) ────────────────────────────────
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini
ENV TASK_NAME=full_mission
ENV MAX_STEPS=50
# HF_TOKEN must be supplied at runtime — no default

WORKDIR /app

# Install dependencies first (leverages Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY models.py    .
COPY env.py       .
COPY graders.py   .
COPY inference.py .
COPY openenv.yaml .

# Smoke-test: verify imports resolve (fails fast if dependencies are missing)
RUN python -c "from env import FieldOpsEnv; from graders import TASK_GRADERS; print('Import check passed.')"

# --- Entry-point -------------------------------------------------------------
CMD ["python", "inference.py"]
