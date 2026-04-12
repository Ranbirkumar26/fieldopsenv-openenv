FROM python:3.10-slim

LABEL maintainer="FieldOpsEnv"
LABEL description="Autonomous Field Robotics Task Environment — OpenEnv submission"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=https://api.openai.com/v1
ENV MODEL_NAME=gpt-4o-mini
ENV TASK_NAME=full_mission
ENV MAX_STEPS=50

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py    .
COPY env.py       .
COPY graders.py   .
COPY inference.py .
COPY openenv.yaml .
COPY server/      ./server/

RUN python -c "from env import FieldOpsEnv; from graders import TASK_GRADERS; print('Import check passed.')"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]