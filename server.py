from fastapi import FastAPI
from env import FieldOpsEnv
from models import Action
import subprocess
import sys

app = FastAPI()

env = FieldOpsEnv()

@app.get("/")
def root():
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=60
        )

        return {
            "status": "running",
            "env": "FieldOpsEnv",
            "logs": result.stdout
        }
    except Exception as e:
        return {
            "status": "running",
            "env": "FieldOpsEnv",
            "error": str(e)
        }

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.model_dump()

@app.post("/step")
def step(action: dict):
    act = Action(**action)
    obs, reward, done, info = env.step(act)

    return {
        "observation": obs.model_dump(),
        "reward": {"score": reward.score},
        "done": done,
        "info": info
    }