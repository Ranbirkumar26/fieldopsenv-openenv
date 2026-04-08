from fastapi import FastAPI
from env import FieldOpsEnv
from models import Action

app = FastAPI()

env = FieldOpsEnv()

@app.get("/")
def root():
    return {"status": "running", "env": "FieldOpsEnv"}

@app.post("/reset")
def reset():
    try:
        obs = env.reset()
        return obs.model_dump()
    except Exception as e:
        return {"error": str(e)}

@app.post("/step")
def step(action: dict):
    try:
        act = Action(**action)
        obs, reward, done, info = env.step(act)

        return {
            "observation": obs.model_dump(),
            "reward": {"score": reward.score},  # safer format
            "done": done,
            "info": info
        }
    except Exception as e:
        return {"error": str(e)}