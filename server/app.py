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

# REQUIRED for OpenEnv
def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()