from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from env import FieldOpsEnv
from models import Action
import subprocess
import sys

app = FastAPI()

env = FieldOpsEnv()

@app.get("/", response_class=HTMLResponse)
def root():
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=60
        )

        logs = result.stdout.replace("<", "&lt;").replace(">", "&gt;")

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FieldOpsEnv Dashboard</title>
            <style>
                body {{
                    background: #0f172a;
                    color: #e2e8f0;
                    font-family: 'Segoe UI', sans-serif;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    max-width: 1000px;
                    margin: auto;
                    padding: 20px;
                }}
                .header {{
                    text-align: center;
                    padding: 20px;
                    background: linear-gradient(90deg, #2563eb, #7c3aed);
                    border-radius: 10px;
                    margin-bottom: 20px;
                }}
                .status {{
                    margin-top: 10px;
                    font-size: 18px;
                    color: #4ade80;
                }}
                .logs {{
                    background: #020617;
                    padding: 20px;
                    border-radius: 10px;
                    white-space: pre-wrap;
                    border: 1px solid #1e293b;
                    overflow-x: auto;
                    box-shadow: 0 0 20px rgba(0,0,0,0.5);
                }}
                .badge {{
                    display: inline-block;
                    padding: 5px 10px;
                    background: #22c55e;
                    color: black;
                    border-radius: 5px;
                    font-weight: bold;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🚀 FieldOpsEnv Dashboard</h1>
                    <div class="status">
                        Status: <span class="badge">RUNNING</span>
                    </div>
                </div>

                <div class="logs">{logs}</div>
            </div>
        </body>
        </html>
        """

    except Exception as e:
        return f"<h1>Error</h1><pre>{str(e)}</pre>"

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
