from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from env import FieldOpsEnv
from models import Action
from graders import TASK_GRADERS, TASK_MAX_STEPS
import subprocess
import sys

app = FastAPI(
    title="FieldOpsEnv",
    description="Autonomous Field Robotics Task Environment — OpenEnv submission",
    version="1.0.0",
)

env = FieldOpsEnv()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    return {
        "name": "FieldOpsEnv",
        "description": "Deterministic autonomous field robotics task environment with navigation, hazard avoidance, and full mission planning.",
    }


@app.get("/schema")
def schema():
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right", "stay", "collect"],
                }
            },
            "required": ["action_type"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "position": {"type": "array"},
                "grid": {"type": "array"},
                "energy": {"type": "number"},
                "has_resource": {"type": "boolean"},
                "resource_position": {"type": "array"},
                "base_position": {"type": "array"},
                "step_count": {"type": "integer"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "position": {"type": "array"},
                "grid": {"type": "array"},
                "energy": {"type": "number"},
                "has_resource": {"type": "boolean"},
            },
        },
    }


@app.get("/state")
def state():
    return env.state().model_dump()


@app.get("/tasks")
def tasks():
    results = []
    for task_id, grader in TASK_GRADERS.items():
        try:
            score = grader()
            results.append({
                "id": task_id,
                "score": score,
                "max_steps": TASK_MAX_STEPS.get(task_id),
            })
        except Exception as e:
            results.append({"id": task_id, "error": str(e)})
    return {"tasks": results}


@app.get("/", response_class=HTMLResponse)
def root():
    try:
        result = subprocess.run(
            [sys.executable, "inference.py"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        logs = result.stdout.replace("<", "&lt;").replace(">", "&gt;")
        exit_code = result.returncode
        status_label = "Success" if exit_code == 0 else "Failed"
        status_color = "#22c55e" if exit_code == 0 else "#ef4444"
        status_bg = "rgba(34,197,94,0.1)" if exit_code == 0 else "rgba(239,68,68,0.1)"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>FieldOps — Inference Dashboard</title>
    <style>
        body {{ background:#0b0f1a; color:#cbd5e1; font-family:'Segoe UI',system-ui,sans-serif; font-size:14px; line-height:1.6; }}
        .layout {{ max-width:1040px; margin:0 auto; padding:40px 24px; }}
        .topbar {{ display:flex; align-items:center; justify-content:space-between; margin-bottom:32px; padding-bottom:20px; border-bottom:1px solid #1e293b; }}
        .app-title {{ font-size:17px; font-weight:600; color:#f1f5f9; }}
        .app-subtitle {{ font-size:12px; color:#64748b; text-transform:uppercase; letter-spacing:.06em; }}
        .status-pill {{ display:inline-flex; align-items:center; gap:7px; padding:6px 14px; border-radius:20px; font-size:12px; font-weight:600; text-transform:uppercase; background:{status_bg}; color:{status_color}; border:1px solid {status_color}33; }}
        .status-dot {{ width:7px; height:7px; border-radius:50%; background:{status_color}; }}
        .metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:28px; }}
        .metric-card {{ background:#111827; border:1px solid #1e293b; border-radius:10px; padding:16px 20px; }}
        .metric-label {{ font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:.07em; margin-bottom:6px; }}
        .metric-value {{ font-size:22px; font-weight:600; color:#f1f5f9; }}
        .metric-value.ok {{ color:#22c55e; }} .metric-value.err {{ color:#ef4444; }}
        .log-panel {{ background:#060b14; border:1px solid #1e293b; border-radius:10px; overflow:hidden; }}
        .log-header {{ display:flex; align-items:center; justify-content:space-between; padding:12px 20px; border-bottom:1px solid #1e293b; background:#0d1424; }}
        .log-header-title {{ font-size:12px; font-weight:600; color:#94a3b8; text-transform:uppercase; letter-spacing:.07em; }}
        .log-body {{ padding:20px; overflow-x:auto; white-space:pre; font-family:monospace; font-size:13px; line-height:1.75; color:#94a3b8; max-height:540px; overflow-y:auto; }}
    </style>
</head>
<body>
<div class="layout">
    <div class="topbar">
        <div><div class="app-title">FieldOps Inference Dashboard</div><div class="app-subtitle">Inference runtime monitor</div></div>
        <div class="status-pill"><span class="status-dot"></span>{status_label}</div>
    </div>
    <div class="metrics">
        <div class="metric-card"><div class="metric-label">Exit Code</div><div class="metric-value {'ok' if exit_code==0 else 'err'}">{exit_code}</div></div>
        <div class="metric-card"><div class="metric-label">Output Lines</div><div class="metric-value">{len(result.stdout.splitlines())}</div></div>
        <div class="metric-card"><div class="metric-label">Stderr Lines</div><div class="metric-value {'err' if result.stderr else ''}">{len(result.stderr.splitlines())}</div></div>
    </div>
    <div class="log-panel">
        <div class="log-header"><span class="log-header-title">Stdout</span><span>{len(result.stdout.splitlines())} lines</span></div>
        <div class="log-body">{logs if logs.strip() else '(no output)'}</div>
    </div>
</div>
</body>
</html>"""
    except Exception as e:
        return f"<html><body style='background:#0b0f1a;color:#fca5a5;padding:40px;font-family:monospace'><h2>Runtime Error</h2>{str(e)}</body></html>"


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
        "info": info,
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()