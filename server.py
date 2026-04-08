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
        exit_code = result.returncode
        status_label = "Success" if exit_code == 0 else "Failed"
        status_color = "#22c55e" if exit_code == 0 else "#ef4444"
        status_bg = "rgba(34,197,94,0.1)" if exit_code == 0 else "rgba(239,68,68,0.1)"

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>FieldOps — Inference Dashboard</title>
    <style>
        *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

        body {{
            background: #0b0f1a;
            color: #cbd5e1;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            font-size: 14px;
            line-height: 1.6;
            min-height: 100vh;
        }}

        .layout {{
            max-width: 1040px;
            margin: 0 auto;
            padding: 40px 24px;
        }}

        /* Top bar */
        .topbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 32px;
            padding-bottom: 20px;
            border-bottom: 1px solid #1e293b;
        }}

        .topbar-left {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}

        .app-title {{
            font-size: 17px;
            font-weight: 600;
            color: #f1f5f9;
            letter-spacing: -0.01em;
        }}

        .app-subtitle {{
            font-size: 12px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background: {status_bg};
            color: {status_color};
            border: 1px solid {status_color}33;
        }}

        .status-dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: {status_color};
        }}

        /* Metrics row */
        .metrics {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 14px;
            margin-bottom: 28px;
        }}

        .metric-card {{
            background: #111827;
            border: 1px solid #1e293b;
            border-radius: 10px;
            padding: 16px 20px;
        }}

        .metric-label {{
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            margin-bottom: 6px;
        }}

        .metric-value {{
            font-size: 22px;
            font-weight: 600;
            color: #f1f5f9;
            font-variant-numeric: tabular-nums;
        }}

        .metric-value.ok {{ color: #22c55e; }}
        .metric-value.err {{ color: #ef4444; }}

        /* Log panel */
        .log-panel {{
            background: #060b14;
            border: 1px solid #1e293b;
            border-radius: 10px;
            overflow: hidden;
        }}

        .log-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 20px;
            border-bottom: 1px solid #1e293b;
            background: #0d1424;
        }}

        .log-header-title {{
            font-size: 12px;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }}

        .log-lines-count {{
            font-size: 11px;
            color: #475569;
        }}

        .log-body {{
            padding: 20px;
            overflow-x: auto;
            white-space: pre;
            font-family: 'Cascadia Code', 'Fira Code', 'Consolas', 'SF Mono', monospace;
            font-size: 13px;
            line-height: 1.75;
            color: #94a3b8;
            max-height: 540px;
            overflow-y: auto;
        }}

        .log-body::-webkit-scrollbar {{ width: 6px; height: 6px; }}
        .log-body::-webkit-scrollbar-track {{ background: transparent; }}
        .log-body::-webkit-scrollbar-thumb {{ background: #1e293b; border-radius: 3px; }}
    </style>
</head>
<body>
    <div class="layout">

        <div class="topbar">
            <div class="topbar-left">
                <span class="app-title">FieldOps Inference Dashboard</span>
                <span class="app-subtitle">Inference runtime monitor</span>
            </div>
            <div class="status-pill">
                <span class="status-dot"></span>
                {status_label}
            </div>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Exit Code</div>
                <div class="metric-value {'ok' if exit_code == 0 else 'err'}">{exit_code}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Output Lines</div>
                <div class="metric-value">{len(result.stdout.splitlines())}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Stderr Lines</div>
                <div class="metric-value {'err' if result.stderr else ''}">{len(result.stderr.splitlines())}</div>
            </div>
        </div>

        <div class="log-panel">
            <div class="log-header">
                <span class="log-header-title">Stdout</span>
                <span class="log-lines-count">{len(result.stdout.splitlines())} lines</span>
            </div>
            <div class="log-body">{logs if logs.strip() else '(no output)'}</div>
        </div>

    </div>
</body>
</html>"""

    except Exception as e:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <title>FieldOps — Error</title>
    <style>
        body {{ background: #0b0f1a; color: #cbd5e1; font-family: system-ui, sans-serif; padding: 40px 24px; }}
        .err {{ background: #1a0a0a; border: 1px solid #7f1d1d; border-radius: 10px; padding: 24px; color: #fca5a5; font-family: monospace; font-size: 13px; }}
        h2 {{ color: #ef4444; font-size: 14px; margin-bottom: 12px; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="err"><h2>Runtime Error</h2>{str(e)}</div>
</body>
</html>"""


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