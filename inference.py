"""
inference.py — FieldOpsEnv Inference Entry-Point
==================================================
Runs one complete mission episode and prints results in the exact
OpenEnv evaluation format.

Environment variables
---------------------
API_BASE_URL   Base URL for the OpenAI-compatible API endpoint.
               Default: https://api.openai.com/v1
MODEL_NAME     Model identifier used for logging and optional LLM calls.
               Default: gpt-4o-mini
HF_TOKEN       API authentication token. REQUIRED — no default.
TASK_NAME      One of: navigation | hazard_navigation | full_mission
               Default: full_mission
MAX_STEPS      Episode step cap (integer).
               Default: derived from task registry.

Output format (strict)
----------------------
[START] task=<task_name> env=<env_name> model=<model_name>
[STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...>
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
import sys
from typing import List

from openai import OpenAI

from env import FieldOpsEnv, compute_distance, get_target
from graders import TASK_MAX_STEPS
from models import Action, Observation

# Load environment variables from .env file (if present)
load_dotenv()

# ---------------------------------------------------------------------------
# LLM-assisted policy (with safe fallback)
# ---------------------------------------------------------------------------

def build_prompt(obs: Observation) -> str:
    return f"""
You are controlling an autonomous field robot.

Goal:
- If resource not collected → go to resource at {obs.resource_position}
- If resource collected → return to base at {obs.base_position}

Constraints:
- Avoid obstacles (cells with value 1)
- Stay within grid
- Minimize steps and energy

Current State:
Position: {obs.position}
Energy: {obs.energy}
Has Resource: {obs.has_resource}
Grid: {obs.grid}

Respond with ONLY ONE word:
up / down / left / right / stay / collect
"""

def get_llm_action(client, obs: Observation) -> str | None:
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": build_prompt(obs)}],
            max_tokens=5,
            temperature=0
        )

        action = response.choices[0].message.content.strip().lower()
        valid = {"up","down","left","right","stay","collect"}

        if action in valid:
            return action
    except Exception:
        pass

    return None

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
TASK_NAME:    str = os.environ.get("TASK_NAME",    "full_mission")
ENV_NAME:     str = "FieldOpsEnv"

# HF_TOKEN is mandatory — raise immediately if absent
try:
    HF_TOKEN: str = os.environ["HF_TOKEN"]
except KeyError:
    sys.stderr.write("FATAL: HF_TOKEN environment variable is required.\n")
    sys.exit(1)

MAX_STEPS: int = int(os.environ.get("MAX_STEPS", str(TASK_MAX_STEPS.get(TASK_NAME, 50))))

# Validate task name
_VALID_TASKS = {"navigation", "hazard_navigation", "full_mission"}
if TASK_NAME not in _VALID_TASKS:
    sys.stderr.write(
        f"FATAL: TASK_NAME '{TASK_NAME}' is invalid. "
        f"Choose from: {sorted(_VALID_TASKS)}\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Deterministic greedy policy
# ---------------------------------------------------------------------------

def _passable(row: int, col: int, grid: list) -> bool:
    """True if (row, col) is within bounds and not an obstacle."""
    return (
        0 <= row < len(grid)
        and 0 <= col < len(grid[0])
        and grid[row][col] != 1
    )


def greedy_action(obs: Observation) -> str:
    """
    Deterministic greedy policy — no randomness, no LLM dependency.

    Strategy
    --------
    Phase 1 (resource not collected):
      Move toward resource deposit; issue "collect" when adjacent.
    Phase 2 (resource collected):
      Move toward base station.

    Movement preference: primary axis toward target → secondary axis →
    any unblocked cardinal direction → "stay" as last resort.
    """
    target = get_target(obs.has_resource, obs.resource_position, obs.base_position)

    # Issue collect when standing on the resource
    if obs.position == obs.resource_position and not obs.has_resource:
        return "collect"

    row, col   = obs.position
    trow, tcol = target

    # Primary-axis candidates toward target
    candidates: list = []
    if trow < row:
        candidates.append(("up",    (row - 1, col)))
    elif trow > row:
        candidates.append(("down",  (row + 1, col)))

    if tcol < col:
        candidates.append(("left",  (row, col - 1)))
    elif tcol > col:
        candidates.append(("right", (row, col + 1)))

    for action, (nr, nc) in candidates:
        if _passable(nr, nc, obs.grid):
            return action

    # Fall back to any passable cardinal direction
    for action, (dr, dc) in [("up", (-1, 0)), ("down", (1, 0)),
                              ("left", (0, -1)), ("right", (0, 1))]:
        nr, nc = row + dr, col + dc
        if _passable(nr, nc, obs.grid):
            return action

    return "stay"


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    # Initialise OpenAI-compatible client
    # The LLM client is instantiated here per spec; the baseline policy is
    # deterministic greedy.  LLM integration can be swapped in by replacing
    # the greedy_action() call below with a prompt-based call to `client`.
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=HF_TOKEN,
    )

    env = FieldOpsEnv()
    obs = env.reset()

    # ── START line ──────────────────────────────────────────────────────────
    print(f"[START] task={TASK_NAME} env={ENV_NAME} model={MODEL_NAME}")

    step          = 0
    rewards: List[float] = []
    done          = False
    success       = False

    while not done and step < MAX_STEPS:
        error_msg   = "null"
        action_str  = "stay"
        reward_val  = 0.0

        try:
            llm_action = get_llm_action(client, obs)

            if llm_action is not None:
                action_str = llm_action
            else:
                action_str = greedy_action(obs)

            action     = Action(action_type=action_str)
            obs, reward, done, info = env.step(action)
            reward_val = reward.score
            rewards.append(reward_val)

            if done and obs.has_resource and obs.position == obs.base_position:
                success = True

        except Exception as exc:
            error_msg = str(exc).replace("\n", " ").strip()
            rewards.append(0.0)
            done = True

        step += 1

        # ── STEP line ────────────────────────────────────────────────────────
        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={reward_val:.2f} "
            f"done={'true' if done else 'false'} "
            f"error={error_msg}"
        )

    # ── END line ─────────────────────────────────────────────────────────────
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step} "
        f"rewards={rewards_str}"
    )


if __name__ == "__main__":
    main()
    import sys
    sys.exit(0)
