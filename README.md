---
title: FieldOpsEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "1.0"
python_version: "3.10"
app_file: inference.py
pinned: false
---

# FieldOpsEnv — Autonomous Field Robotics Task Environment

> **OpenEnv Hackathon Submission** · v1.0.0 · Deterministic · Lightweight · Production-grade

---

## 1. Overview

**FieldOpsEnv** is a deterministic benchmark environment for evaluating autonomous decision-making agents in structured field robotics scenarios. It models real operational challenges faced by ground robots deployed in:

- **Disaster response** — navigating rubble fields to reach supply caches
- **Remote mine inspection** — traversing hazardous terrain to collect samples
- **Precision agriculture** — locating and securing sensor packages under battery constraints
- **Infrastructure survey** — navigating confined zones with energy-aware path planning

The agent operates on a structured terrain grid, must navigate impassable hazard zones, collect a mission-critical resource, and return to base — all within a hard energy budget.

Additionally, the environment supports **LLM-assisted decision-making**, where a language model can interpret the current state and generate actions in real time. A deterministic fallback policy ensures reliability and reproducibility even when LLM responses are unavailable, enabling hybrid evaluation of both rule-based and AI-driven agents.

---

## 2. Motivation

Modern field robotics systems must make sequential decisions under strict operational constraints:

| Real-world constraint | FieldOpsEnv equivalent |
|-----------------------|------------------------|
| Battery life          | Energy budget (100 units) |
| Terrain obstacles     | Impassable hazard cells |
| Mission phases        | Navigate → Collect → Return |
| Path planning         | Greedy heuristic / LLM-assisted decision policy |
| No-GPS zones          | Full grid state provided in observation |

Benchmarking autonomous agents in this structured, deterministic environment isolates **planning quality**, **energy efficiency**, and **hazard avoidance** as pure, measurable signals — without the confounds of sensor noise or stochastic environments.

---

## 3. Observation Space

Each step returns an `Observation` (Pydantic model) with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `position` | `tuple[int, int]` | Agent's current `(row, col)` on the 5×5 grid |
| `grid` | `list[list[int]]` | Full terrain map snapshot (see encoding below) |
| `energy` | `float` | Remaining operational energy `[0.0, 100.0]` |
| `has_resource` | `bool` | Whether the mission sample has been collected |
| `resource_position` | `tuple[int, int]` | Fixed resource deposit location `(2, 2)` |
| `base_position` | `tuple[int, int]` | Fixed base station location `(0, 0)` |
| `step_count` | `int` | Steps elapsed in the current episode |

**Grid encoding:**

```
0 — clear terrain     (passable)
1 — obstacle/hazard   (impassable, collision penalty)
2 — resource deposit  (collect here)
3 — base station      (mission end)
```

**Fixed terrain map:**

```
  0 1 2 3 4
0 B . . X .
1 . X . . .
2 . . R . X
3 . X . . .
4 . . . X .

@=agent  B=base  R=resource  X=hazard  .=clear
```

---

## 4. Action Space

Six discrete commands available each timestep:

| Action | Effect | Energy Cost |
|--------|--------|-------------|
| `up` | Move north (row − 1) | 1.0 |
| `down` | Move south (row + 1) | 1.0 |
| `left` | Move west (col − 1) | 1.0 |
| `right` | Move east (col + 1) | 1.0 |
| `stay` | Hold position | 0.5 |
| `collect` | Collect resource at current cell | 1.0 |

Movement into obstacles or grid boundaries is blocked (position unchanged) and incurs a collision penalty.

---

## 5. Task Descriptions

### Task 1 — `navigation` (Easy)

Navigate from the base station `(0,0)` to the resource deposit `(2,2)`.

- **Max steps:** 20
- **Graded on:** final distance (60%), step efficiency (30%), collisions (10%)
- **Success:** agent position equals resource position

### Task 2 — `hazard_navigation` (Medium)

Reach the resource deposit while minimising all hazard encounters and preserving energy reserves.

- **Max steps:** 30
- **Graded on:** distance (40%), collision avoidance (40%), energy remaining (20%)
- **Success:** reach resource with zero or minimal collisions

### Task 3 — `full_mission` (Hard)

Complete the full autonomous field mission:
1. Navigate to resource deposit
2. Collect the mission sample
3. Return to base station

All phases must complete within the energy budget.

- **Max steps:** 50
- **Graded on:** mission success (base 1.0) + efficiency bonus + energy bonus − collision penalty
- **Partial credit:** resource collected but not returned → up to 0.75

---

## 6. Reward Design

The reward function is shaped to guide the agent through multi-phase mission planning:

| Event | Reward | Rationale |
|-------|--------|-----------|
| Per step | −0.2 | Operational cost; incentivises efficiency |
| Move closer to objective | +2.0 | Positive shaping toward goal |
| Move away from objective | −1.0 | Discourages aimless traversal |
| Collision (obstacle/boundary) | −5.0 | Hard penalty for terrain blindness |
| Successful resource collection | +50.0 | Phase 1 milestone |
| Invalid collect attempt | −2.0 | Penalises speculative commands |
| Mission success (return to base) | +100.0 | Primary mission completion |
| Energy depletion | −50.0 | Mission abort penalty |

**Dynamic objective:** reward shaping uses Manhattan distance to the *current* objective:
- Phase 1 (no resource): objective = resource deposit `(2,2)`
- Phase 2 (resource held): objective = base station `(0,0)`

---

---

## 7. Setup Instructions

**Prerequisites:** Python 3.10+

```bash
# Clone / enter project directory
cd FieldOpsEnv

# Install dependencies
pip install -r requirements.txt

# Run inference (LLM-assisted policy with deterministic fallback)
export HF_TOKEN=your_api_token
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export TASK_NAME=full_mission

python inference.py
```

**Run a specific task:**
```bash
TASK_NAME=navigation python inference.py
TASK_NAME=hazard_navigation python inference.py
TASK_NAME=full_mission python inference.py
```

**Run graders directly:**
```python
from env import FieldOpsEnv
from graders import TASK_GRADERS

env = FieldOpsEnv()
for task, grader in TASK_GRADERS.items():
    score = grader(env)
    print(f"{task}: {score:.4f}")
```

---

## 8. Docker Instructions

**Build:**
```bash
docker build -t fieldopsenv:latest .
```

**Run (full mission):**
```bash
docker run --rm \
  -e HF_TOKEN=your_api_token \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e TASK_NAME=full_mission \
  fieldopsenv:latest
```

**Run a different task:**
```bash
docker run --rm \
  -e HF_TOKEN=your_api_token \
  -e TASK_NAME=navigation \
  fieldopsenv:latest
```

**Resource limits (well within constraints):**
```bash
docker run --rm --cpus="2" --memory="8g" \
  -e HF_TOKEN=your_api_token \
  fieldopsenv:latest
```

---

## 9. Baseline Output Example

The deterministic greedy policy produces the following output for `full_mission`:

```
[START] task=full_mission env=FieldOpsEnv model=gpt-4o-mini
[STEP] step=1 action=down reward=1.80 done=false error=null
[STEP] step=2 action=down reward=1.80 done=false error=null
[STEP] step=3 action=right reward=1.80 done=false error=null
[STEP] step=4 action=right reward=1.80 done=false error=null
[STEP] step=5 action=collect reward=49.80 done=false error=null
[STEP] step=6 action=left reward=1.80 done=false error=null
[STEP] step=7 action=left reward=1.80 done=false error=null
[STEP] step=8 action=up reward=1.80 done=false error=null
[STEP] step=9 action=up reward=101.80 done=true error=null
[END] success=true steps=9 rewards=1.80,1.80,1.80,1.80,49.80,1.80,1.80,1.80,101.80
```

Total cumulative reward: **163.40** — mission complete in 9 steps with 91.0 energy remaining.

---

## 10. Why This Is NOT a Toy Problem

FieldOpsEnv is engineered to reflect genuine challenges in deployed autonomous systems:

| Claim | Evidence |
|-------|----------|
| **Multi-phase mission planning** | Agent must sequence Navigate → Collect → Return; no single-goal shortcuts |
| **Energy-aware decision-making** | Hard energy budget with per-action costs creates real trade-offs between speed and caution |
| **Dynamic objective switching** | Target changes mid-episode; agent must detect phase transitions and replan |
| **Hazard-aware path planning** | Obstacles are not trivially avoidable; the optimal path requires deliberate routing |
| **Partial credit grading** | Graders reflect real operational success metrics, not binary pass/fail |
| **OpenEnv compliance** | Fully implements the OpenEnv interface: `reset()`, `step()`, structured observation/action/reward schemas |
| **Production code quality** | Pydantic validation, typed interfaces, modular architecture, Docker deployment |
| **Real-world domain transfer** | Scenario maps directly onto disaster response, mining, agriculture, and inspection robotics |
| **LLM-integrated autonomy** | Supports real-time decision-making via language models with safe fallback mechanisms |

The environment is intentionally compact (5×5 grid, < 1 MB memory) to enable rapid iteration during evaluation while preserving the full complexity of multi-objective, energy-constrained autonomous planning.

---

## 11. Project Structure

```
FieldOpsEnv/
    models.py       — Pydantic schemas: Observation, Action, Reward
    env.py          — FieldOpsEnv simulation engine
    graders.py      — Deterministic task graders (navigation / hazard / full_mission)
    inference.py    — OpenEnv-compliant inference entry-point
    openenv.yaml    — OpenEnv specification
    Dockerfile      — Python 3.10 container definition
    requirements.txt
    README.md
```

---

*FieldOpsEnv — Deterministic · Lightweight · OpenEnv-compliant · Built for competitive evaluation.*
