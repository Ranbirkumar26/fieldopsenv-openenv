"""
graders.py — FieldOpsEnv Task Graders
=======================================
Three deterministic evaluation tasks, each returning a normalised score
STRICTLY in (0.001, 0.999) — never 0.0 or 1.0.

Tasks
-----
1. navigation        (easy)   — reach the resource deposit from base.
2. hazard_navigation (medium) — reach the resource with zero/minimal collisions.
3. full_mission      (hard)   — collect the resource and return to base.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from env import FieldOpsEnv, compute_distance, get_target
from models import Action, Observation


# ---------------------------------------------------------------------------
# Shared greedy policy used by all graders
# ---------------------------------------------------------------------------

def _greedy_action(obs: Observation) -> str:
    target = get_target(obs.has_resource, obs.resource_position, obs.base_position)

    if obs.position == obs.resource_position and not obs.has_resource:
        return "collect"

    row, col   = obs.position
    trow, tcol = target

    primary: list[Tuple[str, Tuple[int, int]]] = []
    if trow < row:
        primary.append(("up",    (row - 1, col)))
    elif trow > row:
        primary.append(("down",  (row + 1, col)))

    if tcol < col:
        primary.append(("left",  (row, col - 1)))
    elif tcol > col:
        primary.append(("right", (row, col + 1)))

    for action, (nr, nc) in primary:
        if _passable(nr, nc, obs.grid):
            return action

    for action, (dr, dc) in [("up", (-1, 0)), ("down", (1, 0)),
                              ("left", (0, -1)), ("right", (0, 1))]:
        nr, nc = row + dr, col + dc
        if _passable(nr, nc, obs.grid):
            return action

    return "stay"


def _passable(row: int, col: int, grid: list) -> bool:
    return (
        0 <= row < len(grid)
        and 0 <= col < len(grid[0])
        and grid[row][col] != 1
    )


def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0.001, 0.999)."""
    return round(max(0.001, min(0.999, score)), 4)


# ---------------------------------------------------------------------------
# Task 1 — navigation (easy)
# ---------------------------------------------------------------------------

def grade_navigation(env: FieldOpsEnv, max_steps: int = 20) -> float:
    """
    Navigate the agent from base station to the resource deposit.

    Scoring:
      distance_score   (60%) — proximity to resource at episode end.
      efficiency_score (30%) — steps used vs max allowed.
      collision_penalty(10%) — penalises collisions.

    Success floor: 0.72 (deliberately below 1.0 to stay in open range).
    Returns float strictly in (0.001, 0.999).
    """
    obs = env.reset()
    steps      = 0
    collisions = 0
    reached    = False

    for _ in range(max_steps):
        if obs.position == obs.resource_position:
            reached = True
            break
        if obs.energy <= 0:
            break

        action_type = _greedy_action(obs)
        obs, reward, done, info = env.step(Action(action_type=action_type))
        steps += 1

        if info.get("collision"):
            collisions += 1
        if done:
            break

    max_dist       = compute_distance(obs.base_position, obs.resource_position)
    final_dist     = compute_distance(obs.position, obs.resource_position)
    distance_score = max(0.0, 1.0 - final_dist / max(1, max_dist))

    # Efficiency: reward using fewer steps; avoid hitting exactly 1.0
    steps_ratio      = steps / max(1, max_steps)
    efficiency_score = max(0.0, 0.95 - steps_ratio)

    collision_penalty = min(0.9, collisions * 0.15)

    raw = (
        0.60 * distance_score
        + 0.30 * efficiency_score
        - 0.10 * collision_penalty
    )

    # Success floor capped at 0.72 to stay well below 1.0
    if reached:
        raw = max(raw, 0.72)

    return _clamp(raw)


# ---------------------------------------------------------------------------
# Task 2 — hazard_navigation (medium)
# ---------------------------------------------------------------------------

def grade_hazard_navigation(env: FieldOpsEnv, max_steps: int = 30) -> float:
    """
    Reach the resource deposit while minimising hazard collisions and
    conserving energy.

    Scoring:
      distance_score  (40%) — proximity to resource.
      collision_score (40%) — penalises each collision.
      energy_score    (20%) — fraction of energy remaining (scaled to 0.95 max).

    Success floors: 0.82 (clean), 0.62 (with collisions) — below 1.0.
    Returns float strictly in (0.001, 0.999).
    """
    obs            = env.reset()
    initial_energy = obs.energy
    steps          = 0
    collisions     = 0
    reached        = False

    for _ in range(max_steps):
        if obs.position == obs.resource_position:
            reached = True
            break
        if obs.energy <= 0:
            break

        action_type = _greedy_action(obs)
        obs, reward, done, info = env.step(Action(action_type=action_type))
        steps += 1

        if info.get("collision"):
            collisions += 1
        if done:
            break

    max_dist       = compute_distance(obs.base_position, obs.resource_position)
    final_dist     = compute_distance(obs.position, obs.resource_position)
    distance_score = max(0.0, 1.0 - final_dist / max(1, max_dist))
    collision_score = max(0.0, 1.0 - collisions * 0.20)
    # Scale energy score to max 0.95 so weighted sum never reaches 1.0
    energy_score   = 0.95 * (obs.energy / max(0.001, initial_energy))

    raw = (
        0.40 * distance_score
        + 0.40 * collision_score
        + 0.20 * energy_score
    )

    if reached and collisions == 0:
        raw = max(raw, 0.82)   # clean navigation bonus (below 1.0)
    elif reached:
        raw = max(raw, 0.62)

    return _clamp(raw)


# ---------------------------------------------------------------------------
# Task 3 — full_mission (hard)
# ---------------------------------------------------------------------------

def grade_full_mission(env: FieldOpsEnv, max_steps: int = 50) -> float:
    """
    Complete the full autonomous field mission:
      Phase 1 → navigate to resource and collect sample.
      Phase 2 → return to base station with resource secured.

    Scoring:
      Mission success: base 0.75 + efficiency bonus (up to 0.10)
                       + energy bonus (up to 0.08) − collision penalty.
      Total max ≈ 0.93, well below 1.0.
      Partial (resource collected, not returned): up to 0.65.
      Partial (not collected): distance-based up to 0.30.

    Returns float strictly in (0.001, 0.999).
    """
    obs              = env.reset()
    initial_energy   = obs.energy
    steps            = 0
    collisions       = 0
    mission_success  = False
    resource_secured = False

    for _ in range(max_steps):
        if obs.energy <= 0:
            break

        action_type = _greedy_action(obs)
        obs, reward, done, info = env.step(Action(action_type=action_type))
        steps += 1

        if info.get("collision"):
            collisions += 1
        if obs.has_resource:
            resource_secured = True
        if done:
            if obs.has_resource and obs.position == obs.base_position:
                mission_success = True
            break

    if mission_success:
        # Base 0.75 — bonuses capped so total never reaches 1.0
        efficiency_bonus  = 0.10 * max(0.0, 1.0 - steps / max(1, max_steps))
        energy_bonus      = 0.08 * (obs.energy / max(0.001, initial_energy))
        collision_penalty = min(0.20, collisions * 0.05)
        score = 0.75 + efficiency_bonus + energy_bonus - collision_penalty

    elif resource_secured:
        dist_to_base      = compute_distance(obs.position, obs.base_position)
        max_base_dist     = 8
        return_progress   = max(0.0, 1.0 - dist_to_base / max_base_dist)
        collision_penalty = min(0.15, collisions * 0.03)
        score = 0.45 + 0.20 * return_progress - collision_penalty

    else:
        dist_to_resource = compute_distance(obs.position, obs.resource_position)
        max_res_dist     = compute_distance(obs.base_position, obs.resource_position)
        progress         = max(0.0, 1.0 - dist_to_resource / max(1, max_res_dist))
        score = 0.05 + 0.25 * progress   # range: [0.05, 0.30]

    return _clamp(score)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_GRADERS: Dict[str, Callable[[FieldOpsEnv], float]] = {
    "navigation":        grade_navigation,
    "hazard_navigation": grade_hazard_navigation,
    "full_mission":      grade_full_mission,
}

TASK_MAX_STEPS: Dict[str, int] = {
    "navigation":        20,
    "hazard_navigation": 30,
    "full_mission":      50,
}