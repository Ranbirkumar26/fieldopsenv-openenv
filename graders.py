"""
graders.py — FieldOpsEnv Task Graders
=======================================
Three deterministic evaluation tasks, each returning a normalised score
in [0.0, 1.0].

Tasks
-----
1. navigation        (easy)   — reach the resource deposit from base.
2. hazard_navigation (medium) — reach the resource with zero/minimal collisions.
3. full_mission      (hard)   — collect the resource and return to base.

All graders are fully deterministic: given the same environment and policy
they always return the same score.  No random elements exist.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from env import FieldOpsEnv, compute_distance, get_target
from models import Action, Observation


# ---------------------------------------------------------------------------
# Shared greedy policy used by all graders
# ---------------------------------------------------------------------------

def _greedy_action(obs: Observation) -> str:
    """
    Deterministic greedy policy.

    Decision logic
    --------------
    1. If at resource and resource not yet collected → "collect".
    2. Otherwise compute the current objective (resource or base) and move
       one step closer along the preferred axis.
    3. If the preferred move is blocked, try orthogonal moves.
    4. If all moves are blocked → "stay".
    """
    target = get_target(obs.has_resource, obs.resource_position, obs.base_position)

    # Collect opportunity
    if obs.position == obs.resource_position and not obs.has_resource:
        return "collect"

    row, col   = obs.position
    trow, tcol = target

    # Prefer primary-axis moves (row then col)
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

    # Fall back: try all four cardinal directions
    for action, (dr, dc) in [("up", (-1, 0)), ("down", (1, 0)),
                              ("left", (0, -1)), ("right", (0, 1))]:
        nr, nc = row + dr, col + dc
        if _passable(nr, nc, obs.grid):
            return action

    return "stay"


def _passable(row: int, col: int, grid: list) -> bool:
    """True if (row, col) is within the grid and not an obstacle."""
    return (
        0 <= row < len(grid)
        and 0 <= col < len(grid[0])
        and grid[row][col] != 1
    )


# ---------------------------------------------------------------------------
# Task 1 — navigation (easy)
# ---------------------------------------------------------------------------

def grade_navigation(env: FieldOpsEnv, max_steps: int = 20) -> float:
    """
    Objective
    ---------
    Navigate the agent from the base station to the resource deposit.

    Scoring components
    ------------------
    distance_score   (60%) — how close the agent reached the resource.
    efficiency_score (30%) — steps used relative to the maximum allowed.
    collision_penalty(10%) — collisions reduce the final score.

    Returns
    -------
    float in [0.0, 1.0]
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

    # --- component scores ---
    max_dist       = compute_distance(obs.base_position, obs.resource_position)
    final_dist     = compute_distance(obs.position, obs.resource_position)
    distance_score = max(0.0, 1.0 - final_dist / max(1, max_dist))

    efficiency_score  = max(0.0, 1.0 - steps / max_steps) if steps < max_steps else 0.0
    collision_penalty = min(1.0, collisions * 0.15)

    score = (
        0.60 * distance_score
        + 0.30 * efficiency_score
        - 0.10 * collision_penalty
    )
    if reached:
        score = max(score, 0.65)   # floor for successful navigation

    return round(max(0.001, min(0.999, score)), 4)


# ---------------------------------------------------------------------------
# Task 2 — hazard_navigation (medium)
# ---------------------------------------------------------------------------

def grade_hazard_navigation(env: FieldOpsEnv, max_steps: int = 30) -> float:
    """
    Objective
    ---------
    Reach the resource deposit from base while minimising hazard encounters
    and preserving operational energy.

    Scoring components
    ------------------
    distance_score   (40%) — proximity to the resource upon termination.
    collision_score  (40%) — penalises every hazard/boundary collision.
    energy_score     (20%) — fraction of energy remaining.

    Returns
    -------
    float in [0.0, 1.0]
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

    # --- component scores ---
    max_dist       = compute_distance(obs.base_position, obs.resource_position)
    final_dist     = compute_distance(obs.position, obs.resource_position)
    distance_score = max(0.0, 1.0 - final_dist / max(1, max_dist))
    collision_score= max(0.0, 1.0 - collisions * 0.20)
    energy_score   = obs.energy / initial_energy

    score = (
        0.40 * distance_score
        + 0.40 * collision_score
        + 0.20 * energy_score
    )
    if reached and collisions == 0:
        score = max(score, 0.85)   # bonus for clean navigation
    elif reached:
        score = max(score, 0.60)

    return round(max(0.001, min(0.999, score)), 4)


# ---------------------------------------------------------------------------
# Task 3 — full_mission (hard)
# ---------------------------------------------------------------------------

def grade_full_mission(env: FieldOpsEnv, max_steps: int = 50) -> float:
    """
    Objective
    ---------
    Complete the full autonomous field mission:
      Phase 1 → navigate to resource deposit and collect sample.
      Phase 2 → return to base station with resource secured.

    Scoring components
    ------------------
    Mission success: base score 1.0 + efficiency/energy bonuses − penalties.
    Partial credit:
      - resource collected but not returned → up to 0.75
      - neither collected → distance-based partial up to 0.35

    Returns
    -------
    float in [0.0, 1.0]
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

    # --- scoring ---
    if mission_success:
        efficiency_bonus  = 0.15 * max(0.0, 1.0 - steps / max_steps)
        energy_bonus      = 0.10 * (obs.energy / initial_energy)
        collision_penalty = min(0.25, collisions * 0.05)
        score = 0.9 + efficiency_bonus + energy_bonus - collision_penalty

    elif resource_secured:
        # Partial: resource collected but not returned
        dist_to_base = compute_distance(obs.position, obs.base_position)
        max_base_dist = 8   # worst-case Manhattan in a 5×5 grid
        return_progress = max(0.0, 1.0 - dist_to_base / max_base_dist)
        collision_penalty = min(0.20, collisions * 0.04)
        score = 0.50 + 0.25 * return_progress - collision_penalty

    else:
        # Partial: heading toward resource but did not collect
        dist_to_resource = compute_distance(obs.position, obs.resource_position)
        max_res_dist     = compute_distance(obs.base_position, obs.resource_position)
        progress         = max(0.0, 1.0 - dist_to_resource / max(1, max_res_dist))
        score = 0.35 * progress

    return round(max(0.001, min(0.999, score)), 4)


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
