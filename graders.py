"""
graders.py — FieldOpsEnv Task Graders
=======================================
Three deterministic evaluation tasks, each returning a normalised score
STRICTLY in (0.001, 0.999).

Tasks
-----
1. navigation        (easy)
2. hazard_navigation (medium)
3. full_mission      (hard)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from env import FieldOpsEnv, compute_distance, get_target
from models import Action, Observation


def _passable(row: int, col: int, grid: list) -> bool:
    return (
        0 <= row < len(grid)
        and 0 <= col < len(grid[0])
        and grid[row][col] != 1
    )


def _greedy_action(obs: Observation) -> str:
    target = get_target(obs.has_resource, obs.resource_position, obs.base_position)
    if obs.position == obs.resource_position and not obs.has_resource:
        return "collect"

    row, col = obs.position
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


def _clamp(score: float) -> float:
    return round(max(0.001, min(0.999, score)), 4)


def grade_navigation(env: Optional[FieldOpsEnv] = None, max_steps: int = 20) -> float:
    if env is None:
        env = FieldOpsEnv()
    obs = env.reset()
    steps, collisions, reached = 0, 0, False

    for _ in range(max_steps):
        if obs.position == obs.resource_position:
            reached = True
            break
        if obs.energy <= 0:
            break
        obs, reward, done, info = env.step(Action(action_type=_greedy_action(obs)))
        steps += 1
        if info.get("collision"):
            collisions += 1
        if done:
            break

    max_dist = compute_distance(obs.base_position, obs.resource_position)
    final_dist = compute_distance(obs.position, obs.resource_position)
    distance_score = max(0.0, 1.0 - final_dist / max(1, max_dist))
    efficiency_score = max(0.0, 0.95 - steps / max(1, max_steps))
    collision_penalty = min(0.9, collisions * 0.15)

    raw = 0.60 * distance_score + 0.30 * efficiency_score - 0.10 * collision_penalty
    if reached:
        raw = max(raw, 0.72)
    return _clamp(raw)


def grade_hazard_navigation(env: Optional[FieldOpsEnv] = None, max_steps: int = 30) -> float:
    if env is None:
        env = FieldOpsEnv()
    obs = env.reset()
    initial_energy = obs.energy
    steps, collisions, reached = 0, 0, False

    for _ in range(max_steps):
        if obs.position == obs.resource_position:
            reached = True
            break
        if obs.energy <= 0:
            break
        obs, reward, done, info = env.step(Action(action_type=_greedy_action(obs)))
        steps += 1
        if info.get("collision"):
            collisions += 1
        if done:
            break

    max_dist = compute_distance(obs.base_position, obs.resource_position)
    final_dist = compute_distance(obs.position, obs.resource_position)
    distance_score = max(0.0, 1.0 - final_dist / max(1, max_dist))
    collision_score = max(0.0, 1.0 - collisions * 0.20)
    energy_score = 0.95 * (obs.energy / max(0.001, initial_energy))

    raw = 0.40 * distance_score + 0.40 * collision_score + 0.20 * energy_score
    if reached and collisions == 0:
        raw = max(raw, 0.82)
    elif reached:
        raw = max(raw, 0.62)
    return _clamp(raw)


def grade_full_mission(env: Optional[FieldOpsEnv] = None, max_steps: int = 50) -> float:
    if env is None:
        env = FieldOpsEnv()
    obs = env.reset()
    initial_energy = obs.energy
    steps, collisions = 0, 0
    mission_success = resource_secured = False

    for _ in range(max_steps):
        if obs.energy <= 0:
            break
        obs, reward, done, info = env.step(Action(action_type=_greedy_action(obs)))
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
        efficiency_bonus = 0.10 * max(0.0, 1.0 - steps / max(1, max_steps))
        energy_bonus = 0.08 * (obs.energy / max(0.001, initial_energy))
        collision_penalty = min(0.20, collisions * 0.05)
        score = 0.75 + efficiency_bonus + energy_bonus - collision_penalty
    elif resource_secured:
        dist_to_base = compute_distance(obs.position, obs.base_position)
        return_progress = max(0.0, 1.0 - dist_to_base / 8)
        collision_penalty = min(0.15, collisions * 0.03)
        score = 0.45 + 0.20 * return_progress - collision_penalty
    else:
        dist_to_resource = compute_distance(obs.position, obs.resource_position)
        max_res_dist = compute_distance(obs.base_position, obs.resource_position)
        progress = max(0.0, 1.0 - dist_to_resource / max(1, max_res_dist))
        score = 0.05 + 0.25 * progress

    return _clamp(score)


TASK_GRADERS: Dict[str, Callable] = {
    "navigation":        grade_navigation,
    "hazard_navigation": grade_hazard_navigation,
    "full_mission":      grade_full_mission,
}

TASK_MAX_STEPS: Dict[str, int] = {
    "navigation":        20,
    "hazard_navigation": 30,
    "full_mission":      50,
}