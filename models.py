"""
models.py — FieldOpsEnv Data Models
=====================================
Pydantic schemas for all environment I/O:
  - Observation  : full sensor state delivered to the agent each step
  - Action       : discrete command issued by the agent
  - Reward       : scored outcome with human-readable feedback string
"""

from __future__ import annotations

from typing import List, Tuple
from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """
    Full environmental state returned to the agent after every step.

    Grid encoding:
        0 — clear terrain
        1 — terrain obstacle / hazard zone
        2 — mission resource / sample deposit
        3 — base station
    """

    position: Tuple[int, int]
    """Current (row, col) position of the robotic agent."""

    grid: List[List[int]]
    """5×5 terrain map as a nested list."""

    energy: float
    """Remaining operational energy (starts at 100.0)."""

    has_resource: bool
    """True once the agent has successfully collected the mission resource."""

    resource_position: Tuple[int, int]
    """Fixed (row, col) coordinates of the resource / sample deposit."""

    base_position: Tuple[int, int]
    """Fixed (row, col) coordinates of the base station."""

    step_count: int
    """Total decision steps taken in the current mission episode."""

    class Config:
        frozen = True   # immutable snapshot — no accidental mutation


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

VALID_ACTIONS = frozenset({"up", "down", "left", "right", "stay", "collect"})


class Action(BaseModel):
    """
    Discrete command issued by the agent each timestep.

    Allowed action_type values
    --------------------------
    up       — move one cell north  (row - 1)
    down     — move one cell south  (row + 1)
    left     — move one cell west   (col - 1)
    right    — move one cell east   (col + 1)
    stay     — hold position (costs reduced energy)
    collect  — attempt resource collection at current cell
    """

    action_type: str

    @field_validator("action_type")
    @classmethod
    def _validate_action(cls, v: str) -> str:
        if v not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action '{v}'. Must be one of: {sorted(VALID_ACTIONS)}"
            )
        return v


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Scored outcome returned alongside the next Observation.

    score    — signed float reward accumulated over one decision step
    feedback — pipe-delimited human-readable explanation of each reward term
    """

    score: float
    feedback: str
