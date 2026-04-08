"""
env.py — FieldOpsEnv Core Environment
=======================================
Deterministic simulation of an autonomous field robotics mission.

Mission profile
---------------
An autonomous ground robot is deployed in a structured terrain zone to:
  1. Navigate from the base station to a designated resource deposit.
  2. Collect the mission-critical resource/sample.
  3. Return to the base station before operational energy is exhausted.

The environment is FULLY DETERMINISTIC:
  - Fixed 5×5 terrain grid
  - Fixed obstacle placements
  - Fixed resource and base coordinates
  - No stochastic elements

Grid encoding
-------------
  0 — clear terrain
  1 — obstacle / hazard zone (impassable)
  2 — resource / sample deposit
  3 — base station
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from models import Action, Observation, Reward

# ---------------------------------------------------------------------------
# Environment constants
# ---------------------------------------------------------------------------

GRID_SIZE: int = 5

BASE_POSITION: Tuple[int, int] = (0, 0)
RESOURCE_POSITION: Tuple[int, int] = (2, 2)

# 5×5 terrain map — fully deterministic, never mutated at runtime.
# Legend: 0=clear, 1=obstacle, 2=resource, 3=base
INITIAL_GRID: List[List[int]] = [
    [3, 0, 0, 1, 0],   # row 0
    [0, 1, 0, 0, 0],   # row 1
    [0, 0, 2, 0, 1],   # row 2
    [0, 1, 0, 0, 0],   # row 3
    [0, 0, 0, 1, 0],   # row 4
]

INITIAL_ENERGY: float = 100.0

# Energy costs per action type
ENERGY_COST: Dict[str, float] = {
    "up":      1.0,
    "down":    1.0,
    "left":    1.0,
    "right":   1.0,
    "stay":    0.5,
    "collect": 1.0,
}

# Movement deltas (row_delta, col_delta)
MOVEMENT_DELTA: Dict[str, Tuple[int, int]] = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
    "stay":  ( 0,  0),
}

# Reward magnitudes
R_STEP_PENALTY:    float = -0.2
R_APPROACH:        float =  2.0
R_RETREAT:         float = -1.0
R_COLLISION:       float = -5.0
R_COLLECT_SUCCESS: float =  50.0
R_COLLECT_INVALID: float = -2.0
R_MISSION_SUCCESS: float =  100.0
R_ENERGY_DEPLETED: float = -50.0


# ---------------------------------------------------------------------------
# Helper utilities (importable by graders and inference)
# ---------------------------------------------------------------------------

def compute_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Manhattan distance between two grid coordinates."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_target(
    has_resource: bool,
    resource_position: Tuple[int, int],
    base_position: Tuple[int, int],
) -> Tuple[int, int]:
    """
    Dynamic objective selector.
      - Phase 1 (no resource collected) → target is resource deposit.
      - Phase 2 (resource collected)    → target is base station.
    """
    return base_position if has_resource else resource_position


def render_grid(grid: List[List[int]], position: Tuple[int, int]) -> str:
    """
    Returns an ASCII visualisation of the terrain grid with the agent marker.

    Symbols
    -------
    @  — agent current position
    B  — base station
    R  — resource deposit
    X  — obstacle / hazard
    .  — clear terrain
    """
    symbols = {0: ".", 1: "X", 2: "R", 3: "B"}
    lines = []
    for r, row in enumerate(grid):
        row_str = ""
        for c, cell in enumerate(row):
            if (r, c) == position:
                row_str += "@ "
            else:
                row_str += symbols.get(cell, "?") + " "
        lines.append(row_str.rstrip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FieldOpsEnv
# ---------------------------------------------------------------------------

class FieldOpsEnv:
    """
    FieldOpsEnv — Autonomous Field Robotics Task Environment
    =========================================================
    Implements the OpenEnv interface:
        reset()          → Observation
        step(action)     → (Observation, Reward, done: bool, info: dict)
        state()          → Observation   (read-only snapshot)
    """

    def __init__(self) -> None:
        self._base_position: Tuple[int, int] = BASE_POSITION
        self._resource_position: Tuple[int, int] = RESOURCE_POSITION
        self._master_grid: List[List[int]] = [row[:] for row in INITIAL_GRID]

        # Mutable episode state — initialised properly by reset()
        self._position: Tuple[int, int] = BASE_POSITION
        self._grid: List[List[int]] = [row[:] for row in self._master_grid]
        self._energy: float = INITIAL_ENERGY
        self._has_resource: bool = False
        self._step_count: int = 0
        self._done: bool = False

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """
        Reset the environment to its deterministic initial state.
        Returns the opening Observation for the new episode.
        """
        self._position    = self._base_position
        self._grid        = [row[:] for row in self._master_grid]
        self._energy      = INITIAL_ENERGY
        self._has_resource = False
        self._step_count  = 0
        self._done        = False
        return self.state()

    def state(self) -> Observation:
        """Return an immutable snapshot of the current environment state."""
        return Observation(
            position=self._position,
            grid=[row[:] for row in self._grid],
            energy=self._energy,
            has_resource=self._has_resource,
            resource_position=self._resource_position,
            base_position=self._base_position,
            step_count=self._step_count,
        )

    def step(
        self, action: Action
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Advance the simulation by one decision step.

        Parameters
        ----------
        action : Action
            Agent command for this timestep.

        Returns
        -------
        observation : Observation
            Updated environmental state.
        reward : Reward
            Scored outcome with feedback breakdown.
        done : bool
            True when the episode has terminated (success or failure).
        info : dict
            Auxiliary diagnostics:
              distance_to_target, energy_remaining, collision, target
        """
        # Guard: episode already over
        if self._done:
            return (
                self.state(),
                Reward(score=0.0, feedback="Episode already terminated."),
                True,
                {"distance_to_target": 0, "energy_remaining": self._energy,
                 "collision": False, "target": self._base_position},
            )

        action_type   = action.action_type
        reward_score  = 0.0
        feedback      : List[str] = []
        collision     = False

        # Per-step operational cost
        reward_score += R_STEP_PENALTY
        feedback.append(f"Step cost: {R_STEP_PENALTY:.1f}")

        # Determine current objective and distance before action
        target       = get_target(self._has_resource, self._resource_position, self._base_position)
        prev_distance = compute_distance(self._position, target)

        # ----------------------------------------------------------------
        # Action dispatch
        # ----------------------------------------------------------------
        if action_type in MOVEMENT_DELTA:
            dr, dc = MOVEMENT_DELTA[action_type]
            new_row = self._position[0] + dr
            new_col = self._position[1] + dc

            if action_type == "stay":
                self._energy -= ENERGY_COST["stay"]
                feedback.append("Hold position.")
            else:
                # Boundary check
                out_of_bounds = not (0 <= new_row < GRID_SIZE and 0 <= new_col < GRID_SIZE)
                # Obstacle check
                hits_obstacle = (not out_of_bounds) and self._grid[new_row][new_col] == 1

                if out_of_bounds or hits_obstacle:
                    reason = "boundary" if out_of_bounds else "hazard zone"
                    reward_score += R_COLLISION
                    feedback.append(f"Collision ({reason}): {R_COLLISION:.1f}")
                    collision = True
                else:
                    # Valid movement — update position and energy
                    self._position = (new_row, new_col)
                    self._energy  -= ENERGY_COST[action_type]

                    new_distance = compute_distance(self._position, target)
                    if new_distance < prev_distance:
                        reward_score += R_APPROACH
                        feedback.append(f"Closing on objective: +{R_APPROACH:.1f}")
                    elif new_distance > prev_distance:
                        reward_score += R_RETREAT
                        feedback.append(f"Diverging from objective: {R_RETREAT:.1f}")

        elif action_type == "collect":
            self._energy -= ENERGY_COST["collect"]

            at_resource   = self._position == self._resource_position
            already_has   = self._has_resource

            if at_resource and not already_has:
                self._has_resource = True
                # Clear resource marker from grid
                self._grid[self._resource_position[0]][self._resource_position[1]] = 0
                reward_score += R_COLLECT_SUCCESS
                feedback.append(f"Resource collected: +{R_COLLECT_SUCCESS:.1f}")
            else:
                reason = "already collected" if already_has else "not at resource location"
                reward_score += R_COLLECT_INVALID
                feedback.append(f"Invalid collect ({reason}): {R_COLLECT_INVALID:.1f}")

        self._step_count += 1

        # ----------------------------------------------------------------
        # Termination checks (evaluated after action)
        # ----------------------------------------------------------------

        # Energy depletion
        if self._energy <= 0:
            self._energy = 0.0
            reward_score += R_ENERGY_DEPLETED
            feedback.append(f"Energy depleted — mission abort: {R_ENERGY_DEPLETED:.1f}")
            self._done = True

        # Mission success: resource secured AND returned to base
        elif self._has_resource and self._position == self._base_position:
            reward_score += R_MISSION_SUCCESS
            feedback.append(f"Mission complete — resource returned to base: +{R_MISSION_SUCCESS:.1f}")
            self._done = True

        # ----------------------------------------------------------------
        # Build return values
        # ----------------------------------------------------------------
        target_after = get_target(self._has_resource, self._resource_position, self._base_position)
        info: Dict[str, Any] = {
            "distance_to_target": compute_distance(self._position, target_after),
            "energy_remaining":   self._energy,
            "collision":          collision,
            "target":             target_after,
        }

        return (
            self.state(),
            Reward(score=round(reward_score, 4), feedback=" | ".join(feedback)),
            self._done,
            info,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return ASCII terrain visualisation with agent position."""
        header = (
            f"  Step {self._step_count:>3d} | "
            f"Energy {self._energy:>6.1f} | "
            f"Resource {'SECURED' if self._has_resource else 'pending'}\n"
        )
        col_ids = "  " + " ".join(str(c) for c in range(GRID_SIZE))
        rows = []
        for r, row in enumerate(self._grid):
            row_str = f"{r} "
            for c, cell in enumerate(row):
                if (r, c) == self._position:
                    row_str += "@ "
                elif cell == 3:
                    row_str += "B "
                elif cell == 2:
                    row_str += "R "
                elif cell == 1:
                    row_str += "X "
                else:
                    row_str += ". "
            rows.append(row_str.rstrip())
        legend = "  Legend: @=agent  B=base  R=resource  X=hazard  .=clear"
        return header + col_ids + "\n" + "\n".join(rows) + "\n" + legend
