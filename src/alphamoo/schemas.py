"""
AlphaMoo v4.1 — Core data schemas.

All dataclasses that flow between modules. Verbatim field names from the
ARC-AGI-3 JSONL spec (verified against 25 ground-truth replays).

Reference: /home/z/my-project/download/AlphaMoo_v4.1_Delta.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

# =============================================================================
# Environment primitives — verbatim from ARC-AGI-3 docs
# =============================================================================

GRID_SIZE = 64
MAX_COLOR = 15  # cells are integers 0-15


class ActionId(IntEnum):
    """The 7 standardized actions. Each game declares a subset."""
    RESET = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    INTERACT = 5  # game-defined: rotate, select, attach, etc.
    CLICK = 6     # complex — requires (x, y) coordinates
    UNDO = 7      # always undo for games that support it


class GameState(str):
    """Game state enum. Values are verbatim strings from the API."""
    NOT_FINISHED = "NOT_FINISHED"
    WIN = "WIN"
    GAME_OVER = "GAME_OVER"


# =============================================================================
# Frame / Action / Record — direct mirrors of the JSONL schema
# =============================================================================

@dataclass(frozen=True)
class ActionInput:
    """
    An action emitted by the agent. Verbatim schema from replays.

    For ActionId.CLICK (id=6), `data` contains {"x": int, "y": int}.
    For all other actions, `data` contains {"game_id": str}.
    `reasoning` is always None in reference replays; our agent populates it
    for debugging (does not count as an action per RHAE rules).
    """
    id: int
    data: dict
    reasoning: str | None = None

    @property
    def is_click(self) -> bool:
        return self.id == ActionId.CLICK

    @property
    def click_coords(self) -> tuple[int, int] | None:
        if self.is_click and "x" in self.data and "y" in self.data:
            return (int(self.data["x"]), int(self.data["y"]))
        return None

    @property
    def is_reset(self) -> bool:
        return self.id == ActionId.RESET


@dataclass(frozen=True)
class FrameRecord:
    """
    One line of an ARC-AGI-3 replay JSONL file.

    `frame` is a 3D list: [N][64][64] of ints 0-15.
    N varies per record — most are 1, but cascades (animations, level
    transitions, gravity) can produce N up to 37 in observed data.

    The final frame in `frame` is the steady state. Intermediate frames
    reveal what changed during the cascade.
    """
    timestamp: str
    game_id: str
    frame: list[list[list[int]]]  # [N][64][64]
    state: str                     # GameState value
    levels_completed: int
    win_levels: int
    action_input: ActionInput
    guid: str
    full_reset: bool
    available_actions: list[int]

    @property
    def n_subframes(self) -> int:
        """Number of frames in this record's cascade."""
        return len(self.frame)

    @property
    def final_grid(self) -> list[list[int]]:
        """The steady-state grid (last subframe)."""
        return self.frame[-1]

    @property
    def is_terminal(self) -> bool:
        return self.state in (GameState.WIN, GameState.GAME_OVER)


@dataclass(frozen=True)
class LevelSummary:
    """Per-level completion info from the replay's summary record."""
    level: int
    cumulative_actions: int  # total actions taken to reach this level


@dataclass(frozen=True)
class GameSummary:
    """Final-line summary record from a replay."""
    game_id: str
    total_plays: int
    guids: list[str]
    final_levels_completed: list[int]
    final_states: list[str]
    total_actions_per_play: list[int]
    actions_by_level: list[list[LevelSummary]]
    resets_per_play: list[int]
    total_actions: int


@dataclass
class Replay:
    """A complete loaded replay — multiple FrameRecords + summary."""
    game_id: str
    guid: str
    records: list[FrameRecord]
    summary: GameSummary

    @property
    def n_actions(self) -> int:
        """Number of actions taken (excluding the initial RESET record)."""
        return len(self.records) - 1

    @property
    def n_levels(self) -> int:
        return self.records[0].win_levels

    @property
    def available_actions(self) -> list[int]:
        return self.records[0].available_actions

    @property
    def won(self) -> bool:
        return self.records[-1].state == GameState.WIN

    def actions_by_level(self) -> dict[int, int]:
        """Maps level number → number of actions taken on that level."""
        result = {}
        prev_level = 0
        prev_cumulative = 0
        for record in self.records[1:]:
            curr_level = record.levels_completed
            if curr_level > prev_level:
                # level just completed
                result[prev_level + 1] = self._action_count_so_far(record) - prev_cumulative
                for level_num in range(prev_level + 2, curr_level + 1):
                    result[level_num] = 0  # skipped levels (shouldn't happen but safe)
                prev_level = curr_level
                prev_cumulative = self._action_count_so_far(record)
        # actions on the final (incomplete or just-completed) level
        if prev_level < self.n_levels or self.won:
            result[prev_level + 1] = self.n_actions - prev_cumulative
        return result

    def _action_count_so_far(self, record: FrameRecord) -> int:
        """How many actions have been taken up to and including this record."""
        return self.records.index(record)


# =============================================================================
# Perception — what the agent sees after parsing a frame
# =============================================================================

@dataclass
class GameObject:
    """
    One connected component in the scene graph.

    Note: the agent itself may be a composite object (multiple colors).
    `is_agent` is set by the Agent State Tracker, not by Perception.
    """
    id: str
    color: int                       # primary color (most common)
    secondary_colors: list[int]      # additional colors if multi-colored
    cells: list[tuple[int, int]]     # all (x, y) positions
    bounding_box: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    topology: Literal["solid", "hollow", "fragmented"]
    shape_hash: str                  # rotation/translation-invariant signature
    is_agent: bool = False


@dataclass
class SceneGraph:
    """Symbolic representation of one frame."""
    objects: dict[str, GameObject]
    edges: set[tuple[str, str, str]]  # (src_id, dst_id, relation)
    agent_id: str | None = None
    grid: list[list[int]] | None = None  # raw grid for debugging
    hash: str = ""                    # for revisit detection


# =============================================================================
# Cascade interpretation — events between subframes
# =============================================================================

@dataclass
class CascadeEvent:
    """One discrete change observed between two consecutive subframes."""
    type: Literal[
        "appearance",        # new object appeared
        "disappearance",     # object vanished
        "move",              # object changed position
        "color_change",      # object changed color
        "shape_change",      # object's shape transformed
        "level_transition",  # level advanced
    ]
    target_id: str | None = None
    target_color: int | None = None
    before: dict = field(default_factory=dict)
    after: dict = field(default_factory=dict)
    timestep: int = 0  # which subframe index this event occurred at


# =============================================================================
# Working memory — what the agent holds between actions
# =============================================================================

@dataclass
class Condition:
    """One condition in a conjunctive trigger."""
    predicate: str  # agent_has | agent_touches | agent_adjacent | ...
    args: dict
    negated: bool = False


@dataclass
class Trigger:
    """A hypothesis trigger — conjunction of conditions, optional temporal spec."""
    conditions: list[Condition]
    temporal: dict | None = None  # e.g. {"type": "consecutive", "turns": 3}


@dataclass
class Hypothesis:
    """A causal mechanics hypothesis: IF trigger THEN effect."""
    trigger: Trigger
    effect: dict  # {type: ..., args: ...}
    confidence: float = 0.0
    support: int = 0
    mdl_cost: int = 0


@dataclass
class GoalHypothesis:
    """A hypothesis about the win condition."""
    terminal_condition: str  # predicate name
    args: dict
    outcome: Literal["WIN", "LOSE"] = "WIN"
    confidence: float = 0.0
    support: int = 0
    near_miss_support: int = 0


@dataclass
class AgentState:
    """
    Mutable agent state, tracked across timesteps.

    The agent can transform — color, shape, AND orientation can change.
    Multiple Object IDs may compose the agent (composite agent case).
    """
    object_ids: list[str] = field(default_factory=list)
    position: tuple[int, int] = (0, 0)
    orientation: int = 0  # 0-3 (or 0-7 for 8-way)
    shape: str = ""
    color: list[int] = field(default_factory=list)
    energy: int | None = None
    inventory: list[str] = field(default_factory=list)


# =============================================================================
# Compute profile — Phase 0 output
# =============================================================================

@dataclass
class ComputeProfile:
    """Measured compute envelope. Phase 0 output."""
    wall_clock_per_step: float = 0.0
    wall_clock_per_episode: float = 0.0
    peak_gpu_memory_gb: float = 0.0
    tokens_per_step: int = 0
    llm_calls_per_step: int = 0
    eval_time_budget_seconds: int = 32400  # 9 hours
    max_steps_per_episode: int = 0
    affordable_model_size: str = "0.5B"
    affordable_ig_tier: str = "symbolic_only"


# =============================================================================
# ARC-AGI-3 official 16-color palette (RGB values)
# Source: arcprize.org media kit / standard ARC palette
# =============================================================================

ARC_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),           # black
    1: (0, 116, 217),       # blue
    2: (255, 65, 54),       # red
    3: (46, 204, 64),       # green
    4: (255, 220, 0),       # yellow
    5: (170, 170, 170),     # gray
    6: (240, 18, 190),      # magenta
    7: (255, 133, 27),      # orange
    8: (127, 219, 255),     # light blue
    9: (135, 12, 37),       # dark red / purple
    10: (189, 211, 254),    # light pink
    11: (160, 95, 168),     # purple
    12: (110, 80, 45),      # brown
    13: (75, 75, 75),       # dark gray
    14: (200, 200, 200),    # light gray
    15: (255, 255, 255),    # white
}
