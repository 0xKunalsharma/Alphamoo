"""
AlphaMoo v4.1 — Near-Miss Tracker (Module 7).

Extracts goal signal from episodes that end in LOSE. Without this, goal
inference has no training signal during early levels where the agent
dies constantly.

Definition of a near-miss: A predicate that became *truer* over the
episode's trajectory, even though the episode ended in LOSE.

Example:
    Episode ends in LOSE, but:
      - distance_to_exit decreased monotonically → suggests reach-exit goal
      - color_match_count increased → suggests state-matching goal
      - enemies_remaining decreased → suggests elimination goal

These are weak evidence for goal hypotheses (used by Module 6).

Default progress predicates (from v4.1 spec):
    distance_to_exit (minimize)
    gold_collected_count (maximize)
    enemies_remaining (minimize)
    agent_health (maximize)
    rooms_explored (maximize)
    puzzle_pieces_in_place (maximize)
    color_match_count (maximize)  — LS20-style
    shape_match_count (maximize)  — LS20-style
    orientation_match_count (maximize)  — LS20-style
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .schemas import GRID_SIZE, AgentState, SceneGraph

# =============================================================================
# Progress predicate definition
# =============================================================================

class ProgressDirection(StrEnum):
    """Direction of progress for a predicate."""
    MINIMIZE = "min"  # lower is better (e.g. distance_to_exit)
    MAXIMIZE = "max"  # higher is better (e.g. gold_collected)


@dataclass
class ProgressPredicate:
    """
    A predicate tracked across an episode. Records its value at each
    timestep. If the trajectory shows progress (in the right direction)
    before a LOSE, it's a near-miss.
    """
    name: str
    direction: ProgressDirection
    trajectory: list[float] = field(default_factory=list)

    def record(self, value: float) -> None:
        """Record the predicate's value at the current timestep."""
        self.trajectory.append(value)

    def is_near_miss(self) -> bool:
        """
        True if the predicate showed progress (in the right direction)
        before the episode ended.

        A near-miss requires:
          - At least 3 observations
          - The best value (min or max) is better than the first value
          - The trajectory was monotonic in the right direction for at
            least the last 2 steps
        """
        if len(self.trajectory) < 3:
            return False
        first = self.trajectory[0]
        if self.direction == ProgressDirection.MINIMIZE:
            best = min(self.trajectory)
            progress = best < first
            # Check last 2 steps are non-increasing
            recent_monotonic = self.trajectory[-1] <= self.trajectory[-2]
        else:
            best = max(self.trajectory)
            progress = best > first
            recent_monotonic = self.trajectory[-1] >= self.trajectory[-2]
        return progress and recent_monotonic

    def total_progress(self) -> float:
        """Total progress made over the trajectory (positive = good)."""
        if len(self.trajectory) < 2:
            return 0.0
        if self.direction == ProgressDirection.MINIMIZE:
            return self.trajectory[0] - self.trajectory[-1]
        return self.trajectory[-1] - self.trajectory[0]

    def reset(self) -> None:
        """Clear trajectory for a new episode."""
        self.trajectory.clear()


# =============================================================================
# Default progress predicates
# =============================================================================

def create_default_predicates() -> dict[str, ProgressPredicate]:
    """Create the default set of progress predicates."""
    return {
        "distance_to_exit": ProgressPredicate("distance_to_exit", ProgressDirection.MINIMIZE),
        "gold_collected_count": ProgressPredicate("gold_collected_count", ProgressDirection.MAXIMIZE),
        "enemies_remaining": ProgressPredicate("enemies_remaining", ProgressDirection.MINIMIZE),
        "rooms_explored": ProgressPredicate("rooms_explored", ProgressDirection.MAXIMIZE),
        "color_match_count": ProgressPredicate("color_match_count", ProgressDirection.MAXIMIZE),
        "shape_match_count": ProgressPredicate("shape_match_count", ProgressDirection.MAXIMIZE),
        "orientation_match_count": ProgressPredicate("orientation_match_count", ProgressDirection.MAXIMIZE),
        "objects_collected": ProgressPredicate("objects_collected", ProgressDirection.MAXIMIZE),
    }


# =============================================================================
# Progress predicate computation — measure from scene + agent state
# =============================================================================

def compute_distance_to_exit(scene: SceneGraph, agent_state: AgentState | None) -> float:
    """
    Estimate distance to the nearest "exit-like" object.
    Exit-like = the largest non-background object (heuristic).
    Returns a large number if no exit detected.
    """
    if agent_state is None or not agent_state.position:
        return float(GRID_SIZE * 2)
    # Find largest object
    if not scene.objects:
        return float(GRID_SIZE * 2)
    largest = max(scene.objects.values(), key=lambda o: len(o.cells))
    # Distance from agent to nearest cell of largest object
    ax, ay = agent_state.position
    min_dist = min(abs(ox - ax) + abs(oy - ay) for ox, oy in largest.cells)
    return float(min_dist)


def compute_color_match_count(
    scene: SceneGraph,
    agent_state: AgentState | None,
    target_colors: list[int] | None = None,
) -> float:
    """
    Count how many of the agent's colors match objects in the scene.
    Higher = more color matches (suggests state-matching goal).
    """
    if agent_state is None or not agent_state.color:
        return 0.0
    agent_colors = set(agent_state.color)
    if target_colors:
        # Count matches against specified target colors
        return float(len(agent_colors & set(target_colors)))
    # Count objects whose color matches the agent
    matches = sum(1 for obj in scene.objects.values() if obj.color in agent_colors)
    return float(matches)


def compute_shape_match_count(scene: SceneGraph, agent_state: AgentState | None) -> float:
    """Count objects whose shape hash matches the agent's shape hash."""
    if agent_state is None or not agent_state.shape:
        return 0.0
    return float(sum(1 for obj in scene.objects.values() if obj.shape_hash == agent_state.shape))


def compute_objects_collected(
    scene: SceneGraph,
    initial_object_count: int,
) -> float:
    """
    Count how many objects have been collected (disappeared) since
    the start of the episode. Higher = more collected.
    """
    return float(max(0, initial_object_count - len(scene.objects)))


def compute_enemies_remaining(scene: SceneGraph, agent_state: AgentState | None) -> float:
    """
    Estimate enemies remaining. Heuristic: count objects whose color differs
    from agent's color AND are smaller than the agent (likely threats).
    """
    if agent_state is None:
        return float(len(scene.objects))
    agent_colors = set(agent_state.color)
    # Crude heuristic: enemies = non-agent-colored, small-to-medium objects
    enemy_count = sum(
        1 for obj in scene.objects.values()
        if obj.color not in agent_colors and 2 <= len(obj.cells) <= 50
    )
    return float(enemy_count)


def compute_rooms_explored(scene: SceneGraph, agent_state: AgentState | None) -> float:
    """
    Estimate rooms explored. Heuristic: count distinct "regions" of background
    color the agent has visited. Simplified to grid quadrants.
    """
    if agent_state is None or not agent_state.position:
        return 0.0
    ax, ay = agent_state.position
    # Quadrant: 0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right
    return float((ax // 32) + (ay // 32) * 2)


# =============================================================================
# Near-Miss Tracker
# =============================================================================

class NearMissTracker:
    """
    Tracks progress predicates across an episode. On LOSE, identifies
    which predicates showed near-miss progress and provides them as
    weak evidence to the Goal Inference Module.

    Usage:
        tracker = NearMissTracker()
        # On each step:
        tracker.record_step(scene, agent_state)
        # On LOSE:
        near_misses = tracker.on_episode_end(outcome="GAME_OVER")
        # near_misses is a list of predicate names that showed progress
        goal_module.observe_lose(scene, agent_state, near_misses)
        # Reset for next episode
        tracker.reset()
    """

    def __init__(self, predicates: dict[str, ProgressPredicate] | None = None):
        self.predicates = predicates or create_default_predicates()
        self._initial_object_count: int = 0
        self._step_count: int = 0
        self._episode_count: int = 0
        self._near_miss_count: int = 0
        self._progress_history: list[dict[str, float]] = []  # per-episode progress

    def record_step(self, scene: SceneGraph, agent_state: AgentState | None) -> None:
        """
        Record progress predicate values at the current step.

        Args:
            scene: current SceneGraph
            agent_state: current AgentState
        """
        if self._step_count == 0:
            self._initial_object_count = len(scene.objects)

        # Compute and record each predicate
        self.predicates["distance_to_exit"].record(
            compute_distance_to_exit(scene, agent_state)
        )
        self.predicates["gold_collected_count"].record(
            compute_objects_collected(scene, self._initial_object_count)
        )
        self.predicates["enemies_remaining"].record(
            compute_enemies_remaining(scene, agent_state)
        )
        self.predicates["rooms_explored"].record(
            compute_rooms_explored(scene, agent_state)
        )
        self.predicates["color_match_count"].record(
            compute_color_match_count(scene, agent_state)
        )
        self.predicates["shape_match_count"].record(
            compute_shape_match_count(scene, agent_state)
        )
        # Some predicates share computation
        self.predicates["orientation_match_count"].record(
            compute_shape_match_count(scene, agent_state)  # proxy
        )
        self.predicates["objects_collected"].record(
            compute_objects_collected(scene, self._initial_object_count)
        )

        self._step_count += 1

    def on_episode_end(self, outcome: str) -> list[str]:
        """
        Called when an episode ends. If LOSE, identify near-miss predicates.

        Args:
            outcome: "WIN" or "GAME_OVER"

        Returns:
            List of predicate names that showed near-miss progress.
            Empty if outcome is WIN (no near-miss on success).
        """
        self._episode_count += 1
        if outcome == "WIN":
            # No near-miss analysis needed on WIN
            return []

        near_misses: list[str] = []
        episode_progress: dict[str, float] = {}

        for name, pred in self.predicates.items():
            episode_progress[name] = pred.total_progress()
            if pred.is_near_miss():
                near_misses.append(name)

        self._progress_history.append(episode_progress)
        self._near_miss_count += len(near_misses)
        return near_misses

    def reset(self) -> None:
        """Reset predicates for a new episode."""
        for pred in self.predicates.values():
            pred.reset()
        self._step_count = 0
        self._initial_object_count = 0

    def get_stats(self) -> dict:
        return {
            "step_count": self._step_count,
            "episode_count": self._episode_count,
            "near_miss_count": self._near_miss_count,
            "avg_near_misses_per_lose": (
                self._near_miss_count / max(1, self._episode_count)
            ),
            "current_trajectory_lengths": {
                name: len(pred.trajectory)
                for name, pred in self.predicates.items()
            },
        }

    def get_progress_summary(self) -> dict[str, float]:
        """Get total progress for each predicate in the current episode."""
        return {name: pred.total_progress() for name, pred in self.predicates.items()}
