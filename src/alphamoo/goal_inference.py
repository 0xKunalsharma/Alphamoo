"""
AlphaMoo v4.1 — Goal Inference Module (Module 6).

A separate hypothesis layer operating over episode-level outcomes,
not step-level transitions. While Module 5 (Hypothesis Generator)
discovers *mechanics* ("what happens when..."), Module 6 infers
*win conditions* ("what does winning look like?").

How it works:
  1. Starts with a prior over common ARC-AGI-3 goal types
     (reach_exit, collect_all, eliminate_threats, state_matching, etc.)
  2. Each WIN observation updates goal posteriors via Bayesian inference
     — what was true about the scene right before the win?
  3. LOSE observations are weaker evidence (via Near-Miss Tracker, Module 7)
  4. The top goal hypothesis gates the Planner (Module 11) — planning only
     activates when goal confidence ≥ 0.6

Goal hypothesis form:
    terminal_condition: scene_graph_predicate
    outcome: WIN | LOSE
    confidence: float
    support: int (number of WIN observations matching)
    near_miss_support: int (number of LOSE observations with progress)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .schemas import AgentState, SceneGraph

# =============================================================================
# Goal predicate vocabulary
# =============================================================================

class GoalPredicateType(StrEnum):
    """Types of terminal conditions the agent might be trying to achieve."""
    # Reach-type
    AGENT_AT = "agent_at"                       # agent reaches a specific position
    AGENT_REACHES_ZONE = "agent_reaches_zone"   # agent reaches a colored zone
    AGENT_TOUCHES = "agent_touches"             # agent touches a specific color

    # Collection-type
    COLLECT_ALL = "collect_all"                 # all objects of a color collected
    COUNT_COLLECTED = "count_collected"         # N objects collected

    # Elimination-type
    ELIMINATE_ALL = "eliminate_all"             # all objects of a color eliminated
    COUNT_ELIMINATED = "count_eliminated"

    # State-matching type (LS20-style)
    AGENT_STATE_MATCHES = "agent_state_matches" # agent shape/color/orientation matches target
    AGENT_ORIENTATION_MATCHES = "agent_orientation_matches"
    AGENT_COLOR_MATCHES = "agent_color_matches"
    AGENT_SHAPE_MATCHES = "agent_shape_matches"

    # Relational type (LS20 white cross + portal)
    OBJECT_ORIENTATION_MATCHES_OTHER = "object_orientation_matches_other"
    OBJECT_STATE_MATCHES_OTHER = "object_state_matches_other"

    # Configuration type
    ALL_OBJECTS_IN_CONFIGURATION = "all_objects_in_configuration"
    PATTERN_COMPLETE = "pattern_complete"

    # Survival type
    SURVIVE_N_TURNS = "survive_n_turns"
    ENERGY_ABOVE_THRESHOLD_AT_TURN_N = "energy_above_threshold_at_turn_n"


# =============================================================================
# Goal hypothesis
# =============================================================================

@dataclass
class GoalHypothesis:
    """A hypothesis about the win condition."""
    terminal_condition: str  # GoalPredicateType value
    args: dict               # predicate-specific args
    outcome: str = "WIN"     # "WIN" or "LOSE" (we mostly care about WIN)
    confidence: float = 0.0
    support: int = 0         # number of WIN observations matching
    near_miss_support: int = 0  # number of LOSE observations with progress
    mdl_cost: int = 10       # simplicity cost

    @property
    def score(self) -> float:
        """Score = confidence × (support + 0.3 × near_miss_support) / mdl_cost."""
        return self.confidence * (self.support + 0.3 * self.near_miss_support) / max(1, self.mdl_cost)


# =============================================================================
# Default goal priors — what we expect to find
# =============================================================================

# These priors reflect the distribution of goals in the 25 public demo games.
# Updated based on play sessions and the LS20 reverse-engineered spec.
DEFAULT_GOAL_PRIORS: list[tuple[str, dict, float]] = [
    # State-matching goals are most common in observed games (LS20)
    (GoalPredicateType.AGENT_STATE_MATCHES.value, {}, 0.20),
    (GoalPredicateType.AGENT_COLOR_MATCHES.value, {}, 0.15),
    (GoalPredicateType.AGENT_ORIENTATION_MATCHES.value, {}, 0.15),
    (GoalPredicateType.AGENT_SHAPE_MATCHES.value, {}, 0.10),
    # Reach goals
    (GoalPredicateType.AGENT_AT.value, {}, 0.10),
    (GoalPredicateType.AGENT_REACHES_ZONE.value, {}, 0.08),
    (GoalPredicateType.AGENT_TOUCHES.value, {}, 0.05),
    # Collection / elimination
    (GoalPredicateType.COLLECT_ALL.value, {}, 0.05),
    (GoalPredicateType.ELIMINATE_ALL.value, {}, 0.04),
    # Configuration
    (GoalPredicateType.PATTERN_COMPLETE.value, {}, 0.04),
    (GoalPredicateType.ALL_OBJECTS_IN_CONFIGURATION.value, {}, 0.02),
    # Relational
    (GoalPredicateType.OBJECT_ORIENTATION_MATCHES_OTHER.value, {}, 0.02),
    # Survival
    (GoalPredicateType.SURVIVE_N_TURNS.value, {}, 0.01),
]


# =============================================================================
# Goal predicate evaluation — does the goal condition hold right now?
# =============================================================================

def evaluate_goal_predicate(
    predicate: str,
    args: dict,
    scene: SceneGraph,
    agent_state: AgentState | None,
    target_state: dict | None = None,
) -> bool:
    """
    Evaluate whether a goal predicate is currently satisfied.

    Args:
        predicate: GoalPredicateType value
        args: predicate-specific args
        scene: current SceneGraph
        agent_state: current AgentState
        target_state: optional target spec (for state-matching goals)
            e.g. {"color": 5, "orientation": 1, "shape": "abc"}

    Returns:
        True if the goal condition holds.
    """
    if agent_state is None:
        return False

    if predicate == GoalPredicateType.AGENT_AT.value:
        target_pos = args.get("position")
        if target_pos is None:
            return False
        return tuple(agent_state.position) == tuple(target_pos)

    if predicate == GoalPredicateType.AGENT_REACHES_ZONE.value:
        zone_color = args.get("color")
        ax, ay = agent_state.position
        for obj in scene.objects.values():
            if obj.color != zone_color:
                continue
            if (ax, ay) in obj.cells:
                return True
        return False

    if predicate == GoalPredicateType.AGENT_TOUCHES.value:
        target_color = args.get("color")
        ax, ay = agent_state.position
        for obj in scene.objects.values():
            if target_color is not None and obj.color != target_color:
                continue
            for ox, oy in obj.cells:
                if abs(ox - ax) + abs(oy - ay) <= 1:
                    return True
        return False

    if predicate == GoalPredicateType.AGENT_COLOR_MATCHES.value:
        target_color = args.get("color") or (target_state or {}).get("color")
        if target_color is None:
            return False
        return target_color in agent_state.color

    if predicate == GoalPredicateType.AGENT_ORIENTATION_MATCHES.value:
        target_orient = args.get("orientation") or (target_state or {}).get("orientation")
        if target_orient is None:
            return False
        return agent_state.orientation == target_orient

    if predicate == GoalPredicateType.AGENT_SHAPE_MATCHES.value:
        target_shape = args.get("shape") or (target_state or {}).get("shape")
        if target_shape is None:
            return False
        return agent_state.shape == target_shape

    if predicate == GoalPredicateType.AGENT_STATE_MATCHES.value:
        # All specified attributes must match
        if not target_state:
            return False
        if "color" in target_state and target_state["color"] not in agent_state.color:
            return False
        if "orientation" in target_state and agent_state.orientation != target_state["orientation"]:
            return False
        return not ("shape" in target_state and agent_state.shape != target_state["shape"])

    if predicate == GoalPredicateType.COLLECT_ALL.value:
        # All objects of target color gone?
        target_color = args.get("color")
        if target_color is None:
            return False
        return not any(o.color == target_color for o in scene.objects.values())

    if predicate == GoalPredicateType.ELIMINATE_ALL.value:
        target_color = args.get("color")
        if target_color is None:
            return False
        return not any(o.color == target_color for o in scene.objects.values())

    if predicate == GoalPredicateType.OBJECT_ORIENTATION_MATCHES_OTHER.value:
        # Two objects of given colors have matching "orientation" (simplified: same shape hash)
        color_a = args.get("color_a")
        color_b = args.get("color_b")
        objs_a = [o for o in scene.objects.values() if o.color == color_a]
        objs_b = [o for o in scene.objects.values() if o.color == color_b]
        if not objs_a or not objs_b:
            return False
        # Check if any pair has matching shape hash
        for a in objs_a:
            for b in objs_b:
                if a.shape_hash == b.shape_hash:
                    return True
        return False

    if predicate == GoalPredicateType.SURVIVE_N_TURNS.value:
        # Tracked externally by the Near-Miss Tracker (turn count)
        return False  # handled by planner

    return False


# =============================================================================
# Goal hypothesis generation — what could the goal be, given a WIN observation?
# =============================================================================

def generate_goal_hypotheses_from_win(
    scene_before_win: SceneGraph,
    agent_state_before_win: AgentState | None,
    win_levels_delta: int = 1,
) -> list[GoalHypothesis]:
    """
    When a WIN is observed, generate goal hypotheses about what was true
    right before the win.

    Args:
        scene_before_win: SceneGraph from the step before WIN
        agent_state_before_win: AgentState from the step before WIN
        win_levels_delta: how many levels advanced (usually 1)

    Returns:
        List of candidate GoalHypothesis objects.
    """
    hypotheses: list[GoalHypothesis] = []

    if agent_state_before_win is None:
        return hypotheses

    # H1: Agent at specific position (reach goal)
    hypotheses.append(GoalHypothesis(
        terminal_condition=GoalPredicateType.AGENT_AT.value,
        args={"position": agent_state_before_win.position},
        confidence=0.3,
        support=1,
        mdl_cost=12,
    ))

    # H2: Agent reached a zone (any color present at agent's position)
    ax, ay = agent_state_before_win.position
    for obj in scene_before_win.objects.values():
        if (ax, ay) in obj.cells:
            hypotheses.append(GoalHypothesis(
                terminal_condition=GoalPredicateType.AGENT_REACHES_ZONE.value,
                args={"color": obj.color},
                confidence=0.3,
                support=1,
                mdl_cost=14,
            ))

    # H3: Agent touches specific colors
    touched_colors = set()
    for obj in scene_before_win.objects.values():
        for ox, oy in obj.cells:
            if abs(ox - ax) + abs(oy - ay) <= 1:
                touched_colors.add(obj.color)
    for color in touched_colors:
        hypotheses.append(GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_TOUCHES.value,
            args={"color": color},
            confidence=0.25,
            support=1,
            mdl_cost=14,
        ))

    # H4: Agent state matches (color, orientation, shape)
    if agent_state_before_win.color:
        for color in agent_state_before_win.color:
            hypotheses.append(GoalHypothesis(
                terminal_condition=GoalPredicateType.AGENT_COLOR_MATCHES.value,
                args={"color": color},
                confidence=0.3,
                support=1,
                mdl_cost=14,
            ))
    if agent_state_before_win.shape:
        hypotheses.append(GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_SHAPE_MATCHES.value,
            args={"shape": agent_state_before_win.shape},
            confidence=0.3,
            support=1,
            mdl_cost=14,
        ))
    hypotheses.append(GoalHypothesis(
        terminal_condition=GoalPredicateType.AGENT_ORIENTATION_MATCHES.value,
        args={"orientation": agent_state_before_win.orientation},
        confidence=0.25,
        support=1,
        mdl_cost=14,
    ))

    # H5: All objects of a specific color are gone (collect/eliminate)
    # Look at colors that are absent from the scene but were likely present before
    # (we can't know "before" without history; skip for now — Near-Miss Tracker helps)

    # H6: Two objects of different colors have matching shape (relational goal)
    colors_present = {o.color for o in scene_before_win.objects.values()}
    colors_list = sorted(colors_present)
    for i, c_a in enumerate(colors_list):
        for c_b in colors_list[i+1:]:
            objs_a = [o for o in scene_before_win.objects.values() if o.color == c_a]
            objs_b = [o for o in scene_before_win.objects.values() if o.color == c_b]
            for a in objs_a:
                for b in objs_b:
                    if a.shape_hash == b.shape_hash and a.shape_hash != "empty":
                        hypotheses.append(GoalHypothesis(
                            terminal_condition=GoalPredicateType.OBJECT_ORIENTATION_MATCHES_OTHER.value,
                            args={"color_a": c_a, "color_b": c_b},
                            confidence=0.4,  # higher because shape match is rare signal
                            support=1,
                            mdl_cost=18,
                        ))

    return hypotheses


# =============================================================================
# Goal Inference Module
# =============================================================================

class GoalInferenceModule:
    """
    Maintains a probability distribution over goal hypotheses.

    Usage:
        goal_module = GoalInferenceModule()
        # On each step:
        goal_module.observe_step(scene, agent_state, action_id)
        # On WIN:
        goal_module.observe_win(scene_before_win, agent_state_before_win)
        # On LOSE (via Near-Miss Tracker):
        goal_module.observe_lose(scene_before_lose, agent_state_before_lose, progress_predicates)
        # Get top goal:
        top_goal = goal_module.get_top_goal()
        if top_goal and top_goal.confidence >= 0.6:
            # Activate planner
            ...
    """

    def __init__(
        self,
        priors: list[tuple[str, dict, float]] | None = None,
        top_k: int = 15,
        confidence_floor: float = 0.02,
        planning_threshold: float = 0.6,
    ):
        self.top_k = top_k
        self.confidence_floor = confidence_floor
        self.planning_threshold = planning_threshold
        self.hypotheses: list[GoalHypothesis] = []
        self._win_count: int = 0
        self._lose_count: int = 0
        self._step_count: int = 0
        self._init_priors(priors or DEFAULT_GOAL_PRIORS)

    def _init_priors(self, priors: list[tuple[str, dict, float]]) -> None:
        """Initialize hypotheses from priors."""
        for predicate, args, prior_conf in priors:
            self.hypotheses.append(GoalHypothesis(
                terminal_condition=predicate,
                args=args,
                confidence=prior_conf,
                support=0,
                near_miss_support=0,
                mdl_cost=12,
            ))

    def observe_step(
        self,
        scene: SceneGraph,
        agent_state: AgentState | None,
        action_id: int,
    ) -> None:
        """Track current state. Used to record the 'before win' state."""
        self._step_count += 1
        self._last_scene = scene
        self._last_agent_state = agent_state

    def observe_win(
        self,
        scene_before_win: SceneGraph | None = None,
        agent_state_before_win: AgentState | None = None,
    ) -> None:
        """
        Observe a WIN event. Generates and updates goal hypotheses.

        Args:
            scene_before_win: scene from the step before WIN.
                If None, uses the last scene from observe_step.
            agent_state_before_win: agent state from the step before WIN.
                If None, uses the last agent state.
        """
        self._win_count += 1
        if scene_before_win is None:
            scene_before_win = getattr(self, "_last_scene", None)
        if agent_state_before_win is None:
            agent_state_before_win = getattr(self, "_last_agent_state", None)
        if scene_before_win is None or agent_state_before_win is None:
            return

        # 1. Generate new hypotheses from this WIN
        new_hyps = generate_goal_hypotheses_from_win(
            scene_before_win, agent_state_before_win
        )

        # 2. Merge with existing: if similar hypothesis exists, update it; else add
        for new_hyp in new_hyps:
            existing = self._find_similar(new_hyp)
            if existing is not None:
                # Bayesian update
                existing.confidence = self._bayesian_update(
                    existing.confidence, likelihood=0.7, prior_weight=0.5
                )
                existing.support += 1
            else:
                self.hypotheses.append(new_hyp)

        # 3. Downgrade hypotheses whose conditions DON'T hold at the win scene
        # (If the goal was really X, X should be true at the moment of winning)
        for hyp in self.hypotheses:
            holds = evaluate_goal_predicate(
                hyp.terminal_condition, hyp.args, scene_before_win, agent_state_before_win
            )
            if not holds and hyp.support > 0:
                # Condition didn't hold at WIN — weak negative evidence
                hyp.confidence = self._bayesian_update(
                    hyp.confidence, likelihood=0.3, prior_weight=0.7
                )

        # 4. Prune
        self._prune()

    def observe_lose(
        self,
        scene_before_lose: SceneGraph | None = None,
        agent_state_before_lose: AgentState | None = None,
        progress_predicates: list[str] | None = None,
    ) -> None:
        """
        Observe a LOSE event. Uses progress predicates (from Near-Miss Tracker)
        to provide weak evidence for goal hypotheses.

        Args:
            scene_before_lose: scene from the step before LOSE
            agent_state_before_lose: agent state from the step before LOSE
            progress_predicates: list of predicate names that showed progress
                before death (e.g. ["distance_to_exit", "color_match_count"])
        """
        self._lose_count += 1
        if not progress_predicates:
            return
        if scene_before_lose is None:
            scene_before_lose = getattr(self, "_last_scene", None)
        if agent_state_before_lose is None:
            agent_state_before_lose = getattr(self, "_last_agent_state", None)
        if scene_before_lose is None or agent_state_before_lose is None:
            return

        # Map progress predicates to goal types
        progress_to_goal_map = {
            "distance_to_exit": [GoalPredicateType.AGENT_AT.value, GoalPredicateType.AGENT_REACHES_ZONE.value],
            "color_match_count": [GoalPredicateType.AGENT_COLOR_MATCHES.value, GoalPredicateType.AGENT_STATE_MATCHES.value],
            "shape_match_count": [GoalPredicateType.AGENT_SHAPE_MATCHES.value],
            "orientation_match_count": [GoalPredicateType.AGENT_ORIENTATION_MATCHES.value],
            "enemies_remaining": [GoalPredicateType.ELIMINATE_ALL.value],
            "gold_collected_count": [GoalPredicateType.COLLECT_ALL.value],
        }

        relevant_goal_types: set[str] = set()
        for prog_pred in progress_predicates:
            relevant_goal_types.update(progress_to_goal_map.get(prog_pred, []))

        # Weak positive evidence for matching goal types
        for hyp in self.hypotheses:
            if hyp.terminal_condition in relevant_goal_types:
                hyp.near_miss_support += 1
                hyp.confidence = self._bayesian_update(
                    hyp.confidence, likelihood=0.6, prior_weight=0.7
                )

    def _find_similar(self, hypothesis: GoalHypothesis) -> GoalHypothesis | None:
        """Find an existing hypothesis with same predicate and equivalent args."""
        for existing in self.hypotheses:
            if existing.terminal_condition != hypothesis.terminal_condition:
                continue
            if self._args_equivalent(existing.args, hypothesis.args, existing.terminal_condition):
                return existing
        return None

    def _args_equivalent(self, a: dict, b: dict, predicate: str) -> bool:
        """
        Check if two args dicts are equivalent.

        For agent_at, position matters — each position is a distinct hypothesis.
        For other predicates, args must match exactly.
        """
        # For position-based predicates, position is part of the identity
        if predicate == GoalPredicateType.AGENT_AT.value:
            return a.get("position") == b.get("position")
        # For color-based predicates, color is part of the identity
        if predicate in (
            GoalPredicateType.AGENT_COLOR_MATCHES.value,
            GoalPredicateType.AGENT_REACHES_ZONE.value,
            GoalPredicateType.AGENT_TOUCHES.value,
            GoalPredicateType.COLLECT_ALL.value,
            GoalPredicateType.ELIMINATE_ALL.value,
        ):
            return a.get("color") == b.get("color")
        # For shape-based predicates, shape is part of the identity
        if predicate == GoalPredicateType.AGENT_SHAPE_MATCHES.value:
            return a.get("shape") == b.get("shape")
        # For orientation-based predicates, orientation is part of the identity
        if predicate == GoalPredicateType.AGENT_ORIENTATION_MATCHES.value:
            return a.get("orientation") == b.get("orientation")
        # For relational predicates, both colors matter
        if predicate == GoalPredicateType.OBJECT_ORIENTATION_MATCHES_OTHER.value:
            return (
                a.get("color_a") == b.get("color_a")
                and a.get("color_b") == b.get("color_b")
            )
        # Default: exact match
        return a == b

    def _bayesian_update(self, prior: float, likelihood: float,
                          prior_weight: float = 1.0) -> float:
        """Weighted Bayesian update, clamped to [0.001, 0.999]."""
        numerator = likelihood * prior
        denominator = numerator + (1 - likelihood) * (1 - prior) * prior_weight
        if denominator == 0:
            return prior
        posterior = numerator / denominator
        return max(0.001, min(0.999, posterior))

    def _prune(self) -> None:
        """Prune low-confidence hypotheses, keep top-K."""
        # Remove below floor
        self.hypotheses = [h for h in self.hypotheses if h.confidence >= self.confidence_floor]
        if len(self.hypotheses) <= self.top_k:
            return
        # Sort by score descending
        self.hypotheses.sort(key=lambda h: -h.score)
        self.hypotheses = self.hypotheses[:self.top_k]

    def get_top_goal(self) -> GoalHypothesis | None:
        """Return the highest-confidence goal hypothesis, or None."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.score)

    def get_top_goals(self, k: int = 5) -> list[GoalHypothesis]:
        """Return the top-k goal hypotheses by score."""
        return sorted(self.hypotheses, key=lambda h: -h.score)[:k]

    def is_ready_to_plan(self) -> bool:
        """True if the top goal exceeds the planning threshold."""
        top = self.get_top_goal()
        return top is not None and top.confidence >= self.planning_threshold

    def get_stats(self) -> dict:
        return {
            "step_count": self._step_count,
            "win_count": self._win_count,
            "lose_count": self._lose_count,
            "hypothesis_count": len(self.hypotheses),
            "top_goal": (
                f"{self.get_top_goal().terminal_condition}({self.get_top_goal().args})"
                if self.get_top_goal() else "None"
            ),
            "top_confidence": self.get_top_goal().confidence if self.get_top_goal() else 0.0,
            "ready_to_plan": self.is_ready_to_plan(),
            "planning_threshold": self.planning_threshold,
        }


# =============================================================================
# Convenience: run on a replay
# =============================================================================

def run_on_replay(replay, max_steps: int | None = None) -> tuple[GoalInferenceModule, dict]:
    """
    Run the GoalInferenceModule over a replay. Detects WIN events and
    generates goal hypotheses from each.
    """
    import numpy as np

    from .agent_tracker import AgentStateTracker
    from .perception import detect_background_color, perceive
    from .schemas import GameState

    goal_module = GoalInferenceModule()
    tracker = AgentStateTracker()
    n = len(replay.records) if max_steps is None else min(max_steps, len(replay.records))

    for i in range(n):
        record = replay.records[i]
        grid = np.array(record.final_grid, dtype=np.int8)
        bg = detect_background_color(grid)
        scene = perceive(grid.tolist(), background_color=bg)
        agent_state, _ = tracker.update(
            grid,
            record.action_input.id,
            background_color=bg,
            available_actions=record.available_actions,
            action_input=record.action_input,
        )

        # Check for level transition (proxy for WIN within replay)
        prev_record = replay.records[i - 1] if i > 0 else None
        if prev_record and record.levels_completed > prev_record.levels_completed or record.state == GameState.WIN:
            goal_module.observe_win(scene, agent_state)
        elif record.state == GameState.GAME_OVER:
            goal_module.observe_lose(scene, agent_state)
        else:
            goal_module.observe_step(scene, agent_state, record.action_input.id)

    return goal_module, goal_module.get_stats()
