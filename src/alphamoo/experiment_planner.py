"""
AlphaMoo v4.1 — Experiment Planner (Module 8, Curiosity Engine).

Chooses the next action to maximize information gain about the active
hypothesis set.

Three-tier IG computation (per v4.1 spec):
  Tier 1: Symbolic IG — analytical, no LLM call. ~80% of decisions.
  Tier 2: Surrogate IG — fast MLP (placeholder for now).
  Tier 3: LLM IG — DROPPED per Phase 0 (0.5B confirmed, no spare tokens).

Selection rule:
  1. Compute IG for each candidate action (Tier 1, fall back to Tier 2)
  2. Filter via Affordance (skip actions the affordance module says are useless)
  3. If max IG > IG_FLOOR: pick argmax IG
  4. If max IG ≤ IG_FLOOR: ε-greedy escape (random untried action)
  5. Penalize revisits by 0.5×

Experiment types, in priority order:
  1. Contrastive — vary one variable, hold others fixed
  2. Boundary probes — test edge cases of active hypotheses
  3. Novel state discovery — reach states not in episodic memory
  4. Intentional termination — deliberately trigger suspected terminal state
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .hypothesis_generator import (
    evaluate_trigger,
)
from .schemas import (
    GRID_SIZE,
    ActionId,
    AgentState,
    GoalHypothesis,
    Hypothesis,
    SceneGraph,
)

# =============================================================================
# Constants
# =============================================================================

# IG_FLOOR: below this, switch to ε-greedy escape
DEFAULT_IG_FLOOR = 0.05

# Revisit penalty
REVISIT_PENALTY = 0.5

# Default ε for exploration
DEFAULT_EPSILON = 0.1


# =============================================================================
# Action representation
# =============================================================================

@dataclass
class CandidateAction:
    """A candidate action the planner is considering."""
    action_id: int
    click_coords: tuple[int, int] | None = None  # for ACTION6
    ig_score: float = 0.0  # information gain score
    ig_tier: int = 0       # 1=symbolic, 2=surrogate, 3=LLM
    experiment_type: str = ""  # "contrastive" | "boundary" | "novel" | "termination"
    notes: str = ""


# =============================================================================
# Symbolic IG computation (Tier 1)
# =============================================================================

def compute_symbolic_ig(
    action: CandidateAction,
    scene: SceneGraph,
    agent_state: AgentState | None,
    hypotheses: list[Hypothesis],
    visited_state_hashes: set[str],
) -> float | None:
    """
    Compute information gain analytically, no LLM call.

    Strategy:
      1. For each hypothesis whose trigger currently holds, IG = confidence
         (because taking an action will likely produce an event that
         confirms or denies the hypothesis)
      2. For each hypothesis whose trigger would newly hold if this action
         moves the agent into position, IG = confidence × novelty
      3. Penalize if the resulting state has been visited before

    Returns:
        IG score, or None if symbolic IG can't compute (caller falls back
        to Tier 2).
    """
    if not hypotheses:
        return 0.0

    ig_total = 0.0
    action_id = action.action_id

    # For each hypothesis, check if its trigger currently holds or would hold
    for hyp in hypotheses:
        # Does the trigger currently hold (before action)?
        trigger_holds_now = evaluate_trigger(hyp.trigger, scene, agent_state, action_id)

        if trigger_holds_now:
            # Action may produce an event that confirms/denies this hypothesis
            # IG = confidence × log2(1 + support) — higher confidence + support = more IG
            ig_total += hyp.confidence * math.log2(1 + hyp.support + 1)

        else:
            # Could this action cause the trigger to hold?
            # Simplified: movement actions might bring agent into position
            if action_id in (ActionId.UP, ActionId.DOWN, ActionId.LEFT, ActionId.RIGHT) and (
                agent_state and agent_state.position
            ):
                # Check if moving would satisfy an agent_touches/agent_at condition
                new_pos = _simulate_movement(agent_state.position, action_id)
                hypothetical_agent = AgentState(
                    object_ids=agent_state.object_ids,
                    position=new_pos,
                    orientation=agent_state.orientation,
                    shape=agent_state.shape,
                    color=agent_state.color,
                    energy=agent_state.energy,
                    inventory=agent_state.inventory,
                )
                trigger_holds_after = evaluate_trigger(
                    hyp.trigger, scene, hypothetical_agent, action_id
                )
                if trigger_holds_after:
                    # Action would newly satisfy the trigger — high IG
                    ig_total += hyp.confidence * 0.5  # discounted (uncertain)

    # Novel state bonus: if the resulting state hasn't been visited, +IG
    state_hash = _compute_state_hash(scene, agent_state, action_id)
    if state_hash not in visited_state_hashes:
        ig_total += 0.1  # small novelty bonus
    else:
        ig_total *= REVISIT_PENALTY  # penalize revisits

    return ig_total


def _simulate_movement(position: tuple[int, int], action_id: int) -> tuple[int, int]:
    """Simulate agent movement under a directional action."""
    x, y = position
    if action_id == ActionId.UP:
        return (x, max(0, y - 1))
    if action_id == ActionId.DOWN:
        return (x, min(GRID_SIZE - 1, y + 1))
    if action_id == ActionId.LEFT:
        return (max(0, x - 1), y)
    if action_id == ActionId.RIGHT:
        return (min(GRID_SIZE - 1, x + 1), y)
    return position


def _compute_state_hash(
    scene: SceneGraph,
    agent_state: AgentState | None,
    action_id: int,
) -> str:
    """Compute a hash of (state, action) for revisit detection."""
    import hashlib
    state_str = f"{scene.hash}|{agent_state.position if agent_state else 'None'}|{action_id}"
    return hashlib.md5(state_str.encode()).hexdigest()[:12]


# =============================================================================
# Surrogate IG (Tier 2) — placeholder
# =============================================================================

def compute_surrogate_ig(
    action: CandidateAction,
    scene: SceneGraph,
    agent_state: AgentState | None,
    hypotheses: list[Hypothesis],
) -> float:
    """
    Surrogate IG via a tiny MLP. Placeholder: returns a heuristic
    based on action type and hypothesis count.
    """
    # Heuristic: prefer actions that haven't been tried recently
    base_score = 0.05
    if action.action_id == ActionId.INTERACT:
        base_score = 0.08  # interact often reveals mechanics
    elif action.action_id == ActionId.CLICK:
        base_score = 0.06
    elif action.action_id in (ActionId.UP, ActionId.DOWN, ActionId.LEFT, ActionId.RIGHT):
        base_score = 0.04
    return base_score * (1 + 0.1 * len(hypotheses))


# =============================================================================
# Action candidate generation
# =============================================================================

def generate_candidate_actions(
    available_actions: list[int],
    scene: SceneGraph,
    agent_state: AgentState | None,
    max_click_targets: int = 5,
) -> list[CandidateAction]:
    """
    Generate candidate actions to consider.

    For movement games: one candidate per available movement action.
    For click games: candidates for click on each "interesting" object.
    """
    candidates: list[CandidateAction] = []

    for action_id in available_actions:
        if action_id == ActionId.CLICK:
            # Generate click candidates for each interesting object
            click_targets = _select_click_targets(scene, agent_state, max_click_targets)
            for target in click_targets:
                candidates.append(CandidateAction(
                    action_id=action_id,
                    click_coords=target,
                ))
            if not click_targets:
                # No targets — click on agent position or grid center
                target = agent_state.position if agent_state else (32, 32)
                candidates.append(CandidateAction(
                    action_id=action_id,
                    click_coords=target,
                ))
        elif action_id == ActionId.RESET:
            # RESET is rarely a useful exploration action; include but mark low priority
            candidates.append(CandidateAction(
                action_id=action_id,
                experiment_type="termination",
                notes="RESET (intentional restart)",
            ))
        elif action_id == ActionId.UNDO:
            candidates.append(CandidateAction(
                action_id=action_id,
                notes="UNDO",
            ))
        else:
            candidates.append(CandidateAction(action_id=action_id))

    return candidates


def _select_click_targets(
    scene: SceneGraph,
    agent_state: AgentState | None,
    max_targets: int,
) -> list[tuple[int, int]]:
    """Select interesting click targets from the scene."""
    if not scene.objects:
        return []
    # Sort by size (smaller objects are often more interactive)
    sorted_objs = sorted(scene.objects.values(), key=lambda o: len(o.cells))
    targets: list[tuple[int, int]] = []
    for obj in sorted_objs[:max_targets]:
        # Click on the centroid of the object
        xs = [x for x, _ in obj.cells]
        ys = [y for _, y in obj.cells]
        cx = sum(xs) // len(xs)
        cy = sum(ys) // len(ys)
        targets.append((cx, cy))
    return targets


# =============================================================================
# Experiment Planner
# =============================================================================

class ExperimentPlanner:
    """
    The Curiosity Engine. Chooses the next action to maximize information
    gain about the active hypothesis set.

    Usage:
        planner = ExperimentPlanner()
        # On each step:
        action = planner.select_action(
            available_actions=record.available_actions,
            scene=scene,
            agent_state=agent_state,
            hypotheses=gen.get_top_hypotheses(k=10),
            goal_hypotheses=goal_module.get_top_goals(k=5),
        )
    """

    def __init__(
        self,
        ig_floor: float = DEFAULT_IG_FLOOR,
        epsilon: float = DEFAULT_EPSILON,
        rng_seed: int | None = None,
    ):
        self.ig_floor = ig_floor
        self.epsilon = epsilon
        self.rng = random.Random(rng_seed)
        self.visited_state_hashes: set[str] = set()
        self._selection_count: int = 0
        self._escape_count: int = 0
        self._tier1_count: int = 0
        self._tier2_count: int = 0
        self.recent_actions: list[int] = []  # for ε-greedy escape

    def select_action(
        self,
        available_actions: list[int],
        scene: SceneGraph,
        agent_state: AgentState | None,
        hypotheses: list[Hypothesis],
        goal_hypotheses: list[GoalHypothesis] | None = None,
        goal_module: object | None = None,
    ) -> CandidateAction:
        """
        Select the next action via information gain maximization.

        Args:
            available_actions: list of valid action IDs
            scene: current SceneGraph
            agent_state: current AgentState
            hypotheses: top-K mechanics hypotheses from Module 5
            goal_hypotheses: top-K goal hypotheses from Module 6
            goal_module: the GoalInferenceModule (for intentional termination
                when goal confidence is high)

        Returns:
            CandidateAction with ig_score and experiment_type set.
        """
        self._selection_count += 1

        # Generate candidates
        candidates = generate_candidate_actions(
            available_actions, scene, agent_state
        )

        if not candidates:
            # Fallback: pick any available action
            return CandidateAction(
                action_id=available_actions[0] if available_actions else 0,
                notes="fallback (no candidates generated)",
            )

        # Compute IG for each candidate
        for cand in candidates:
            # Tier 1: symbolic IG
            ig = compute_symbolic_ig(
                cand, scene, agent_state, hypotheses, self.visited_state_hashes
            )
            if ig is not None:
                cand.ig_score = ig
                cand.ig_tier = 1
                self._tier1_count += 1
            else:
                # Tier 2: surrogate IG
                cand.ig_score = compute_surrogate_ig(
                    cand, scene, agent_state, hypotheses
                )
                cand.ig_tier = 2
                self._tier2_count += 1

            # Classify experiment type
            cand.experiment_type = self._classify_experiment(
                cand, scene, agent_state, hypotheses
            )

        # Sort by IG descending
        candidates.sort(key=lambda c: -c.ig_score)

        # ε-greedy escape: if max IG is below floor, take random untried action
        if candidates[0].ig_score < self.ig_floor:
            self._escape_count += 1
            untried = [
                c for c in candidates
                if c.action_id not in self.recent_actions[-5:]
            ]
            if untried and self.rng.random() < self.epsilon:
                chosen = self.rng.choice(untried)
                chosen.notes = "ε-greedy escape"
                self._record_action(chosen)
                return chosen

        # Take the highest-IG action
        chosen = candidates[0]
        self._record_action(chosen)
        return chosen

    def _classify_experiment(
        self,
        action: CandidateAction,
        scene: SceneGraph,
        agent_state: AgentState | None,
        hypotheses: list[Hypothesis],
    ) -> str:
        """Classify what type of experiment this action represents."""
        # If action is RESET, it's intentional termination
        if action.action_id == ActionId.RESET:
            return "termination"

        # If action would test a hypothesis whose trigger doesn't currently hold
        for hyp in hypotheses:
            trigger_holds = evaluate_trigger(hyp.trigger, scene, agent_state, action.action_id)
            if not trigger_holds and action.action_id in (
                ActionId.UP, ActionId.DOWN, ActionId.LEFT, ActionId.RIGHT
            ):
                # Movement to test if trigger would newly hold
                return "boundary"

        # If state hasn't been visited, it's novel discovery
        state_hash = _compute_state_hash(scene, agent_state, action.action_id)
        if state_hash not in self.visited_state_hashes:
            return "novel"

        # Default: contrastive (we're varying something)
        return "contrastive"

    def _record_action(self, action: CandidateAction) -> None:
        """Record the chosen action for revisit detection."""
        self.recent_actions.append(action.action_id)
        if len(self.recent_actions) > 20:
            self.recent_actions = self.recent_actions[-20:]

    def record_visited_state(self, scene: SceneGraph, agent_state: AgentState | None,
                               action_id: int) -> None:
        """Record a visited state for revisit detection."""
        state_hash = _compute_state_hash(scene, agent_state, action_id)
        self.visited_state_hashes.add(state_hash)

    def get_stats(self) -> dict:
        return {
            "selection_count": self._selection_count,
            "escape_count": self._escape_count,
            "escape_rate": self._escape_count / max(1, self._selection_count),
            "tier1_count": self._tier1_count,
            "tier2_count": self._tier2_count,
            "visited_states": len(self.visited_state_hashes),
            "ig_floor": self.ig_floor,
            "epsilon": self.epsilon,
        }
