"""
AlphaMoo v4.1 — Hypothesis Generator (Module 5).

Maintains a probability distribution over causal mechanics hypotheses.

A hypothesis is: IF trigger THEN effect [confidence, support]

The generator:
  1. Observes events from the Cascade Interpreter
  2. Generates candidate hypotheses from those events
  3. Updates confidence via Bayesian update on each new observation
  4. Prunes to top-K=30 with diversity filter

Predicate vocabulary (from v4.1 spec):
  TRIGGER predicates:
    agent_touches(obj_type), agent_has(obj_type), agent_adjacent(obj_type),
    agent_enters(zone), agent_at(position), obj_count(obj_type) == N,
    agent_state_is(attr, value), object_orientation_is(obj_id, value),
    energy_below(threshold), visible(obj_id), not_visible(obj_id)

  EFFECT primitives:
    state_changes(obj_type, attr, new_val), obj_appears(obj_type, position),
    obj_disappears(obj_id), score_increments(delta),
    episode_ends(outcome), agent_teleports(position),
    agent_state_changes(attr, new_value), agent_displace(dx, dy),
    obj_appears_at(obj_type, position)

Hypothesis form supports:
  - Conjunctive triggers: IF A AND B THEN effect
  - Negated conditions: IF NOT A THEN effect
  - Temporal specs: IF A for 3 turns THEN effect
  - Quantified: IF all(gold_collected) THEN exit_opens
"""
from __future__ import annotations

from enum import StrEnum

from .schemas import AgentState, CascadeEvent, Condition, Hypothesis, SceneGraph, Trigger

# =============================================================================
# Predicate vocabulary
# =============================================================================

class PredicateType(StrEnum):
    """Trigger predicate types."""
    AGENT_TOUCHES = "agent_touches"
    AGENT_HAS = "agent_has"
    AGENT_ADJACENT = "agent_adjacent"
    AGENT_ENTERS = "agent_enters"
    AGENT_AT = "agent_at"
    OBJ_COUNT = "obj_count"
    AGENT_STATE_IS = "agent_state_is"
    OBJECT_ORIENTATION_IS = "object_orientation_is"
    ENERGY_BELOW = "energy_below"
    VISIBLE = "visible"
    NOT_VISIBLE = "not_visible"


class EffectType(StrEnum):
    """Effect types."""
    STATE_CHANGES = "state_changes"
    OBJ_APPEARS = "obj_appears"
    OBJ_DISAPPEARS = "obj_disappears"
    SCORE_INCREMENTS = "score_increments"
    EPISODE_ENDS = "episode_ends"
    AGENT_TELEPORTS = "agent_teleports"
    AGENT_STATE_CHANGES = "agent_state_changes"
    AGENT_DISPLACE = "agent_displace"
    OBJ_APPEARS_AT = "obj_appears_at"


# =============================================================================
# Hypothesis evaluation — does a trigger match the current state?
# =============================================================================

def evaluate_condition(condition: Condition, scene: SceneGraph,
                       agent_state: AgentState | None,
                       action_id: int) -> bool:
    """
    Evaluate whether a condition is currently true.

    Args:
        condition: the Condition to evaluate
        scene: current SceneGraph
        agent_state: current AgentState (or None)
        action_id: the action being taken (1-7)

    Returns:
        True if the condition holds, False otherwise.
    """
    result = _evaluate_condition_inner(condition, scene, agent_state, action_id)
    return (not result) if condition.negated else result


def _evaluate_condition_inner(condition: Condition, scene: SceneGraph,
                              agent_state: AgentState | None,
                              action_id: int) -> bool:
    """Inner evaluation (before negation)."""
    pred = condition.predicate
    args = condition.args

    if pred == PredicateType.AGENT_TOUCHES.value:
        # Is agent adjacent to an object of the given type/color?
        if agent_state is None or not agent_state.position:
            return False
        target_color = args.get("color")
        ax, ay = agent_state.position
        for obj in scene.objects.values():
            if target_color is not None and obj.color != target_color:
                continue
            for ox, oy in obj.cells:
                if abs(ox - ax) + abs(oy - ay) <= 1:
                    return True
        return False

    if pred == PredicateType.AGENT_ADJACENT.value:
        # Same as touches but strictly adjacent (not on same cell)
        if agent_state is None or not agent_state.position:
            return False
        target_color = args.get("color")
        ax, ay = agent_state.position
        for obj in scene.objects.values():
            if target_color is not None and obj.color != target_color:
                continue
            for ox, oy in obj.cells:
                if abs(ox - ax) + abs(oy - ay) == 1:
                    return True
        return False

    if pred == PredicateType.AGENT_AT.value:
        if agent_state is None:
            return False
        target_pos = args.get("position")
        if target_pos is None:
            return False
        return tuple(agent_state.position) == tuple(target_pos)

    if pred == PredicateType.AGENT_HAS.value:
        # Agent inventory check (simplified — we don't track inventory yet)
        return False

    if pred == PredicateType.AGENT_STATE_IS.value:
        if agent_state is None:
            return False
        attr = args.get("attr")
        value = args.get("value")
        if attr == "color":
            return value in agent_state.color
        if attr == "orientation":
            return agent_state.orientation == value
        if attr == "shape":
            return agent_state.shape == value
        return False

    if pred == PredicateType.OBJ_COUNT.value:
        target_color = args.get("color")
        expected = args.get("count", 0)
        actual = sum(1 for o in scene.objects.values() if o.color == target_color)
        return actual == expected

    if pred == PredicateType.ENERGY_BELOW.value:
        if agent_state is None or agent_state.energy is None:
            return False
        threshold = args.get("threshold", 0)
        return agent_state.energy < threshold

    if pred == PredicateType.VISIBLE.value:
        # Simplified: assume full visibility (partial observability is Module 15)
        return True

    if pred == PredicateType.NOT_VISIBLE.value:
        return False  # simplified

    return False


def evaluate_trigger(trigger: Trigger, scene: SceneGraph,
                     agent_state: AgentState | None,
                     action_id: int) -> bool:
    """
    Evaluate whether all conditions in a trigger hold (AND semantics).
    """
    return all(
        evaluate_condition(cond, scene, agent_state, action_id)
        for cond in trigger.conditions
    )


# =============================================================================
# Hypothesis matching — does an effect match an observed event?
# =============================================================================

def effect_matches_event(effect: dict, event: CascadeEvent) -> bool:
    """
    Check if an observed CascadeEvent matches a hypothesis's effect.

    Args:
        effect: the hypothesis effect dict {type: ..., args: ...}
        event: the observed CascadeEvent

    Returns:
        True if the event is an instance of the effect.
    """
    eff_type = effect.get("type")
    event_type = event.type

    if eff_type == EffectType.OBJ_APPEARS.value and event_type == "appearance":
        target_color = effect.get("args", {}).get("color")
        return target_color is None or event.target_color == target_color

    if eff_type == EffectType.OBJ_DISAPPEARS.value and event_type == "disappearance":
        target_color = effect.get("args", {}).get("color")
        return target_color is None or event.target_color == target_color

    if eff_type == EffectType.AGENT_STATE_CHANGES.value and event_type == "color_change":
        # Color change of agent = agent_state_changes(attr="color", ...)
        return True

    if eff_type == EffectType.EPISODE_ENDS.value and event_type == "level_transition":
        return True

    return bool(eff_type == EffectType.AGENT_DISPLACE.value and event_type == "move")


# =============================================================================
# Bayesian update
# =============================================================================

def bayesian_update(prior: float, likelihood: float, prior_weight: float = 1.0) -> float:
    """
    Bayesian update of a hypothesis's confidence.

    Args:
        prior: current confidence P(H)
        likelihood: P(obs | H) — how likely is this observation if H is true
        prior_weight: weight of the prior (0-1, higher = more conservative)

    Returns:
        Posterior confidence P(H | obs)
    """
    # Simplified Bayesian update with weight
    # P(H|obs) ∝ P(obs|H) * P(H)
    # We use a weighted update to avoid collapse to 0 or 1 too fast
    numerator = likelihood * prior
    denominator = numerator + (1 - likelihood) * (1 - prior) * prior_weight
    if denominator == 0:
        return prior
    posterior = numerator / denominator
    # Clamp to [0.001, 0.999] to avoid log(0) issues
    return max(0.001, min(0.999, posterior))


# =============================================================================
# Hypothesis generation from events
# =============================================================================

def generate_hypotheses_from_event(
    event: CascadeEvent,
    scene: SceneGraph,
    agent_state: AgentState | None,
    action_id: int,
) -> list[Hypothesis]:
    """
    Generate candidate hypotheses that could explain an observed event.

    For each event, we generate multiple hypotheses with different trigger
    conditions. The Bayesian update will then strengthen the ones that match
    future observations and weaken the ones that don't.

    Args:
        event: the observed CascadeEvent
        scene: current SceneGraph
        agent_state: current AgentState
        action_id: the action that triggered this event

    Returns:
        List of candidate Hypothesis objects (all with low initial confidence)
    """
    hypotheses: list[Hypothesis] = []

    # Build the effect dict from the event
    effect = _event_to_effect(event)
    if effect is None:
        return []

    # Generate trigger candidates based on the current state
    triggers = _generate_trigger_candidates(scene, agent_state, action_id, event)

    for trigger in triggers:
        hyp = Hypothesis(
            trigger=trigger,
            effect=effect,
            confidence=0.1,  # low initial prior
            support=1,       # one observation supports this
            mdl_cost=_compute_mdl_cost(trigger, effect),
        )
        hypotheses.append(hyp)

    return hypotheses


def _event_to_effect(event: CascadeEvent) -> dict | None:
    """Convert a CascadeEvent to an effect dict."""
    if event.type == "appearance":
        return {
            "type": EffectType.OBJ_APPEARS.value,
            "args": {"color": event.target_color},
        }
    if event.type == "disappearance":
        return {
            "type": EffectType.OBJ_DISAPPEARS.value,
            "args": {"color": event.target_color},
        }
    if event.type == "color_change":
        return {
            "type": EffectType.AGENT_STATE_CHANGES.value,
            "args": {
                "attr": "color",
                "old": event.before.get("old_color"),
                "new": event.after.get("new_color"),
            },
        }
    if event.type == "move":
        return {
            "type": EffectType.AGENT_DISPLACE.value,
            "args": event.after.get("displacement", (0, 0)),
        }
    if event.type == "level_transition":
        return {
            "type": EffectType.EPISODE_ENDS.value,
            "args": {"outcome": "WIN"},
        }
    return None


def _generate_trigger_candidates(
    scene: SceneGraph,
    agent_state: AgentState | None,
    action_id: int,
    event: CascadeEvent,
) -> list[Trigger]:
    """Generate multiple trigger candidates for an event."""
    triggers: list[Trigger] = []

    if agent_state is None or not agent_state.position:
        return triggers

    # Candidate 1: agent touches the event's target color
    if event.target_color is not None:
        triggers.append(Trigger(
            conditions=[Condition(
                predicate=PredicateType.AGENT_TOUCHES.value,
                args={"color": event.target_color},
            )],
        ))

    # Candidate 2: agent adjacent to the event's target color
    if event.target_color is not None:
        triggers.append(Trigger(
            conditions=[Condition(
                predicate=PredicateType.AGENT_ADJACENT.value,
                args={"color": event.target_color},
            )],
        ))

    # Candidate 3: agent at specific position
    triggers.append(Trigger(
        conditions=[Condition(
            predicate=PredicateType.AGENT_AT.value,
            args={"position": agent_state.position},
        )],
    ))

    # Candidate 4: agent state (color) triggers the event
    if agent_state.color:
        for color in agent_state.color:
            triggers.append(Trigger(
                conditions=[Condition(
                    predicate=PredicateType.AGENT_STATE_IS.value,
                    args={"attr": "color", "value": color},
                )],
            ))

    # Candidate 5: action-based trigger (the action itself caused the event)
    # This is a simplified "action_id == X" condition encoded as agent_at
    # In a full implementation we'd have an ACTION_EQUALS predicate

    return triggers


def _compute_mdl_cost(trigger: Trigger, effect: dict) -> int:
    """
    Compute the Minimum Description Length cost of a hypothesis.
    Shorter/simpler hypotheses have lower cost (preferred).
    """
    cost = 0
    # Each condition costs ~10 bits
    cost += len(trigger.conditions) * 10
    # Each condition argument costs ~2 bits
    for cond in trigger.conditions:
        cost += len(cond.args) * 2
    # Effect costs ~10 bits + args
    cost += 10 + len(effect.get("args", {})) * 2
    # Temporal spec costs extra
    if trigger.temporal:
        cost += 5
    return cost


# =============================================================================
# Hypothesis Generator class
# =============================================================================

class HypothesisGenerator:
    """
    Maintains a probability distribution over causal mechanics hypotheses.

    Usage:
        gen = HypothesisGenerator()
        for record in replay.records:
            scene = perceive(record.final_grid)
            agent_state = tracker.update(...)
            events = interpret_cascade(record.frame)
            gen.observe(scene, agent_state, events, record.action_input.id)
            top_hypotheses = gen.get_top_hypotheses(k=10)
    """

    def __init__(self, top_k: int = 30, prior: float = 0.1,
                 confidence_floor: float = 0.05):
        self.top_k = top_k
        self.default_prior = prior
        self.confidence_floor = confidence_floor
        self.hypotheses: list[Hypothesis] = []
        self._observation_count: int = 0
        self._hypothesis_generation_count: int = 0
        self._pruning_count: int = 0

    def observe(
        self,
        scene: SceneGraph,
        agent_state: AgentState | None,
        events: list[CascadeEvent],
        action_id: int,
    ) -> None:
        """
        Process one observation: update existing hypotheses, generate new ones.

        Args:
            scene: current SceneGraph
            agent_state: current AgentState (or None)
            events: list of CascadeEvents observed this step
            action_id: the action that produced this observation
        """
        self._observation_count += 1

        # 1. Update existing hypotheses
        for hyp in self.hypotheses:
            trigger_holds = evaluate_trigger(hyp.trigger, scene, agent_state, action_id)

            if trigger_holds and events:
                # Trigger fired AND events occurred — did any event match the effect?
                matched = any(effect_matches_event(hyp.effect, e) for e in events)
                if matched:
                    # Positive evidence: increase confidence
                    hyp.confidence = bayesian_update(hyp.confidence, likelihood=0.8)
                    hyp.support += 1
                else:
                    # Trigger fired but expected effect didn't occur — decrease confidence
                    hyp.confidence = bayesian_update(hyp.confidence, likelihood=0.2)
            # If trigger didn't fire, no update (we don't learn from non-firing)

        # 2. Generate new hypotheses from events
        for event in events:
            new_hyps = generate_hypotheses_from_event(
                event, scene, agent_state, action_id
            )
            for new_hyp in new_hyps:
                # Check if we already have a similar hypothesis
                if not self._has_similar(new_hyp):
                    self.hypotheses.append(new_hyp)
                    self._hypothesis_generation_count += 1

        # 3. Prune
        self._prune()

    def _has_similar(self, hypothesis: Hypothesis) -> bool:
        """Check if a similar hypothesis already exists."""
        return any(self._hypotheses_similar(existing, hypothesis) for existing in self.hypotheses)

    def _hypotheses_similar(self, a: Hypothesis, b: Hypothesis) -> bool:
        """Check if two hypotheses are semantically similar."""
        # Same effect type
        if a.effect.get("type") != b.effect.get("type"):
            return False
        # Same effect target color (if applicable)
        a_color = a.effect.get("args", {}).get("color")
        b_color = b.effect.get("args", {}).get("color")
        if a_color != b_color:
            return False
        # Same number of conditions
        if len(a.trigger.conditions) != len(b.trigger.conditions):
            return False
        # Same condition predicates (order-insensitive)
        a_preds = sorted(c.predicate for c in a.trigger.conditions)
        b_preds = sorted(c.predicate for c in b.trigger.conditions)
        return a_preds == b_preds

    def _prune(self) -> None:
        """Prune low-confidence hypotheses, keep top-K with diversity."""
        if len(self.hypotheses) <= self.top_k:
            return

        # Score: confidence × support × MDL simplicity bonus
        scored = [
            (h, h.confidence * h.support * (1.0 / max(1, h.mdl_cost)))
            for h in self.hypotheses
        ]
        scored.sort(key=lambda x: -x[1])

        # Diversity filter: don't keep 5 variants of the same hypothesis
        kept: list[Hypothesis] = []
        for hyp, _ in scored:
            if all(not self._hypotheses_similar(hyp, k) for k in kept):
                kept.append(hyp)
            if len(kept) >= self.top_k:
                break

        self._pruning_count += len(self.hypotheses) - len(kept)
        self.hypotheses = kept

    def get_top_hypotheses(self, k: int = 10) -> list[Hypothesis]:
        """Return the top-k hypotheses by confidence × support."""
        scored = sorted(
            self.hypotheses,
            key=lambda h: -h.confidence * h.support,
        )
        return scored[:k]

    def get_confirmed_hypotheses(self, threshold: float = 0.7) -> list[Hypothesis]:
        """Return hypotheses above the confidence threshold."""
        return [h for h in self.hypotheses if h.confidence >= threshold]

    def get_stats(self) -> dict:
        return {
            "observation_count": self._observation_count,
            "hypothesis_count": len(self.hypotheses),
            "hypothesis_generation_count": self._hypothesis_generation_count,
            "pruning_count": self._pruning_count,
            "avg_confidence": (
                sum(h.confidence for h in self.hypotheses) / len(self.hypotheses)
                if self.hypotheses else 0.0
            ),
            "max_confidence": max((h.confidence for h in self.hypotheses), default=0.0),
            "confirmed_count": len(self.get_confirmed_hypotheses()),
        }


# =============================================================================
# Convenience: run on a replay
# =============================================================================

def run_on_replay(replay, max_steps: int | None = None) -> tuple[HypothesisGenerator, dict]:
    """
    Run the HypothesisGenerator over an entire replay.

    Returns:
        (generator, stats)
    """
    import numpy as np

    from .agent_tracker import AgentStateTracker
    from .cascade_interpreter import interpret_cascade
    from .perception import detect_background_color, perceive

    gen = HypothesisGenerator()
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
        events = []
        if record.n_subframes > 1:
            with __import__("contextlib").suppress(Exception):
                _, events = interpret_cascade(record.frame)
        elif i > 0:
            # Diff against previous frame
            prev_grid = np.array(replay.records[i-1].final_grid, dtype=np.int8)
            from .cascade_interpreter import classify_event, diff_grids
            diff = diff_grids(prev_grid, grid)
            events = classify_event(diff, None, None, timestep=0)

        gen.observe(scene, agent_state, events, record.action_input.id)

    return gen, gen.get_stats()
