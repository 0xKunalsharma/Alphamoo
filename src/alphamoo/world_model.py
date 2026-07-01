"""
AlphaMoo v4.1 — Executable World Model (Module 9).

A Python simulator that predicts the next state given current state + action.
Built from confirmed hypotheses (confidence > 0.7) from Module 5.

The world model is the agent's internal "physics engine" for the game.
It applies confirmed causal rules to predict what will happen when an
action is taken. The Verifier (Module 10) compares predictions to
observations and updates the model when wrong.

Key operations:
  - predict(action) → predicted SceneGraph + list of predicted events
  - simplify() — MDL pruning with importance weighting (don't delete
    rare-but-critical rules)
  - rebuild_from_surviving_rules() — full rebuild when mismatch rate > 30%
  - avg_rule_confidence() / rule_entropy() — for Planner Interface selection
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from .hypothesis_generator import (
    EffectType,
    evaluate_trigger,
)
from .schemas import (
    GRID_SIZE,
    ActionId,
    AgentState,
    CascadeEvent,
    GameObject,
    Hypothesis,
    SceneGraph,
)

# =============================================================================
# Configuration
# =============================================================================

CONFIDENCE_THRESHOLD = 0.7  # rules above this confidence are included in WM
SIMPLIFY_WINDOW = 10         # check last N observations for pruning decisions
MISMATCH_REBUILD_THRESHOLD = 0.30  # rebuild WM if mismatch rate exceeds this


# =============================================================================
# Causal Rule — a confirmed hypothesis promoted to the world model
# =============================================================================

@dataclass
class CausalRule:
    """A confirmed causal rule, promoted from Hypothesis to world model."""
    trigger_conditions: list  # list of Condition
    trigger_temporal: dict | None
    effect: dict
    confidence: float
    support: int
    mdl_cost: int
    last_used_in_winning_plan: bool = False
    last_used_step: int = 0
    prediction_correct_count: int = 0
    prediction_incorrect_count: int = 0

    @classmethod
    def from_hypothesis(cls, hyp: Hypothesis) -> CausalRule:
        return cls(
            trigger_conditions=hyp.trigger.conditions,
            trigger_temporal=hyp.trigger.temporal,
            effect=hyp.effect,
            confidence=hyp.confidence,
            support=hyp.support,
            mdl_cost=hyp.mdl_cost,
        )

    def evaluate_trigger(self, scene: SceneGraph,
                         agent_state: AgentState | None,
                         action_id: int) -> bool:
        """Check if this rule's trigger currently holds."""
        from .hypothesis_generator import Trigger
        trigger = Trigger(
            conditions=self.trigger_conditions,
            temporal=self.trigger_temporal,
        )
        return evaluate_trigger(trigger, scene, agent_state, action_id)

    @property
    def prediction_accuracy(self) -> float:
        """Fraction of predictions that were correct."""
        total = self.prediction_correct_count + self.prediction_incorrect_count
        if total == 0:
            return 1.0  # optimistic default
        return self.prediction_correct_count / total


# =============================================================================
# Prediction result
# =============================================================================

@dataclass
class WorldModelPrediction:
    """Result of predicting the next state."""
    predicted_scene: SceneGraph
    predicted_agent_state: AgentState | None
    predicted_events: list[CascadeEvent]
    triggered_rules: list[CausalRule]
    confidence: float  # overall confidence in the prediction


# =============================================================================
# Executable World Model
# =============================================================================

class WorldModel:
    """
    An executable Python simulator built from confirmed causal rules.

    Usage:
        wm = WorldModel()
        wm.update_from_hypotheses(hyp_gen.get_confirmed_hypotheses())

        # Before taking an action:
        prediction = wm.predict(scene, agent_state, action_id)

        # After observing the outcome:
        verifier_result = wm.verify(prediction, observed_scene, observed_events)

        # Periodically:
        wm.simplify()
    """

    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.rules: list[CausalRule] = []
        self.scene_graph: SceneGraph | None = None
        self._mismatch_history: list[bool] = []  # True = mismatch, False = correct
        self._rebuild_count: int = 0
        self._simplify_count: int = 0
        self._prediction_count: int = 0

    def update_from_hypotheses(self, hypotheses: list[Hypothesis]) -> None:
        """
        Update the world model from a list of hypotheses.

        Adds new rules for hypotheses above the confidence threshold.
        Updates existing rules' confidence and support.
        Removes rules whose source hypothesis has dropped below threshold.
        """
        # Index existing rules by their hypothesis signature
        existing = {self._rule_signature(r): r for r in self.rules}

        new_rules: list[CausalRule] = []
        seen_signatures: set[str] = set()

        for hyp in hypotheses:
            if hyp.confidence < self.confidence_threshold:
                continue
            sig = self._hypothesis_signature(hyp)
            seen_signatures.add(sig)
            if sig in existing:
                # Update existing rule
                rule = existing[sig]
                rule.confidence = hyp.confidence
                rule.support = hyp.support
                new_rules.append(rule)
            else:
                # Add new rule
                new_rules.append(CausalRule.from_hypothesis(hyp))

        # Drop rules whose hypothesis is no longer confirmed
        self.rules = new_rules

    def _rule_signature(self, rule: CausalRule) -> str:
        """Compute a signature for a rule (for dedup)."""
        conds = sorted(
            f"{c.predicate}:{c.negated}:{sorted(c.args.items())}"
            for c in rule.trigger_conditions
        )
        eff = f"{rule.effect.get('type')}:{sorted(rule.effect.get('args', {}).items())}"
        return f"{'|'.join(conds)}>>{eff}"

    def _hypothesis_signature(self, hyp: Hypothesis) -> str:
        """Compute a signature for a hypothesis (matches _rule_signature format)."""
        conds = sorted(
            f"{c.predicate}:{c.negated}:{sorted(c.args.items())}"
            for c in hyp.trigger.conditions
        )
        eff = f"{hyp.effect.get('type')}:{sorted(hyp.effect.get('args', {}).items())}"
        return f"{'|'.join(conds)}>>{eff}"

    def predict(
        self,
        scene: SceneGraph,
        agent_state: AgentState | None,
        action_id: int,
    ) -> WorldModelPrediction:
        """
        Predict the next state given current state + action.

        Applies all rules whose triggers hold, simulating their effects.
        Returns a prediction with the resulting SceneGraph and events.
        """
        self._prediction_count += 1

        # Start with a copy of the current scene
        predicted_objects = {oid: copy.deepcopy(obj) for oid, obj in scene.objects.items()}
        predicted_edges = set(scene.edges)
        predicted_agent = copy.deepcopy(agent_state) if agent_state else None
        predicted_events: list[CascadeEvent] = []
        triggered_rules: list[CausalRule] = []

        # Apply agent movement first (if movement action)
        if predicted_agent and action_id in (
            ActionId.UP, ActionId.DOWN, ActionId.LEFT, ActionId.RIGHT
        ):
            new_pos = self._simulate_movement(predicted_agent.position, action_id)
            predicted_agent.position = new_pos

        # Apply each rule whose trigger holds
        for rule in self.rules:
            if rule.evaluate_trigger(scene, agent_state, action_id):
                triggered_rules.append(rule)
                rule.last_used_step = self._prediction_count
                event = self._apply_effect(rule.effect, predicted_objects, predicted_agent)
                if event is not None:
                    predicted_events.append(event)

        # Build predicted scene
        predicted_scene = SceneGraph(
            objects=predicted_objects,
            edges=predicted_edges,
            agent_id=scene.agent_id,
            grid=None,  # don't reconstruct grid; use objects
            hash="",    # will be computed if needed
        )

        # Overall confidence = product of triggered rule confidences
        # (or 1.0 if no rules triggered — we just predict "nothing changes")
        confidence = math.prod(r.confidence for r in triggered_rules) if triggered_rules else 1.0

        return WorldModelPrediction(
            predicted_scene=predicted_scene,
            predicted_agent_state=predicted_agent,
            predicted_events=predicted_events,
            triggered_rules=triggered_rules,
            confidence=confidence,
        )

    def _simulate_movement(
        self, position: tuple[int, int], action_id: int
    ) -> tuple[int, int]:
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

    def _apply_effect(
        self,
        effect: dict,
        objects: dict[str, GameObject],
        agent_state: AgentState | None,
    ) -> CascadeEvent | None:
        """
        Apply an effect to the predicted objects/agent.
        Returns a CascadeEvent describing what changed.
        """
        eff_type = effect.get("type")
        args = effect.get("args", {})

        if eff_type == EffectType.OBJ_APPEARS.value:
            color = args.get("color")
            return CascadeEvent(
                type="appearance",
                target_color=color,
                after={"n_cells": 1},
            )

        if eff_type == EffectType.OBJ_DISAPPEARS.value:
            color = args.get("color")
            # Remove objects of this color from prediction
            to_remove = [oid for oid, obj in objects.items() if obj.color == color]
            for oid in to_remove:
                del objects[oid]
            return CascadeEvent(
                type="disappearance",
                target_color=color,
                before={"n_cells": 1},
            )

        if eff_type == EffectType.AGENT_STATE_CHANGES.value:
            if agent_state is None:
                return None
            attr = args.get("attr")
            new_val = args.get("new")
            old_val = args.get("old")
            if attr == "color" and new_val is not None:
                if old_val and old_val in agent_state.color:
                    agent_state.color = [
                        new_val if c == old_val else c for c in agent_state.color
                    ]
                elif new_val not in agent_state.color:
                    agent_state.color.append(new_val)
                return CascadeEvent(
                    type="color_change",
                    target_color=new_val,
                    before={"old_color": old_val},
                    after={"new_color": new_val},
                )
            return None

        if eff_type == EffectType.AGENT_DISPLACE.value:
            if agent_state is None:
                return None
            dx, dy = args if isinstance(args, (list, tuple)) else (0, 0)
            x, y = agent_state.position
            agent_state.position = (
                max(0, min(GRID_SIZE - 1, x + dx)),
                max(0, min(GRID_SIZE - 1, y + dy)),
            )
            return CascadeEvent(
                type="move",
                target_color=agent_state.color[0] if agent_state.color else None,
                after={"displacement": (dx, dy)},
            )

        if eff_type == EffectType.EPISODE_ENDS.value:
            return CascadeEvent(
                type="level_transition",
                after={"outcome": args.get("outcome", "WIN")},
            )

        return None

    def record_prediction_result(
        self, rule: CausalRule, correct: bool
    ) -> None:
        """Record whether a rule's prediction was correct."""
        if correct:
            rule.prediction_correct_count += 1
        else:
            rule.prediction_incorrect_count += 1
            # Downgrade confidence
            rule.confidence = max(0.01, rule.confidence * 0.8)

    def record_mismatch(self, mismatch: bool) -> None:
        """Record a verification result for rebuild-trigger analysis."""
        self._mismatch_history.append(mismatch)
        if len(self._mismatch_history) > SIMPLIFY_WINDOW * 2:
            self._mismatch_history = self._mismatch_history[-SIMPLIFY_WINDOW * 2:]

    def should_rebuild(self) -> bool:
        """True if mismatch rate over last window exceeds threshold."""
        if len(self._mismatch_history) < SIMPLIFY_WINDOW:
            return False
        recent = self._mismatch_history[-SIMPLIFY_WINDOW:]
        mismatch_rate = sum(recent) / len(recent)
        return mismatch_rate > MISMATCH_REBUILD_THRESHOLD

    def rebuild_from_surviving_rules(self) -> None:
        """Full rebuild from high-confidence rules only."""
        self._rebuild_count += 1
        # Keep only rules with prediction accuracy > 50% and confidence > 0.5
        self.rules = [
            r for r in self.rules
            if r.prediction_accuracy > 0.5 and r.confidence > 0.5
        ]
        self._mismatch_history.clear()

    def simplify(self) -> None:
        """
        MDL pruning with importance weighting (v4.1 fix).

        Remove rules whose removal doesn't decrease prediction accuracy,
        UNLESS:
          - The rule was last used in a winning plan
          - The rule's removal would break prediction on critical events
        """
        self._simplify_count += 1
        surviving: list[CausalRule] = []

        for rule in self.rules:
            # Don't remove rules that contributed to winning plans
            if rule.last_used_in_winning_plan:
                surviving.append(rule)
                continue
            # Don't remove rules with high prediction accuracy and recent use
            if rule.prediction_accuracy > 0.8 and rule.prediction_correct_count > 0:
                surviving.append(rule)
                continue
            # Don't remove rules whose effect is episode_ends (critical)
            if rule.effect.get("type") == EffectType.EPISODE_ENDS.value:
                surviving.append(rule)
                continue
            # Standard MDL: remove if accuracy is poor and support is low
            if rule.prediction_accuracy < 0.5 and rule.support < 3:
                continue  # drop this rule
            surviving.append(rule)

        self.rules = surviving

    def avg_rule_confidence(self) -> float:
        """Average confidence across all rules."""
        if not self.rules:
            return 0.0
        return sum(r.confidence for r in self.rules) / len(self.rules)

    def rule_entropy(self) -> float:
        """
        Entropy of the rule confidence distribution.
        High entropy = rules are uncertain (prefer MCTS).
        Low entropy = rules are confident (prefer A*).
        """
        if not self.rules:
            return 0.0
        confs = [r.confidence for r in self.rules]
        total = sum(confs)
        if total == 0:
            return 0.0
        probs = [c / total for c in confs]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def get_stats(self) -> dict:
        return {
            "rule_count": len(self.rules),
            "avg_confidence": self.avg_rule_confidence(),
            "rule_entropy": self.rule_entropy(),
            "prediction_count": self._prediction_count,
            "rebuild_count": self._rebuild_count,
            "simplify_count": self._simplify_count,
            "recent_mismatch_rate": (
                sum(self._mismatch_history[-SIMPLIFY_WINDOW:]) /
                max(1, len(self._mismatch_history[-SIMPLIFY_WINDOW:]))
            ),
            "should_rebuild": self.should_rebuild(),
        }
