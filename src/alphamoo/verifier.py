"""
AlphaMoo v4.1 — Verifier (Module 10).

Every action produces a prediction (from World Model) and an observation
(from the environment). The Verifier computes the diff and feeds it back
to:
  1. The World Model — downgrade rules that predicted incorrectly
  2. The Hypothesis Generator — unexpected events are new observations
  3. The Experiment Planner — mismatch is new information; update IG

If mismatch rate > 30% over last 10 steps, trigger full world model rebuild.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .hypothesis_generator import effect_matches_event
from .schemas import (
    AgentState,
    CascadeEvent,
    GameObject,
    SceneGraph,
)
from .world_model import CausalRule, WorldModel, WorldModelPrediction

# =============================================================================
# Verification result
# =============================================================================

@dataclass
class VerifierResult:
    """Result of comparing a prediction to an observation."""
    match: bool                          # True if prediction matched observation
    unexpected_events: list[CascadeEvent] = field(default_factory=list)
    failed_predictions: list[dict] = field(default_factory=list)
    severity: float = 0.0                # 0.0 = perfect match, 1.0 = total mismatch
    triggered_rules_correct: list[CausalRule] = field(default_factory=list)
    triggered_rules_incorrect: list[CausalRule] = field(default_factory=list)

    @property
    def mismatch_rate(self) -> float:
        """Fraction of triggered rules that predicted incorrectly."""
        total = len(self.triggered_rules_correct) + len(self.triggered_rules_incorrect)
        if total == 0:
            return 0.0
        return len(self.triggered_rules_incorrect) / total


# =============================================================================
# Scene diff utilities
# =============================================================================

def scene_diff(predicted: SceneGraph, observed: SceneGraph) -> dict:
    """
    Compute the diff between two scene graphs.

    Returns:
        {
            "objects_appeared": [GameObject, ...],  # in observed but not predicted
            "objects_disappeared": [GameObject, ...],  # in predicted but not observed
            "objects_modified": [(predicted_obj, observed_obj), ...],
            "n_differences": int,
        }
    """
    predicted_ids = set(predicted.objects.keys())
    observed_ids = set(observed.objects.keys())

    appeared_ids = observed_ids - predicted_ids
    disappeared_ids = predicted_ids - observed_ids
    common_ids = predicted_ids & observed_ids

    appeared = [observed.objects[oid] for oid in appeared_ids]
    disappeared = [predicted.objects[oid] for oid in disappeared_ids]
    modified = []
    for oid in common_ids:
        p = predicted.objects[oid]
        o = observed.objects[oid]
        if _objects_differ(p, o):
            modified.append((p, o))

    return {
        "objects_appeared": appeared,
        "objects_disappeared": disappeared,
        "objects_modified": modified,
        "n_differences": len(appeared) + len(disappeared) + len(modified),
    }


def _objects_differ(a: GameObject, b: GameObject) -> bool:
    """Check if two GameObjects differ in any meaningful way."""
    if a.color != b.color:
        return True
    if a.bounding_box != b.bounding_box:
        return True
    if a.shape_hash != b.shape_hash:
        return True
    return len(a.cells) != len(b.cells)


# =============================================================================
# Verifier
# =============================================================================

class Verifier:
    """
    Compares World Model predictions to observed outcomes.

    Usage:
        verifier = Verifier(world_model)
        # Before action:
        prediction = world_model.predict(scene, agent_state, action_id)
        # After action, observe actual outcome:
        result = verifier.verify(prediction, observed_scene, observed_events, observed_agent_state)
        # Result feeds back to world model and hypothesis generator
    """

    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self._verification_count: int = 0
        self._match_count: int = 0
        self._mismatch_count: int = 0
        self._total_unexpected_events: int = 0
        self._total_failed_predictions: int = 0
        self._rebuilds_triggered: int = 0
        self._recent_results: list[VerifierResult] = []

    def verify(
        self,
        prediction: WorldModelPrediction,
        observed_scene: SceneGraph,
        observed_events: list[CascadeEvent],
        observed_agent_state: AgentState | None = None,
    ) -> VerifierResult:
        """
        Compare a prediction to the actual observation.

        Args:
            prediction: WorldModelPrediction from world_model.predict()
            observed_scene: the actual SceneGraph after the action
            observed_events: the actual CascadeEvents that occurred
            observed_agent_state: the actual agent state after the action

        Returns:
            VerifierResult with match status, unexpected events, failed predictions
        """
        self._verification_count += 1

        result = VerifierResult(match=True)

        # 1. Compare predicted vs observed events
        predicted_events = prediction.predicted_events
        result = self._compare_events(
            result, predicted_events, observed_events
        )

        # 2. Compare predicted vs observed scene
        diff = scene_diff(prediction.predicted_scene, observed_scene)
        if diff["n_differences"] > 0:
            result.match = False
            result.severity = min(1.0, diff["n_differences"] / 10.0)
            # Convert appeared/disappeared objects to events
            for obj in diff["objects_appeared"]:
                result.unexpected_events.append(CascadeEvent(
                    type="appearance",
                    target_color=obj.color,
                    after={"n_cells": len(obj.cells)},
                ))
            for obj in diff["objects_disappeared"]:
                result.unexpected_events.append(CascadeEvent(
                    type="disappearance",
                    target_color=obj.color,
                    before={"n_cells": len(obj.cells)},
                ))

        # 3. Check which triggered rules predicted correctly
        for rule in prediction.triggered_rules:
            # Did any observed event match this rule's effect?
            matched = any(
                effect_matches_event(rule.effect, e) for e in observed_events
            )
            if matched:
                result.triggered_rules_correct.append(rule)
                self.world_model.record_prediction_result(rule, correct=True)
            else:
                # Rule triggered but effect didn't occur
                # OR effect occurred but on different target
                result.triggered_rules_incorrect.append(rule)
                result.failed_predictions.append({
                    "rule": rule,
                    "expected_effect": rule.effect,
                    "observed_events": [e.type for e in observed_events],
                })
                self.world_model.record_prediction_result(rule, correct=False)

        # 4. Update stats
        if result.match and not result.triggered_rules_incorrect:
            self._match_count += 1
        else:
            self._mismatch_count += 1
        self._total_unexpected_events += len(result.unexpected_events)
        self._total_failed_predictions += len(result.failed_predictions)

        # 5. Record mismatch for rebuild-trigger analysis
        is_mismatch = not result.match or len(result.triggered_rules_incorrect) > 0
        self.world_model.record_mismatch(is_mismatch)

        # 6. Check if rebuild needed
        if self.world_model.should_rebuild():
            self.world_model.rebuild_from_surviving_rules()
            self._rebuilds_triggered += 1

        # 7. Keep recent results
        self._recent_results.append(result)
        if len(self._recent_results) > 20:
            self._recent_results = self._recent_results[-20:]

        return result

    def _compare_events(
        self,
        result: VerifierResult,
        predicted: list[CascadeEvent],
        observed: list[CascadeEvent],
    ) -> VerifierResult:
        """Compare predicted vs observed events."""
        # Mark predicted events as matched or not
        matched_observed_indices: set[int] = set()
        for pred_event in predicted:
            matched = False
            for i, obs_event in enumerate(observed):
                if i in matched_observed_indices:
                    continue
                if self._events_match(pred_event, obs_event):
                    matched = True
                    matched_observed_indices.add(i)
                    break
            if not matched:
                result.failed_predictions.append({
                    "predicted_event": pred_event,
                    "observed_events": [e.type for e in observed],
                })
                result.match = False

        # Unexpected events = observed events that weren't predicted
        for i, obs_event in enumerate(observed):
            if i not in matched_observed_indices:
                result.unexpected_events.append(obs_event)

        if result.unexpected_events:
            result.match = False
            result.severity = max(result.severity, min(1.0, len(result.unexpected_events) / 5.0))

        return result

    def _events_match(self, predicted: CascadeEvent, observed: CascadeEvent) -> bool:
        """Check if a predicted event matches an observed event."""
        if predicted.type != observed.type:
            return False
        if predicted.target_color is not None and observed.target_color is not None:
            return predicted.target_color == observed.target_color
        return True

    def get_unexpected_events_for_hypothesis_generator(
        self, result: VerifierResult
    ) -> list[CascadeEvent]:
        """
        Extract unexpected events to feed back to the Hypothesis Generator
        as new observations.
        """
        return result.unexpected_events

    def get_stats(self) -> dict:
        return {
            "verification_count": self._verification_count,
            "match_count": self._match_count,
            "mismatch_count": self._mismatch_count,
            "match_rate": self._match_count / max(1, self._verification_count),
            "mismatch_rate": self._mismatch_count / max(1, self._verification_count),
            "total_unexpected_events": self._total_unexpected_events,
            "total_failed_predictions": self._total_failed_predictions,
            "rebuilds_triggered": self._rebuilds_triggered,
            "avg_severity": (
                sum(r.severity for r in self._recent_results) /
                max(1, len(self._recent_results))
            ),
        }
