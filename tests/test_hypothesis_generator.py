"""Unit tests for hypothesis_generator.py (Module 5)."""
import pytest

from alphamoo.hypothesis_generator import (
    CascadeEvent,
    Condition,
    EffectType,
    HypothesisGenerator,
    PredicateType,
    Trigger,
    _compute_mdl_cost,
    _event_to_effect,
    bayesian_update,
    effect_matches_event,
    evaluate_condition,
    evaluate_trigger,
    generate_hypotheses_from_event,
)
from alphamoo.schemas import AgentState, GameObject, SceneGraph

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def empty_scene():
    return SceneGraph(objects={}, edges=set(), agent_id=None)


@pytest.fixture
def agent_at_center():
    return AgentState(
        object_ids=[],
        position=(32, 32),
        orientation=0,
        shape="abc",
        color=[1],
        energy=None,
        inventory=[],
    )


@pytest.fixture
def scene_with_object():
    """Scene with a red object at (10, 10)."""
    obj = GameObject(
        id="obj_001",
        color=2,  # red
        secondary_colors=[],
        cells=[(10, 10), (10, 11), (11, 10), (11, 11)],
        bounding_box=(10, 10, 11, 11),
        topology="solid",
        shape_hash="xyz",
        is_agent=False,
    )
    return SceneGraph(
        objects={"obj_001": obj},
        edges=set(),
        agent_id=None,
    )


@pytest.fixture
def agent_adjacent_to_object():
    """Agent at (12, 10) — adjacent to the object at (10-11, 10-11)."""
    return AgentState(
        object_ids=[],
        position=(12, 10),
        orientation=0,
        shape="abc",
        color=[1],
        energy=None,
        inventory=[],
    )


# =============================================================================
# Bayesian update tests
# =============================================================================

class TestBayesianUpdate:
    def test_positive_evidence_increases_confidence(self):
        prior = 0.5
        posterior = bayesian_update(prior, likelihood=0.8)
        assert posterior > prior

    def test_negative_evidence_decreases_confidence(self):
        prior = 0.5
        posterior = bayesian_update(prior, likelihood=0.2)
        assert posterior < prior

    def test_neutral_evidence_stays_same(self):
        prior = 0.5
        posterior = bayesian_update(prior, likelihood=0.5)
        assert abs(posterior - prior) < 0.01

    def test_clamps_to_valid_range(self):
        # Very high prior with positive evidence should not exceed 0.999
        posterior = bayesian_update(0.99, likelihood=0.99)
        assert posterior <= 0.999
        # Very low prior with negative evidence should not go below 0.001
        posterior = bayesian_update(0.01, likelihood=0.01)
        assert posterior >= 0.001

    def test_repeated_updates_converge(self):
        """Repeated positive evidence should converge toward 1."""
        prior = 0.1
        for _ in range(20):
            prior = bayesian_update(prior, likelihood=0.8)
        assert prior > 0.9


# =============================================================================
# Condition evaluation tests
# =============================================================================

class TestConditionEvaluation:
    def test_agent_at_condition_matches(self, empty_scene, agent_at_center):
        cond = Condition(
            predicate=PredicateType.AGENT_AT.value,
            args={"position": (32, 32)},
        )
        assert evaluate_condition(cond, empty_scene, agent_at_center, action_id=1)

    def test_agent_at_condition_no_match(self, empty_scene, agent_at_center):
        cond = Condition(
            predicate=PredicateType.AGENT_AT.value,
            args={"position": (10, 10)},
        )
        assert not evaluate_condition(cond, empty_scene, agent_at_center, action_id=1)

    def test_agent_at_no_agent(self, empty_scene):
        cond = Condition(
            predicate=PredicateType.AGENT_AT.value,
            args={"position": (32, 32)},
        )
        assert not evaluate_condition(cond, empty_scene, None, action_id=1)

    def test_agent_touches_color(self, scene_with_object, agent_adjacent_to_object):
        """Agent at (12,10) touches object at (10-11, 10-11) of color 2."""
        cond = Condition(
            predicate=PredicateType.AGENT_TOUCHES.value,
            args={"color": 2},
        )
        assert evaluate_condition(cond, scene_with_object, agent_adjacent_to_object, action_id=1)

    def test_agent_touches_wrong_color(self, scene_with_object, agent_adjacent_to_object):
        cond = Condition(
            predicate=PredicateType.AGENT_TOUCHES.value,
            args={"color": 3},
        )
        assert not evaluate_condition(cond, scene_with_object, agent_adjacent_to_object, action_id=1)

    def test_agent_state_is_color(self, agent_at_center):
        cond = Condition(
            predicate=PredicateType.AGENT_STATE_IS.value,
            args={"attr": "color", "value": 1},
        )
        assert evaluate_condition(cond, SceneGraph(objects={}, edges=set()), agent_at_center, action_id=1)

    def test_negated_condition(self, empty_scene, agent_at_center):
        cond = Condition(
            predicate=PredicateType.AGENT_AT.value,
            args={"position": (10, 10)},
            negated=True,
        )
        # Agent is at (32,32), not (10,10) → negated condition is True
        assert evaluate_condition(cond, empty_scene, agent_at_center, action_id=1)


# =============================================================================
# Trigger evaluation tests
# =============================================================================

class TestTriggerEvaluation:
    def test_conjunctive_trigger_all_match(self, scene_with_object, agent_adjacent_to_object):
        trigger = Trigger(conditions=[
            Condition(predicate=PredicateType.AGENT_TOUCHES.value, args={"color": 2}),
            Condition(predicate=PredicateType.AGENT_STATE_IS.value, args={"attr": "color", "value": 1}),
        ])
        assert evaluate_trigger(trigger, scene_with_object, agent_adjacent_to_object, action_id=1)

    def test_conjunctive_trigger_partial_match(self, scene_with_object, agent_adjacent_to_object):
        trigger = Trigger(conditions=[
            Condition(predicate=PredicateType.AGENT_TOUCHES.value, args={"color": 2}),
            Condition(predicate=PredicateType.AGENT_STATE_IS.value, args={"attr": "color", "value": 99}),
        ])
        assert not evaluate_trigger(trigger, scene_with_object, agent_adjacent_to_object, action_id=1)

    def test_empty_trigger_always_true(self, empty_scene, agent_at_center):
        trigger = Trigger(conditions=[])
        assert evaluate_trigger(trigger, empty_scene, agent_at_center, action_id=1)


# =============================================================================
# Effect-event matching tests
# =============================================================================

class TestEffectEventMatching:
    def test_appearance_event_matches(self):
        effect = {"type": EffectType.OBJ_APPEARS.value, "args": {"color": 5}}
        event = CascadeEvent(type="appearance", target_color=5)
        assert effect_matches_event(effect, event)

    def test_appearance_event_wrong_color(self):
        effect = {"type": EffectType.OBJ_APPEARS.value, "args": {"color": 5}}
        event = CascadeEvent(type="appearance", target_color=3)
        assert not effect_matches_event(effect, event)

    def test_appearance_event_no_color_filter(self):
        effect = {"type": EffectType.OBJ_APPEARS.value, "args": {}}
        event = CascadeEvent(type="appearance", target_color=7)
        assert effect_matches_event(effect, event)

    def test_disappearance_event_matches(self):
        effect = {"type": EffectType.OBJ_DISAPPEARS.value, "args": {"color": 5}}
        event = CascadeEvent(type="disappearance", target_color=5)
        assert effect_matches_event(effect, event)

    def test_wrong_event_type(self):
        effect = {"type": EffectType.OBJ_APPEARS.value, "args": {"color": 5}}
        event = CascadeEvent(type="disappearance", target_color=5)
        assert not effect_matches_event(effect, event)


# =============================================================================
# Hypothesis generation tests
# =============================================================================

class TestHypothesisGeneration:
    def test_generate_from_appearance_event(self, scene_with_object, agent_adjacent_to_object):
        event = CascadeEvent(
            type="appearance",
            target_color=5,
            after={"n_cells": 9, "bbox": (30, 30, 32, 32)},
        )
        hyps = generate_hypotheses_from_event(
            event, scene_with_object, agent_adjacent_to_object, action_id=1
        )
        assert len(hyps) > 0
        # All should have the appearance effect
        for h in hyps:
            assert h.effect["type"] == EffectType.OBJ_APPEARS.value
            assert h.effect["args"]["color"] == 5

    def test_generate_from_no_agent(self, scene_with_object):
        event = CascadeEvent(type="appearance", target_color=5)
        hyps = generate_hypotheses_from_event(
            event, scene_with_object, None, action_id=1
        )
        # No agent → no trigger candidates
        assert len(hyps) == 0

    def test_generated_hypotheses_have_low_confidence(self, scene_with_object, agent_adjacent_to_object):
        event = CascadeEvent(type="appearance", target_color=5)
        hyps = generate_hypotheses_from_event(
            event, scene_with_object, agent_adjacent_to_object, action_id=1
        )
        for h in hyps:
            assert h.confidence == 0.1
            assert h.support == 1
            assert h.mdl_cost > 0

    def test_event_to_effect_appearance(self):
        event = CascadeEvent(type="appearance", target_color=3)
        effect = _event_to_effect(event)
        assert effect["type"] == EffectType.OBJ_APPEARS.value
        assert effect["args"]["color"] == 3

    def test_event_to_effect_level_transition(self):
        event = CascadeEvent(type="level_transition")
        effect = _event_to_effect(event)
        assert effect["type"] == EffectType.EPISODE_ENDS.value

    def test_mdl_cost_scales_with_complexity(self):
        simple_trigger = Trigger(conditions=[
            Condition(predicate="agent_at", args={"position": (1, 1)})
        ])
        complex_trigger = Trigger(conditions=[
            Condition(predicate="agent_at", args={"position": (1, 1)}),
            Condition(predicate="agent_state_is", args={"attr": "color", "value": 1}),
            Condition(predicate="agent_touches", args={"color": 2}),
        ])
        effect = {"type": "obj_appears", "args": {"color": 1}}
        simple_cost = _compute_mdl_cost(simple_trigger, effect)
        complex_cost = _compute_mdl_cost(complex_trigger, effect)
        assert complex_cost > simple_cost


# =============================================================================
# HypothesisGenerator integration tests
# =============================================================================

class TestHypothesisGenerator:
    def test_observe_with_no_events(self, empty_scene, agent_at_center):
        gen = HypothesisGenerator()
        gen.observe(empty_scene, agent_at_center, events=[], action_id=1)
        assert gen.get_stats()["observation_count"] == 1
        assert gen.get_stats()["hypothesis_count"] == 0

    def test_observe_generates_hypotheses(self, scene_with_object, agent_adjacent_to_object):
        gen = HypothesisGenerator()
        event = CascadeEvent(type="appearance", target_color=5)
        gen.observe(scene_with_object, agent_adjacent_to_object, [event], action_id=1)
        assert gen.get_stats()["hypothesis_count"] > 0

    def test_positive_evidence_increases_confidence(self, scene_with_object, agent_adjacent_to_object):
        """Observe the same event twice — confidence should increase."""
        gen = HypothesisGenerator()
        event = CascadeEvent(type="appearance", target_color=5)
        # First observation: generates hypotheses
        gen.observe(scene_with_object, agent_adjacent_to_object, [event], action_id=1)
        first_confs = [h.confidence for h in gen.hypotheses]

        # Second observation: should update existing hypotheses
        gen.observe(scene_with_object, agent_adjacent_to_object, [event], action_id=1)
        second_confs = [h.confidence for h in gen.hypotheses]

        # At least some hypotheses should have higher confidence
        assert max(second_confs) >= max(first_confs)

    def test_pruning_keeps_top_k(self, scene_with_object, agent_adjacent_to_object):
        """When hypotheses exceed top_k, prune to keep only top_k."""
        gen = HypothesisGenerator(top_k=5)
        # Generate many different events to create many hypotheses
        for color in range(20):
            event = CascadeEvent(type="appearance", target_color=color)
            gen.observe(scene_with_object, agent_adjacent_to_object, [event], action_id=1)
        assert len(gen.hypotheses) <= 5

    def test_get_top_hypotheses(self, scene_with_object, agent_adjacent_to_object):
        gen = HypothesisGenerator()
        event = CascadeEvent(type="appearance", target_color=5)
        gen.observe(scene_with_object, agent_adjacent_to_object, [event], action_id=1)
        top = gen.get_top_hypotheses(k=3)
        assert len(top) <= 3
        # Sorted by confidence × support (descending)
        for i in range(len(top) - 1):
            assert top[i].confidence * top[i].support >= top[i+1].confidence * top[i+1].support

    def test_confirmed_hypotheses_threshold(self, scene_with_object, agent_adjacent_to_object):
        """After many positive observations, some hypotheses should be confirmed."""
        gen = HypothesisGenerator()
        event = CascadeEvent(type="appearance", target_color=5)
        # Observe 10 times — should boost confidence significantly
        for _ in range(10):
            gen.observe(scene_with_object, agent_adjacent_to_object, [event], action_id=1)
        confirmed = gen.get_confirmed_hypotheses(threshold=0.5)
        assert len(confirmed) >= 1

    def test_stats_tracking(self, scene_with_object, agent_adjacent_to_object):
        gen = HypothesisGenerator()
        event = CascadeEvent(type="appearance", target_color=5)
        gen.observe(scene_with_object, agent_adjacent_to_object, [event], action_id=1)
        stats = gen.get_stats()
        assert stats["observation_count"] == 1
        assert stats["hypothesis_count"] > 0
        assert stats["hypothesis_generation_count"] > 0
        assert stats["avg_confidence"] > 0
