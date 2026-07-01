"""Unit tests for verifier.py (Module 10)."""
import pytest

from alphamoo.schemas import (
    ActionId,
    AgentState,
    CascadeEvent,
    GameObject,
    SceneGraph,
)
from alphamoo.verifier import (
    Verifier,
    VerifierResult,
    _objects_differ,
    scene_diff,
)
from alphamoo.world_model import CausalRule, WorldModel


@pytest.fixture
def empty_scene():
    return SceneGraph(objects={}, edges=set(), hash="empty")


@pytest.fixture
def scene_with_obj():
    obj = GameObject(
        id="obj_001", color=2, secondary_colors=[],
        cells=[(10, 10), (10, 11), (11, 10), (11, 11)],
        bounding_box=(10, 10, 11, 11),
        topology="solid", shape_hash="abc", is_agent=False,
    )
    return SceneGraph(objects={"obj_001": obj}, edges=set(), hash="with_obj")


@pytest.fixture
def agent_at_center():
    return AgentState(
        object_ids=[], position=(32, 32), orientation=0,
        shape="abc", color=[1], energy=None, inventory=[],
    )


@pytest.fixture
def world_model():
    return WorldModel()


class TestSceneDiff:
    def test_no_diff(self, scene_with_obj):
        diff = scene_diff(scene_with_obj, scene_with_obj)
        assert diff["n_differences"] == 0

    def test_object_appeared(self, empty_scene, scene_with_obj):
        diff = scene_diff(empty_scene, scene_with_obj)
        assert len(diff["objects_appeared"]) == 1
        assert diff["n_differences"] == 1

    def test_object_disappeared(self, empty_scene, scene_with_obj):
        diff = scene_diff(scene_with_obj, empty_scene)
        assert len(diff["objects_disappeared"]) == 1
        assert diff["n_differences"] == 1

    def test_object_modified_color(self):
        obj_a = GameObject(
            id="o", color=1, secondary_colors=[],
            cells=[(5, 5)], bounding_box=(5, 5, 5, 5),
            topology="solid", shape_hash="x", is_agent=False,
        )
        obj_b = GameObject(
            id="o", color=2, secondary_colors=[],
            cells=[(5, 5)], bounding_box=(5, 5, 5, 5),
            topology="solid", shape_hash="x", is_agent=False,
        )
        scene_a = SceneGraph(objects={"o": obj_a}, edges=set())
        scene_b = SceneGraph(objects={"o": obj_b}, edges=set())
        diff = scene_diff(scene_a, scene_b)
        assert len(diff["objects_modified"]) == 1
        assert diff["n_differences"] == 1

    def test_objects_differ_same(self):
        obj = GameObject(
            id="o", color=1, secondary_colors=[],
            cells=[(5, 5)], bounding_box=(5, 5, 5, 5),
            topology="solid", shape_hash="x", is_agent=False,
        )
        assert not _objects_differ(obj, obj)


class TestVerifier:
    def test_verify_perfect_match(self, empty_scene, agent_at_center):
        wm = WorldModel()
        verifier = Verifier(wm)
        prediction = wm.predict(empty_scene, agent_at_center, ActionId.UP)
        # Observe the same outcome (no events, agent moved up)
        observed_scene = SceneGraph(objects={}, edges=set(), hash="empty")
        result = verifier.verify(prediction, observed_scene, observed_events=[], observed_agent_state=None)
        # With no rules triggered and no events, should match
        assert result.match

    def test_verify_with_unexpected_event(self, empty_scene, agent_at_center):
        wm = WorldModel()
        verifier = Verifier(wm)
        prediction = wm.predict(empty_scene, agent_at_center, ActionId.UP)
        # Observe an unexpected appearance
        observed_events = [CascadeEvent(type="appearance", target_color=5)]
        result = verifier.verify(prediction, empty_scene, observed_events)
        assert not result.match
        assert len(result.unexpected_events) >= 1

    def test_verify_with_correct_prediction(self, empty_scene):
        """Rule predicts appearance, appearance observed → correct."""
        from alphamoo.schemas import Condition, Hypothesis, Trigger
        wm = WorldModel()
        wm.update_from_hypotheses([Hypothesis(
            trigger=Trigger(conditions=[
                Condition(predicate="agent_at", args={"position": (32, 32)})
            ]),
            effect={"type": "obj_appears", "args": {"color": 5}},
            confidence=0.9, support=5, mdl_cost=20,
        )])
        agent = AgentState(
            object_ids=[], position=(32, 32), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        prediction = wm.predict(empty_scene, agent, ActionId.UP)

        verifier = Verifier(wm)
        # Observe the predicted appearance
        observed_events = [CascadeEvent(type="appearance", target_color=5)]
        result = verifier.verify(prediction, empty_scene, observed_events)
        assert len(result.triggered_rules_correct) == 1
        assert len(result.triggered_rules_incorrect) == 0

    def test_verify_with_incorrect_prediction(self, empty_scene):
        """Rule predicts appearance, but no appearance observed → incorrect."""
        from alphamoo.schemas import Condition, Hypothesis, Trigger
        wm = WorldModel()
        wm.update_from_hypotheses([Hypothesis(
            trigger=Trigger(conditions=[
                Condition(predicate="agent_at", args={"position": (32, 32)})
            ]),
            effect={"type": "obj_appears", "args": {"color": 5}},
            confidence=0.9, support=5, mdl_cost=20,
        )])
        agent = AgentState(
            object_ids=[], position=(32, 32), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        prediction = wm.predict(empty_scene, agent, ActionId.UP)

        verifier = Verifier(wm)
        # No events observed
        result = verifier.verify(prediction, empty_scene, observed_events=[])
        assert len(result.triggered_rules_incorrect) == 1
        assert len(result.triggered_rules_correct) == 0
        # Rule's confidence should be downgraded
        assert wm.rules[0].confidence < 0.9

    def test_verify_stats(self, empty_scene, agent_at_center):
        wm = WorldModel()
        verifier = Verifier(wm)
        prediction = wm.predict(empty_scene, agent_at_center, ActionId.UP)
        verifier.verify(prediction, empty_scene, observed_events=[])
        stats = verifier.get_stats()
        assert stats["verification_count"] == 1

    def test_verify_triggers_rebuild(self, empty_scene, agent_at_center):
        """If mismatch rate > 30%, trigger rebuild."""
        from alphamoo.schemas import Condition, Hypothesis, Trigger
        wm = WorldModel()
        wm.update_from_hypotheses([Hypothesis(
            trigger=Trigger(conditions=[
                Condition(predicate="agent_at", args={"position": (32, 32)})
            ]),
            effect={"type": "obj_appears", "args": {"color": 5}},
            confidence=0.9, support=5, mdl_cost=20,
        )])
        agent = AgentState(
            object_ids=[], position=(32, 32), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        verifier = Verifier(wm)
        # 10 mismatches in a row
        for _ in range(10):
            prediction = wm.predict(empty_scene, agent, ActionId.UP)
            verifier.verify(prediction, empty_scene, observed_events=[])
        # Should have triggered at least one rebuild
        stats = verifier.get_stats()
        assert stats["rebuilds_triggered"] >= 1


class TestVerifierResult:
    def test_mismatch_rate_no_rules(self):
        result = VerifierResult(match=True)
        assert result.mismatch_rate == 0.0

    def test_mismatch_rate_all_correct(self):
        rule = CausalRule(
            trigger_conditions=[], trigger_temporal=None,
            effect={}, confidence=0.9, support=1, mdl_cost=10,
        )
        result = VerifierResult(match=True, triggered_rules_correct=[rule])
        assert result.mismatch_rate == 0.0

    def test_mismatch_rate_all_incorrect(self):
        rule = CausalRule(
            trigger_conditions=[], trigger_temporal=None,
            effect={}, confidence=0.9, support=1, mdl_cost=10,
        )
        result = VerifierResult(match=False, triggered_rules_incorrect=[rule])
        assert result.mismatch_rate == 1.0

    def test_mismatch_rate_mixed(self):
        rule = CausalRule(
            trigger_conditions=[], trigger_temporal=None,
            effect={}, confidence=0.9, support=1, mdl_cost=10,
        )
        result = VerifierResult(
            match=False,
            triggered_rules_correct=[rule],
            triggered_rules_incorrect=[rule],
        )
        assert result.mismatch_rate == 0.5
