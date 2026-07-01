"""Unit tests for world_model.py (Module 9)."""
import pytest

from alphamoo.schemas import (
    ActionId,
    AgentState,
    Condition,
    Hypothesis,
    SceneGraph,
    Trigger,
)
from alphamoo.world_model import (
    CausalRule,
    WorldModel,
)


@pytest.fixture
def confirmed_hypothesis():
    """A hypothesis above the confidence threshold."""
    return Hypothesis(
        trigger=Trigger(conditions=[
            Condition(predicate="agent_touches", args={"color": 5})
        ]),
        effect={"type": "obj_appears", "args": {"color": 3}},
        confidence=0.85,
        support=10,
        mdl_cost=20,
    )


@pytest.fixture
def unconfirmed_hypothesis():
    """A hypothesis below the confidence threshold."""
    return Hypothesis(
        trigger=Trigger(conditions=[
            Condition(predicate="agent_at", args={"position": (10, 10)})
        ]),
        effect={"type": "obj_disappears", "args": {"color": 2}},
        confidence=0.3,
        support=2,
        mdl_cost=15,
    )


@pytest.fixture
def empty_scene():
    return SceneGraph(objects={}, edges=set(), hash="empty")


@pytest.fixture
def agent_at_center():
    return AgentState(
        object_ids=[], position=(32, 32), orientation=0,
        shape="abc", color=[1], energy=None, inventory=[],
    )


class TestCausalRule:
    def test_from_hypothesis(self, confirmed_hypothesis):
        rule = CausalRule.from_hypothesis(confirmed_hypothesis)
        assert rule.confidence == 0.85
        assert rule.support == 10
        assert rule.effect == confirmed_hypothesis.effect

    def test_prediction_accuracy_default(self, confirmed_hypothesis):
        rule = CausalRule.from_hypothesis(confirmed_hypothesis)
        # No predictions recorded → default 1.0
        assert rule.prediction_accuracy == 1.0

    def test_prediction_accuracy_with_results(self, confirmed_hypothesis):
        rule = CausalRule.from_hypothesis(confirmed_hypothesis)
        rule.prediction_correct_count = 8
        rule.prediction_incorrect_count = 2
        assert rule.prediction_accuracy == 0.8


class TestWorldModelUpdate:
    def test_update_adds_confirmed_hypotheses(self, confirmed_hypothesis):
        wm = WorldModel()
        wm.update_from_hypotheses([confirmed_hypothesis])
        assert len(wm.rules) == 1
        assert wm.rules[0].confidence == 0.85

    def test_update_skips_unconfirmed(self, unconfirmed_hypothesis):
        wm = WorldModel()
        wm.update_from_hypotheses([unconfirmed_hypothesis])
        assert len(wm.rules) == 0

    def test_update_removes_dropped_hypotheses(self, confirmed_hypothesis):
        wm = WorldModel()
        wm.update_from_hypotheses([confirmed_hypothesis])
        assert len(wm.rules) == 1
        # Now the hypothesis drops below threshold
        dropped = Hypothesis(
            trigger=confirmed_hypothesis.trigger,
            effect=confirmed_hypothesis.effect,
            confidence=0.3,  # below threshold
            support=10,
            mdl_cost=20,
        )
        wm.update_from_hypotheses([dropped])
        assert len(wm.rules) == 0

    def test_update_merges_existing_rules(self, confirmed_hypothesis):
        wm = WorldModel()
        wm.update_from_hypotheses([confirmed_hypothesis])
        # Same hypothesis with higher confidence
        updated = Hypothesis(
            trigger=confirmed_hypothesis.trigger,
            effect=confirmed_hypothesis.effect,
            confidence=0.95,
            support=20,
            mdl_cost=20,
        )
        wm.update_from_hypotheses([updated])
        assert len(wm.rules) == 1
        assert wm.rules[0].confidence == 0.95
        assert wm.rules[0].support == 20


class TestWorldModelPredict:
    def test_predict_with_no_rules(self, empty_scene, agent_at_center):
        wm = WorldModel()
        prediction = wm.predict(empty_scene, agent_at_center, ActionId.UP)
        assert prediction.predicted_agent_state.position == (32, 31)  # moved up
        assert len(prediction.triggered_rules) == 0
        assert prediction.confidence == 1.0  # no rules → default confident

    def test_predict_movement_up(self, empty_scene, agent_at_center):
        wm = WorldModel()
        prediction = wm.predict(empty_scene, agent_at_center, ActionId.UP)
        assert prediction.predicted_agent_state.position == (32, 31)

    def test_predict_movement_down(self, empty_scene, agent_at_center):
        wm = WorldModel()
        prediction = wm.predict(empty_scene, agent_at_center, ActionId.DOWN)
        assert prediction.predicted_agent_state.position == (32, 33)

    def test_predict_movement_left(self, empty_scene, agent_at_center):
        wm = WorldModel()
        prediction = wm.predict(empty_scene, agent_at_center, ActionId.LEFT)
        assert prediction.predicted_agent_state.position == (31, 32)

    def test_predict_movement_right(self, empty_scene, agent_at_center):
        wm = WorldModel()
        prediction = wm.predict(empty_scene, agent_at_center, ActionId.RIGHT)
        assert prediction.predicted_agent_state.position == (33, 32)

    def test_predict_movement_at_edge(self, empty_scene):
        agent = AgentState(
            object_ids=[], position=(0, 0), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        wm = WorldModel()
        prediction = wm.predict(empty_scene, agent, ActionId.LEFT)
        # Should stay at (0, 0) — can't go left
        assert prediction.predicted_agent_state.position == (0, 0)

    def test_predict_with_triggered_rule(self, empty_scene):
        """Rule: agent_at(32,32) → obj_appears(color=3)."""
        wm = WorldModel()
        wm.update_from_hypotheses([Hypothesis(
            trigger=Trigger(conditions=[
                Condition(predicate="agent_at", args={"position": (32, 32)})
            ]),
            effect={"type": "obj_appears", "args": {"color": 3}},
            confidence=0.9, support=5, mdl_cost=20,
        )])
        agent = AgentState(
            object_ids=[], position=(32, 32), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        prediction = wm.predict(empty_scene, agent, ActionId.UP)
        # Rule should trigger (agent is at 32,32)
        assert len(prediction.triggered_rules) == 1
        assert len(prediction.predicted_events) >= 1
        # Confidence = product of triggered rule confidences
        assert prediction.confidence == 0.9


class TestWorldModelSimplify:
    def test_simplify_removes_poor_rules(self):
        wm = WorldModel()
        # Add a rule with poor accuracy and low support
        rule = CausalRule(
            trigger_conditions=[Condition(predicate="agent_at", args={"position": (10, 10)})],
            trigger_temporal=None,
            effect={"type": "obj_appears", "args": {"color": 1}},
            confidence=0.5, support=2, mdl_cost=20,
            prediction_correct_count=1, prediction_incorrect_count=5,
        )
        wm.rules.append(rule)
        wm.simplify()
        assert len(wm.rules) == 0

    def test_simplify_keeps_winning_rules(self):
        wm = WorldModel()
        rule = CausalRule(
            trigger_conditions=[Condition(predicate="agent_at", args={"position": (10, 10)})],
            trigger_temporal=None,
            effect={"type": "obj_appears", "args": {"color": 1}},
            confidence=0.5, support=2, mdl_cost=20,
            prediction_correct_count=1, prediction_incorrect_count=5,
            last_used_in_winning_plan=True,
        )
        wm.rules.append(rule)
        wm.simplify()
        assert len(wm.rules) == 1  # kept because of winning plan

    def test_simplify_keeps_episode_ends_rules(self):
        wm = WorldModel()
        rule = CausalRule(
            trigger_conditions=[Condition(predicate="agent_at", args={"position": (10, 10)})],
            trigger_temporal=None,
            effect={"type": "episode_ends", "args": {"outcome": "WIN"}},
            confidence=0.5, support=2, mdl_cost=20,
            prediction_correct_count=1, prediction_incorrect_count=5,
        )
        wm.rules.append(rule)
        wm.simplify()
        assert len(wm.rules) == 1  # kept because effect is episode_ends


class TestWorldModelRebuild:
    def test_rebuild_removes_low_accuracy_rules(self):
        wm = WorldModel()
        good_rule = CausalRule(
            trigger_conditions=[], trigger_temporal=None,
            effect={"type": "obj_appears", "args": {}},
            confidence=0.9, support=10, mdl_cost=20,
            prediction_correct_count=9, prediction_incorrect_count=1,
        )
        bad_rule = CausalRule(
            trigger_conditions=[], trigger_temporal=None,
            effect={"type": "obj_appears", "args": {}},
            confidence=0.3, support=5, mdl_cost=20,
            prediction_correct_count=1, prediction_incorrect_count=9,
        )
        wm.rules = [good_rule, bad_rule]
        wm.rebuild_from_surviving_rules()
        assert len(wm.rules) == 1
        assert wm.rules[0].confidence == 0.9

    def test_should_rebuild_below_threshold(self):
        wm = WorldModel()
        # 20% mismatch rate — below 30% threshold
        for _i in range(8):
            wm.record_mismatch(False)
        for _i in range(2):
            wm.record_mismatch(True)
        assert not wm.should_rebuild()

    def test_should_rebuild_above_threshold(self):
        wm = WorldModel()
        # 40% mismatch rate — above 30% threshold
        for _i in range(6):
            wm.record_mismatch(False)
        for _i in range(4):
            wm.record_mismatch(True)
        assert wm.should_rebuild()


class TestWorldModelStats:
    def test_avg_rule_confidence(self):
        wm = WorldModel()
        wm.rules = [
            CausalRule(trigger_conditions=[], trigger_temporal=None,
                       effect={}, confidence=0.8, support=1, mdl_cost=10),
            CausalRule(trigger_conditions=[], trigger_temporal=None,
                       effect={}, confidence=0.6, support=1, mdl_cost=10),
        ]
        assert wm.avg_rule_confidence() == 0.7

    def test_avg_rule_confidence_empty(self):
        wm = WorldModel()
        assert wm.avg_rule_confidence() == 0.0

    def test_rule_entropy_empty(self):
        wm = WorldModel()
        assert wm.rule_entropy() == 0.0

    def test_rule_entropy_with_rules(self):
        wm = WorldModel()
        wm.rules = [
            CausalRule(trigger_conditions=[], trigger_temporal=None,
                       effect={}, confidence=0.5, support=1, mdl_cost=10),
            CausalRule(trigger_conditions=[], trigger_temporal=None,
                       effect={}, confidence=0.5, support=1, mdl_cost=10),
        ]
        # Two rules with equal confidence → max entropy = 1.0 bit
        assert wm.rule_entropy() > 0.9

    def test_get_stats(self, empty_scene, agent_at_center):
        wm = WorldModel()
        wm.predict(empty_scene, agent_at_center, ActionId.UP)
        stats = wm.get_stats()
        assert stats["prediction_count"] == 1
        assert stats["rule_count"] == 0
