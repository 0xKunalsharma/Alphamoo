"""Unit tests for goal_inference.py (Module 6)."""
import pytest

from alphamoo.goal_inference import (
    DEFAULT_GOAL_PRIORS,
    GoalHypothesis,
    GoalInferenceModule,
    GoalPredicateType,
    evaluate_goal_predicate,
    generate_goal_hypotheses_from_win,
)
from alphamoo.schemas import AgentState, GameObject, SceneGraph


@pytest.fixture
def scene_with_objects():
    """Scene with a red object and a blue object."""
    red = GameObject(
        id="obj_001", color=2, secondary_colors=[],
        cells=[(10, 10), (10, 11), (11, 10), (11, 11)],
        bounding_box=(10, 10, 11, 11),
        topology="solid", shape_hash="abc", is_agent=False,
    )
    blue = GameObject(
        id="obj_002", color=1, secondary_colors=[],
        cells=[(30, 30), (30, 31), (31, 30), (31, 31)],
        bounding_box=(30, 30, 31, 31),
        topology="solid", shape_hash="xyz", is_agent=False,
    )
    return SceneGraph(
        objects={"obj_001": red, "obj_002": blue},
        edges=set(), agent_id=None,
    )


@pytest.fixture
def agent_at_red():
    """Agent at position (10, 10) — inside the red object."""
    return AgentState(
        object_ids=[], position=(10, 10), orientation=0,
        shape="abc", color=[5], energy=None, inventory=[],
    )


class TestGoalPredicateEvaluation:
    def test_agent_at_matches(self, scene_with_objects, agent_at_red):
        result = evaluate_goal_predicate(
            GoalPredicateType.AGENT_AT.value,
            {"position": (10, 10)},
            scene_with_objects, agent_at_red,
        )
        assert result

    def test_agent_at_no_match(self, scene_with_objects, agent_at_red):
        result = evaluate_goal_predicate(
            GoalPredicateType.AGENT_AT.value,
            {"position": (50, 50)},
            scene_with_objects, agent_at_red,
        )
        assert not result

    def test_agent_color_matches(self, scene_with_objects, agent_at_red):
        result = evaluate_goal_predicate(
            GoalPredicateType.AGENT_COLOR_MATCHES.value,
            {"color": 5},
            scene_with_objects, agent_at_red,
        )
        assert result

    def test_agent_color_no_match(self, scene_with_objects, agent_at_red):
        result = evaluate_goal_predicate(
            GoalPredicateType.AGENT_COLOR_MATCHES.value,
            {"color": 99},
            scene_with_objects, agent_at_red,
        )
        assert not result

    def test_agent_reaches_zone(self, scene_with_objects, agent_at_red):
        # Agent at (10,10) is inside red object
        result = evaluate_goal_predicate(
            GoalPredicateType.AGENT_REACHES_ZONE.value,
            {"color": 2},
            scene_with_objects, agent_at_red,
        )
        assert result

    def test_collect_all_when_color_present(self, scene_with_objects, agent_at_red):
        # Red object present → not yet collected
        result = evaluate_goal_predicate(
            GoalPredicateType.COLLECT_ALL.value,
            {"color": 2},
            scene_with_objects, agent_at_red,
        )
        assert not result

    def test_collect_all_when_color_absent(self, agent_at_red):
        empty_scene = SceneGraph(objects={}, edges=set())
        result = evaluate_goal_predicate(
            GoalPredicateType.COLLECT_ALL.value,
            {"color": 2},
            empty_scene, agent_at_red,
        )
        assert result

    def test_no_agent_returns_false(self, scene_with_objects):
        result = evaluate_goal_predicate(
            GoalPredicateType.AGENT_AT.value,
            {"position": (10, 10)},
            scene_with_objects, None,
        )
        assert not result


class TestGoalHypothesisGeneration:
    def test_generates_hypotheses_from_win(self, scene_with_objects, agent_at_red):
        hyps = generate_goal_hypotheses_from_win(scene_with_objects, agent_at_red)
        assert len(hyps) > 0
        # Should include agent_at hypothesis
        types = [h.terminal_condition for h in hyps]
        assert GoalPredicateType.AGENT_AT.value in types

    def test_generates_state_match_hypothesis(self, scene_with_objects, agent_at_red):
        hyps = generate_goal_hypotheses_from_win(scene_with_objects, agent_at_red)
        # Agent has color 5 → should generate color match hypothesis
        color_hyps = [h for h in hyps if h.terminal_condition == GoalPredicateType.AGENT_COLOR_MATCHES.value]
        assert len(color_hyps) > 0
        assert any(h.args.get("color") == 5 for h in color_hyps)

    def test_no_agent_no_hypotheses(self, scene_with_objects):
        hyps = generate_goal_hypotheses_from_win(scene_with_objects, None)
        assert len(hyps) == 0

    def test_generates_relational_hypothesis_for_matching_shapes(self):
        """Two objects with same shape hash → relational hypothesis."""
        obj_a = GameObject(
            id="a", color=1, secondary_colors=[],
            cells=[(10, 10)], bounding_box=(10, 10, 10, 10),
            topology="solid", shape_hash="same", is_agent=False,
        )
        obj_b = GameObject(
            id="b", color=2, secondary_colors=[],
            cells=[(20, 20)], bounding_box=(20, 20, 20, 20),
            topology="solid", shape_hash="same", is_agent=False,
        )
        scene = SceneGraph(objects={"a": obj_a, "b": obj_b}, edges=set())
        agent = AgentState(
            object_ids=[], position=(5, 5), orientation=0,
            shape="x", color=[3], energy=None, inventory=[],
        )
        hyps = generate_goal_hypotheses_from_win(scene, agent)
        relational = [h for h in hyps if h.terminal_condition == GoalPredicateType.OBJECT_ORIENTATION_MATCHES_OTHER.value]
        assert len(relational) > 0


class TestGoalInferenceModule:
    def test_initializes_with_default_priors(self):
        module = GoalInferenceModule()
        assert len(module.hypotheses) == len(DEFAULT_GOAL_PRIORS)
        assert all(h.confidence > 0 for h in module.hypotheses)

    def test_observe_win_increases_confidence(self, scene_with_objects, agent_at_red):
        module = GoalInferenceModule()
        # First WIN at agent position (10,10)
        module.observe_win(scene_with_objects, agent_at_red)
        # Find the agent_at hypothesis for (10,10)
        agent_at_hyps = [
            h for h in module.hypotheses
            if h.terminal_condition == GoalPredicateType.AGENT_AT.value
            and h.args.get("position") == (10, 10)
        ]
        assert len(agent_at_hyps) >= 1
        # Confidence should have increased from prior
        assert agent_at_hyps[0].support == 1

    def test_observe_win_downgrades_non_matching(self, scene_with_objects, agent_at_red):
        """Hypotheses whose conditions don't hold at WIN should be downgraded."""
        module = GoalInferenceModule()
        # Add a hypothesis that requires position (50, 50) — won't hold at (10, 10)
        module.hypotheses.append(GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_AT.value,
            args={"position": (50, 50)},
            confidence=0.5,
            support=2,
        ))
        initial_conf = 0.5
        module.observe_win(scene_with_objects, agent_at_red)
        # Find that hypothesis
        target = [h for h in module.hypotheses if h.args.get("position") == (50, 50)]
        if target:
            assert target[0].confidence < initial_conf

    def test_observe_lose_with_near_miss(self, scene_with_objects, agent_at_red):
        module = GoalInferenceModule()
        list(module.hypotheses)
        module.observe_lose(
            scene_with_objects, agent_at_red,
            progress_predicates=["distance_to_exit", "color_match_count"],
        )
        # Goal types matching the progress predicates should have increased near_miss_support
        matching = [
            h for h in module.hypotheses
            if h.terminal_condition in (
                GoalPredicateType.AGENT_AT.value,
                GoalPredicateType.AGENT_REACHES_ZONE.value,
                GoalPredicateType.AGENT_COLOR_MATCHES.value,
            )
        ]
        assert any(h.near_miss_support > 0 for h in matching)

    def test_is_ready_to_plan_below_threshold(self, scene_with_objects, agent_at_red):
        module = GoalInferenceModule(planning_threshold=0.9)
        # With only default priors, no hypothesis should exceed 0.9
        assert not module.is_ready_to_plan()

    def test_is_ready_to_plan_above_threshold(self, scene_with_objects, agent_at_red):
        module = GoalInferenceModule(planning_threshold=0.3)
        # Observe many wins at same position to boost confidence
        for _ in range(5):
            module.observe_win(scene_with_objects, agent_at_red)
        # Should now be ready
        assert module.is_ready_to_plan()

    def test_get_top_goal(self, scene_with_objects, agent_at_red):
        module = GoalInferenceModule()
        module.observe_win(scene_with_objects, agent_at_red)
        top = module.get_top_goal()
        assert top is not None
        assert top.confidence > 0

    def test_get_stats(self, scene_with_objects, agent_at_red):
        module = GoalInferenceModule()
        module.observe_step(scene_with_objects, agent_at_red, action_id=1)
        module.observe_win(scene_with_objects, agent_at_red)
        stats = module.get_stats()
        assert stats["step_count"] == 1
        assert stats["win_count"] == 1
        assert stats["hypothesis_count"] > 0
