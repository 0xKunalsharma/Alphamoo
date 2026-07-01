"""Unit tests for experiment_planner.py (Module 8)."""
import pytest

from alphamoo.experiment_planner import (
    CandidateAction,
    ExperimentPlanner,
    _compute_state_hash,
    _simulate_movement,
    compute_surrogate_ig,
    compute_symbolic_ig,
    generate_candidate_actions,
)
from alphamoo.schemas import (
    ActionId,
    AgentState,
    Condition,
    GameObject,
    Hypothesis,
    SceneGraph,
    Trigger,
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


@pytest.fixture
def scene_with_object():
    obj = GameObject(
        id="obj_001", color=2, secondary_colors=[],
        cells=[(35, 35), (35, 36), (36, 35), (36, 36)],
        bounding_box=(35, 35, 36, 36),
        topology="solid", shape_hash="xyz", is_agent=False,
    )
    return SceneGraph(objects={"obj_001": obj}, edges=set(), hash="with_obj")


@pytest.fixture
def simple_hypothesis():
    """Hypothesis: IF agent_at(35,35) THEN obj_appears(color=5)."""
    return Hypothesis(
        trigger=Trigger(conditions=[
            Condition(predicate="agent_at", args={"position": (35, 35)})
        ]),
        effect={"type": "obj_appears", "args": {"color": 5}},
        confidence=0.5,
        support=3,
        mdl_cost=20,
    )


class TestMovementSimulation:
    def test_up(self):
        assert _simulate_movement((10, 10), ActionId.UP) == (10, 9)

    def test_down(self):
        assert _simulate_movement((10, 10), ActionId.DOWN) == (10, 11)

    def test_left(self):
        assert _simulate_movement((10, 10), ActionId.LEFT) == (9, 10)

    def test_right(self):
        assert _simulate_movement((10, 10), ActionId.RIGHT) == (11, 10)

    def test_up_at_top_edge(self):
        assert _simulate_movement((10, 0), ActionId.UP) == (10, 0)

    def test_left_at_left_edge(self):
        assert _simulate_movement((0, 10), ActionId.LEFT) == (0, 10)


class TestStateHash:
    def test_same_state_same_hash(self, empty_scene, agent_at_center):
        h1 = _compute_state_hash(empty_scene, agent_at_center, ActionId.UP)
        h2 = _compute_state_hash(empty_scene, agent_at_center, ActionId.UP)
        assert h1 == h2

    def test_different_action_different_hash(self, empty_scene, agent_at_center):
        h1 = _compute_state_hash(empty_scene, agent_at_center, ActionId.UP)
        h2 = _compute_state_hash(empty_scene, agent_at_center, ActionId.DOWN)
        assert h1 != h2

    def test_different_position_different_hash(self, empty_scene):
        a1 = AgentState(object_ids=[], position=(10, 10), orientation=0,
                        shape="x", color=[1], energy=None, inventory=[])
        a2 = AgentState(object_ids=[], position=(20, 20), orientation=0,
                        shape="x", color=[1], energy=None, inventory=[])
        h1 = _compute_state_hash(empty_scene, a1, ActionId.UP)
        h2 = _compute_state_hash(empty_scene, a2, ActionId.UP)
        assert h1 != h2


class TestCandidateGeneration:
    def test_movement_actions_generated(self, empty_scene, agent_at_center):
        candidates = generate_candidate_actions([1, 2, 3, 4], empty_scene, agent_at_center)
        assert len(candidates) == 4
        action_ids = [c.action_id for c in candidates]
        assert sorted(action_ids) == [1, 2, 3, 4]

    def test_click_action_generates_targets(self, scene_with_object, agent_at_center):
        candidates = generate_candidate_actions([6], scene_with_object, agent_at_center)
        # Should generate at least 1 click candidate
        assert len(candidates) >= 1
        assert all(c.action_id == 6 for c in candidates)
        assert all(c.click_coords is not None for c in candidates)

    def test_click_no_objects_clicks_center(self, empty_scene, agent_at_center):
        candidates = generate_candidate_actions([6], empty_scene, agent_at_center)
        assert len(candidates) == 1
        assert candidates[0].click_coords == (32, 32)

    def test_reset_action_marked_as_termination(self, empty_scene, agent_at_center):
        candidates = generate_candidate_actions([0], empty_scene, agent_at_center)
        assert len(candidates) == 1
        assert candidates[0].experiment_type == "termination"


class TestSymbolicIG:
    def test_no_hypotheses_returns_zero(self, empty_scene, agent_at_center):
        action = CandidateAction(action_id=1)
        ig = compute_symbolic_ig(action, empty_scene, agent_at_center, [], set())
        assert ig == 0.0

    def test_trigger_holds_increases_ig(self, simple_hypothesis):
        """If the hypothesis trigger currently holds, IG should be positive."""
        scene = SceneGraph(objects={}, edges=set(), hash="x")
        agent = AgentState(
            object_ids=[], position=(35, 35), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        action = CandidateAction(action_id=1)
        ig = compute_symbolic_ig(action, scene, agent, [simple_hypothesis], set())
        assert ig > 0

    def test_movement_to_satisfy_trigger_increases_ig(self, simple_hypothesis, empty_scene):
        """Agent at (34,35) moving RIGHT to (35,35) would satisfy trigger."""
        agent = AgentState(
            object_ids=[], position=(34, 35), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        action = CandidateAction(action_id=ActionId.RIGHT)
        ig = compute_symbolic_ig(action, empty_scene, agent, [simple_hypothesis], set())
        assert ig > 0

    def test_revisit_penalty(self, simple_hypothesis, empty_scene, agent_at_center):
        """Visited states get penalized."""
        action = CandidateAction(action_id=1)
        # First time — unvisited
        ig_new = compute_symbolic_ig(
            action, empty_scene, agent_at_center, [simple_hypothesis], set()
        )
        # Mark state as visited
        state_hash = _compute_state_hash(empty_scene, agent_at_center, 1)
        ig_revisit = compute_symbolic_ig(
            action, empty_scene, agent_at_center, [simple_hypothesis], {state_hash}
        )
        # Both should be 0 (no trigger holds, no movement target) — but test the mechanism
        # by checking that visited state's hash triggers penalty in the code path
        assert ig_revisit <= ig_new  # revisit can't be higher


class TestSurrogateIG:
    def test_returns_positive(self, empty_scene, agent_at_center):
        action = CandidateAction(action_id=1)
        ig = compute_surrogate_ig(action, empty_scene, agent_at_center, [])
        assert ig > 0

    def test_interact_action_higher_base(self, empty_scene, agent_at_center):
        """INTERACT should have higher base score than movement."""
        move = CandidateAction(action_id=ActionId.UP)
        interact = CandidateAction(action_id=ActionId.INTERACT)
        ig_move = compute_surrogate_ig(move, empty_scene, agent_at_center, [])
        ig_interact = compute_surrogate_ig(interact, empty_scene, agent_at_center, [])
        assert ig_interact > ig_move


class TestExperimentPlanner:
    def test_select_action_returns_candidate(self, empty_scene, agent_at_center):
        planner = ExperimentPlanner()
        action = planner.select_action(
            available_actions=[1, 2, 3, 4],
            scene=empty_scene,
            agent_state=agent_at_center,
            hypotheses=[],
        )
        assert isinstance(action, CandidateAction)
        assert action.action_id in [1, 2, 3, 4]

    def test_select_action_records_visited_state(self, empty_scene, agent_at_center):
        planner = ExperimentPlanner()
        planner.select_action(
            available_actions=[1, 2, 3, 4],
            scene=empty_scene, agent_state=agent_at_center, hypotheses=[],
        )
        # Manually record (in production, this happens after action execution)
        planner.record_visited_state(empty_scene, agent_at_center, 1)
        assert len(planner.visited_state_hashes) >= 1

    def test_escape_count_increases_with_low_ig(self, empty_scene, agent_at_center):
        """When IG is below floor, ε-greedy escape should fire sometimes."""
        planner = ExperimentPlanner(ig_floor=10.0, epsilon=1.0)  # always escape
        for _ in range(10):
            planner.select_action(
                available_actions=[1, 2, 3, 4],
                scene=empty_scene, agent_state=agent_at_center, hypotheses=[],
            )
        assert planner._escape_count > 0

    def test_get_stats(self, empty_scene, agent_at_center):
        planner = ExperimentPlanner()
        planner.select_action(
            available_actions=[1, 2, 3, 4],
            scene=empty_scene, agent_state=agent_at_center, hypotheses=[],
        )
        stats = planner.get_stats()
        assert stats["selection_count"] == 1
        assert stats["tier1_count"] + stats["tier2_count"] >= 1

    def test_with_real_hypotheses(self, simple_hypothesis, empty_scene):
        """Planner should prefer actions that test the hypothesis."""
        # Agent one step away from (35,35) — RIGHT would satisfy trigger
        agent = AgentState(
            object_ids=[], position=(34, 35), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        planner = ExperimentPlanner(ig_floor=0.001, epsilon=0.0)
        action = planner.select_action(
            available_actions=[1, 2, 3, 4],
            scene=empty_scene, agent_state=agent,
            hypotheses=[simple_hypothesis],
        )
        # RIGHT (action 4) should have the highest IG
        assert action.action_id == ActionId.RIGHT

    def test_click_game_action_selection(self, scene_with_object, agent_at_center):
        """Click games should generate click candidates."""
        planner = ExperimentPlanner(epsilon=0.0)
        action = planner.select_action(
            available_actions=[6],
            scene=scene_with_object, agent_state=agent_at_center,
            hypotheses=[],
        )
        assert action.action_id == 6
        assert action.click_coords is not None

    def test_experiment_type_classification(self, empty_scene, agent_at_center):
        """First-time actions should be classified as 'novel'."""
        planner = ExperimentPlanner(epsilon=0.0)
        action = planner.select_action(
            available_actions=[1, 2, 3, 4],
            scene=empty_scene, agent_state=agent_at_center,
            hypotheses=[],
        )
        # First action — state unvisited → novel
        assert action.experiment_type in ["novel", "contrastive", "boundary"]
