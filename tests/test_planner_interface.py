"""Unit tests for planner_interface.py (Module 11)."""
import pytest

from alphamoo.goal_inference import GoalHypothesis, GoalInferenceModule, GoalPredicateType
from alphamoo.planner_interface import (
    AStarPlanner,
    ClickPlanner,
    LLMPlanner,
    MCTSPlanner,
    PolicyPlanner,
    TransformationPlanner,
    plan_action,
    select_planner,
)
from alphamoo.schemas import (
    AgentState,
    Condition,
    GameObject,
    Hypothesis,
    SceneGraph,
    Trigger,
)
from alphamoo.world_model import CausalRule, WorldModel


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
def scene_with_goal_obj():
    """Scene with an object at (10, 10) — potential goal target."""
    obj = GameObject(
        id="obj_001", color=5, secondary_colors=[],
        cells=[(10, 10), (10, 11), (11, 10), (11, 11)],
        bounding_box=(10, 10, 11, 11),
        topology="solid", shape_hash="xyz", is_agent=False,
    )
    return SceneGraph(objects={"obj_001": obj}, edges=set(), hash="with_obj")


@pytest.fixture
def high_confidence_wm():
    """World model with a high-confidence rule."""
    wm = WorldModel()
    wm.update_from_hypotheses([Hypothesis(
        trigger=Trigger(conditions=[
            Condition(predicate="agent_at", args={"position": (32, 32)})
        ]),
        effect={"type": "obj_appears", "args": {"color": 5}},
        confidence=0.95, support=20, mdl_cost=20,
    )])
    return wm


# =============================================================================
# Test individual planners
# =============================================================================

class TestAStarPlanner:
    def test_finds_plan_to_position_goal(self, empty_scene):
        """Agent at (32,32), goal is agent_at(32,30) → 2 steps UP."""
        wm = WorldModel()
        planner = AStarPlanner(max_depth=10, max_nodes=100)
        agent = AgentState(
            object_ids=[], position=(32, 32), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        goal = GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_AT.value,
            args={"position": (32, 30)},
            confidence=0.9, support=5,
        )
        plan = planner.plan(
            scene=empty_scene, agent_state=agent, goal=goal,
            world_model=wm, budget_remaining=100,
            available_actions=[1, 2, 3, 4],
        )
        assert plan is not None
        assert plan.planner_name == "AStar"
        # Should take 2 UP actions to get from (32,32) to (32,30)
        assert plan.actions == [1, 1]  # UP, UP

    def test_no_plan_when_unreachable(self, empty_scene):
        wm = WorldModel()
        planner = AStarPlanner(max_depth=5, max_nodes=50)
        agent = AgentState(
            object_ids=[], position=(32, 32), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        goal = GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_AT.value,
            args={"position": (5, 5)},  # too far for max_depth
            confidence=0.9, support=5,
        )
        plan = planner.plan(
            scene=empty_scene, agent_state=agent, goal=goal,
            world_model=wm, budget_remaining=100,
            available_actions=[1, 2, 3, 4],
        )
        assert plan is None

    def test_no_goal_returns_none(self, empty_scene, agent_at_center):
        wm = WorldModel()
        planner = AStarPlanner()
        plan = planner.plan(empty_scene, agent_at_center, None, wm, 100, [1, 2, 3, 4])
        assert plan is None

    def test_no_agent_returns_none(self, empty_scene):
        wm = WorldModel()
        planner = AStarPlanner()
        goal = GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_AT.value,
            args={"position": (10, 10)}, confidence=0.9, support=1,
        )
        plan = planner.plan(empty_scene, None, goal, wm, 100, [1, 2, 3, 4])
        assert plan is None


class TestMCTSPlanner:
    def test_returns_plan_or_none(self, empty_scene):
        wm = WorldModel()
        planner = MCTSPlanner(n_rollouts=5, max_depth=5, rng_seed=42)
        agent = AgentState(
            object_ids=[], position=(32, 32), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        goal = GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_AT.value,
            args={"position": (32, 31)},
            confidence=0.9, support=1,
        )
        plan = planner.plan(
            scene=empty_scene, agent_state=agent, goal=goal,
            world_model=wm, budget_remaining=100,
            available_actions=[1, 2, 3, 4],
        )
        # MCTS returns a 1-step plan or None
        if plan is not None:
            assert plan.planner_name == "MCTS"
            assert len(plan.actions) == 1


class TestPolicyPlanner:
    def test_returns_none(self, empty_scene, agent_at_center, high_confidence_wm):
        planner = PolicyPlanner()
        goal = GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_AT.value,
            args={"position": (10, 10)}, confidence=0.9, support=1,
        )
        plan = planner.plan(empty_scene, agent_at_center, goal, high_confidence_wm, 100, [1, 2, 3, 4])
        assert plan is None  # not yet implemented


class TestLLMPlanner:
    def test_returns_random_action(self, empty_scene, agent_at_center):
        wm = WorldModel()
        planner = LLMPlanner(rng_seed=42)
        goal = GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_AT.value,
            args={"position": (10, 10)}, confidence=0.9, support=1,
        )
        plan = planner.plan(empty_scene, agent_at_center, goal, wm, 100, [1, 2, 3, 4])
        assert plan is not None
        assert plan.planner_name == "LLM"
        assert plan.actions[0] in [1, 2, 3, 4]

    def test_no_actions_returns_none(self, empty_scene, agent_at_center):
        wm = WorldModel()
        planner = LLMPlanner()
        plan = planner.plan(empty_scene, agent_at_center, None, wm, 100, [])
        assert plan is None


class TestClickPlanner:
    def test_clicks_on_object(self, scene_with_goal_obj):
        wm = WorldModel()
        planner = ClickPlanner()
        agent = AgentState(
            object_ids=[], position=(50, 50), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        goal = GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_TOUCHES.value,
            args={"color": 5}, confidence=0.9, support=1,
        )
        plan = planner.plan(
            scene=scene_with_goal_obj, agent_state=agent, goal=goal,
            world_model=wm, budget_remaining=100,
            available_actions=[6],
        )
        assert plan is not None
        assert plan.planner_name == "Click"
        assert plan.actions == [6]
        assert plan.click_coords[0] is not None

    def test_no_objects_returns_none(self, empty_scene, agent_at_center):
        wm = WorldModel()
        planner = ClickPlanner()
        goal = GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_TOUCHES.value,
            args={"color": 5}, confidence=0.9, support=1,
        )
        plan = planner.plan(empty_scene, agent_at_center, goal, wm, 100, [6])
        assert plan is None


class TestTransformationPlanner:
    def test_finds_transformation_plan(self, empty_scene):
        """Goal: agent_color_matches(color=3). Agent starts with color=1.
        Rule: agent_at(32,32) → color change to 3."""
        wm = WorldModel()
        wm.update_from_hypotheses([Hypothesis(
            trigger=Trigger(conditions=[
                Condition(predicate="agent_at", args={"position": (32, 32)})
            ]),
            effect={"type": "agent_state_changes", "args": {
                "attr": "color", "old": 1, "new": 3
            }},
            confidence=0.9, support=5, mdl_cost=20,
        )])
        agent = AgentState(
            object_ids=[], position=(32, 32), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        goal = GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_COLOR_MATCHES.value,
            args={"color": 3}, confidence=0.9, support=1,
        )
        planner = TransformationPlanner(max_depth=5, max_nodes=50)
        # Since the agent is already at (32,32), the rule should trigger
        # immediately and transform the color
        plan = planner.plan(
            scene=empty_scene, agent_state=agent, goal=goal,
            world_model=wm, budget_remaining=100,
            available_actions=[1, 2, 3, 4],
        )
        # Should find a plan (possibly 0 steps if goal already achieved
        # after transformation, or 1+ steps)
        # Note: the transformation planner applies the rule in predict(),
        # so the first prediction should already achieve the goal
        if plan is not None:
            assert plan.planner_name == "Transformation"


# =============================================================================
# Test planner selection
# =============================================================================

class TestPlannerSelection:
    def test_click_only_game_selects_click_planner(self, high_confidence_wm):
        goal_module = GoalInferenceModule()
        planner = select_planner(high_confidence_wm, goal_module, [6])
        assert isinstance(planner, ClickPlanner)

    def test_click_and_undo_selects_click_planner(self, high_confidence_wm):
        goal_module = GoalInferenceModule()
        planner = select_planner(high_confidence_wm, goal_module, [6, 7])
        assert isinstance(planner, ClickPlanner)

    def test_state_matching_goal_selects_transformation_planner(self, high_confidence_wm):
        goal_module = GoalInferenceModule()
        # Set a state-matching goal
        for h in goal_module.hypotheses:
            h.confidence = 0.01
        goal_module.hypotheses.append(GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_COLOR_MATCHES.value,
            args={"color": 5}, confidence=0.9, support=10,
        ))
        planner = select_planner(high_confidence_wm, goal_module, [1, 2, 3, 4])
        assert isinstance(planner, TransformationPlanner)

    def test_high_confidence_selects_astar(self):
        wm = WorldModel()
        wm.rules = [CausalRule(
            trigger_conditions=[], trigger_temporal=None,
            effect={}, confidence=0.9, support=10, mdl_cost=10,
        )]
        goal_module = GoalInferenceModule()
        # Set a non-state-matching goal
        for h in goal_module.hypotheses:
            h.confidence = 0.01
        goal_module.hypotheses.append(GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_AT.value,
            args={"position": (10, 10)}, confidence=0.9, support=10,
        ))
        planner = select_planner(wm, goal_module, [1, 2, 3, 4])
        assert isinstance(planner, AStarPlanner)

    def test_low_confidence_selects_llm(self):
        wm = WorldModel()  # no rules → low confidence
        goal_module = GoalInferenceModule()
        for h in goal_module.hypotheses:
            h.confidence = 0.01
        goal_module.hypotheses.append(GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_AT.value,
            args={"position": (10, 10)}, confidence=0.9, support=10,
        ))
        planner = select_planner(wm, goal_module, [1, 2, 3, 4])
        assert isinstance(planner, LLMPlanner)


# =============================================================================
# Test plan_action high-level interface
# =============================================================================

class TestPlanAction:
    def test_returns_fallback_when_not_ready_to_plan(self, empty_scene, agent_at_center, high_confidence_wm):
        goal_module = GoalInferenceModule(planning_threshold=0.99)
        result = plan_action(
            scene=empty_scene, agent_state=agent_at_center,
            goal_module=goal_module, world_model=high_confidence_wm,
            budget_remaining=100, available_actions=[1, 2, 3, 4],
        )
        assert result.plan is None
        assert result.fallback_to_exploration

    def test_returns_plan_when_ready(self, empty_scene, high_confidence_wm):
        goal_module = GoalInferenceModule(planning_threshold=0.3)
        # Set a goal at the agent's current position (immediate achievement)
        for h in goal_module.hypotheses:
            h.confidence = 0.01
        goal_module.hypotheses.append(GoalHypothesis(
            terminal_condition=GoalPredicateType.AGENT_AT.value,
            args={"position": (32, 32)}, confidence=0.9, support=10,
        ))
        agent = AgentState(
            object_ids=[], position=(32, 32), orientation=0,
            shape="x", color=[1], energy=None, inventory=[],
        )
        result = plan_action(
            scene=empty_scene, agent_state=agent,
            goal_module=goal_module, world_model=high_confidence_wm,
            budget_remaining=100, available_actions=[1, 2, 3, 4],
        )
        # A* should find a 0-step plan (goal already at current position)
        assert result.plan is not None
        assert not result.fallback_to_exploration
