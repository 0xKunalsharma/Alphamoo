"""
AlphaMoo v4.1 — Planner Interface (Module 11).

A protocol-based planner with 6 variants, selected by world model
confidence profile and goal type.

Planner variants:
  1. AStarPlanner      — when WM rule confidence > 0.8, no rule uncertainty
  2. MCTSPlanner       — when rule uncertainty exists, need to sample
  3. PolicyPlanner     — when prior levels solved similar goals (future)
  4. LLMPlanner        — when novel goal, world model has gaps (uses LLM)
  5. ClickPlanner      — for click-only games (action_space = [6])
  6. TransformationPlanner — for state-matching goals (LS20-style)

Selection rule (from v4.1 spec):
  if action_space == [6]:
      return ClickPlanner()
  if wm_conf > 0.8 and rule_uncertainty < 0.2:
      return AStarPlanner()
  if goal_type == "state_matching":
      return TransformationPlanner()
  if rule_uncertainty >= 0.2:
      return MCTSPlanner()
  if has_prior:
      return PolicyPlanner()
  return LLMPlanner()

Budget awareness: if shortest plan > 60% of remaining budget, drop back
to exploration (return None to signal "explore more").
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from .goal_inference import (
    GoalInferenceModule,
    GoalPredicateType,
    evaluate_goal_predicate,
)
from .schemas import (
    ActionId,
    AgentState,
    GoalHypothesis,
    SceneGraph,
)
from .world_model import WorldModel

# =============================================================================
# Plan representation
# =============================================================================

@dataclass
class Plan:
    """A sequence of actions to execute."""
    actions: list[int] = field(default_factory=list)  # action IDs
    click_coords: list[tuple[int, int] | None] = field(default_factory=list)
    expected_goal_achievement: float = 0.0  # 0-1, how confident this plan achieves the goal
    planner_name: str = ""
    notes: str = ""
    n_steps: int = 0  # number of actions in the plan

    def __post_init__(self):
        self.n_steps = len(self.actions)


# =============================================================================
# Planner Protocol
# =============================================================================

class Planner(Protocol):
    """Interface for all planner variants."""

    def plan(
        self,
        scene: SceneGraph,
        agent_state: AgentState | None,
        goal: GoalHypothesis | None,
        world_model: WorldModel,
        budget_remaining: int,
        available_actions: list[int],
    ) -> Plan | None:
        """
        Generate a plan to achieve the goal.

        Returns:
            Plan if a valid plan is found, None if no plan (fall back to exploration).
        """
        ...


# =============================================================================
# Planner 1: A* Planner
# =============================================================================

class AStarPlanner:
    """
    A* search over the world model. Use when WM confidence is high and
    rule uncertainty is low.

    Path cost: number of actions.
    Heuristic: Manhattan distance to goal (or 0 for non-spatial goals).
    """

    def __init__(self, max_depth: int = 20, max_nodes: int = 500):
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def plan(
        self,
        scene: SceneGraph,
        agent_state: AgentState | None,
        goal: GoalHypothesis | None,
        world_model: WorldModel,
        budget_remaining: int,
        available_actions: list[int],
    ) -> Plan | None:
        if agent_state is None or goal is None:
            return None

        # A* search
        # State = (agent_position, frozenset of object IDs)
        # We simplify: state = agent_position only
        import heapq

        start_state = agent_state.position
        goal_predicate = goal.terminal_condition
        goal_args = goal.args

        # Priority queue: (f_score, g_score, position, plan_actions, plan_coords)
        counter = 0
        pq: list[tuple[float, int, tuple[int, int], list[int], list, dict]] = []
        heapq.heappush(pq, (0, 0, start_state, [], [], {"scene": scene, "agent": agent_state}))

        visited: set[tuple[int, int]] = {start_state}

        while pq and counter < self.max_nodes:
            f, g, pos, actions, coords, state_dict = heapq.heappop(pq)
            counter += 1

            current_scene = state_dict["scene"]
            current_agent = state_dict["agent"]

            # Check if goal is achieved
            if evaluate_goal_predicate(goal_predicate, goal_args, current_scene, current_agent):
                return Plan(
                    actions=actions,
                    click_coords=coords,
                    expected_goal_achievement=goal.confidence,
                    planner_name="AStar",
                    n_steps=len(actions),
                )

            # Check depth limit
            if len(actions) >= self.max_depth:
                continue

            # Check budget (60% rule)
            if len(actions) > 0.6 * budget_remaining:
                continue

            # Expand neighbors (try each available action)
            for action_id in available_actions:
                if action_id in (ActionId.RESET, ActionId.UNDO):
                    continue  # don't include RESET/UNDO in planning

                # Predict next state
                prediction = world_model.predict(current_scene, current_agent, action_id)
                new_pos = prediction.predicted_agent_state.position if prediction.predicted_agent_state else pos

                if new_pos in visited:
                    continue
                visited.add(new_pos)

                new_actions = actions + [action_id]
                new_coords = coords + [None]
                g_new = g + 1
                h = self._heuristic(new_pos, goal, goal_args, prediction.predicted_scene, prediction.predicted_agent_state)
                f_new = g_new + h

                heapq.heappush(pq, (
                    f_new, g_new, new_pos, new_actions, new_coords,
                    {"scene": prediction.predicted_scene, "agent": prediction.predicted_agent_state}
                ))

        return None  # no plan found

    def _heuristic(
        self,
        position: tuple[int, int],
        goal: GoalHypothesis,
        goal_args: dict,
        scene: SceneGraph,
        agent_state: AgentState | None,
    ) -> float:
        """Estimate distance to goal."""
        # For position-based goals, use Manhattan distance
        if goal.terminal_condition == GoalPredicateType.AGENT_AT.value:
            target_pos = goal_args.get("position")
            if target_pos:
                return abs(position[0] - target_pos[0]) + abs(position[1] - target_pos[1])
        # For other goals, use 0 (no heuristic — becomes BFS)
        return 0.0


# =============================================================================
# Planner 2: MCTS Planner
# =============================================================================

class MCTSPlanner:
    """
    Monte Carlo Tree Search over the world model. Use when rule
    uncertainty exists (rule_entropy >= 0.2).

    Samples N world models from rule posteriors, runs A* on each,
    picks the action sequence robust across samples.
    """

    def __init__(self, n_rollouts: int = 50, max_depth: int = 15, rng_seed: int | None = None):
        self.n_rollouts = n_rollouts
        self.max_depth = max_depth
        self.rng = random.Random(rng_seed)
        self._a_star = AStarPlanner(max_depth=max_depth)

    def plan(
        self,
        scene: SceneGraph,
        agent_state: AgentState | None,
        goal: GoalHypothesis | None,
        world_model: WorldModel,
        budget_remaining: int,
        available_actions: list[int],
    ) -> Plan | None:
        if agent_state is None or goal is None:
            return None

        # For each rollout, sample a perturbed world model (drop some rules)
        # and run A* on it. Collect the first action from each successful plan.
        first_actions: list[int] = []
        best_plan: Plan | None = None

        for i in range(self.n_rollouts):
            # Create a perturbed world model by sampling rules
            perturbed_wm = self._sample_world_model(world_model)

            plan = self._a_star.plan(
                scene, agent_state, goal, perturbed_wm,
                budget_remaining, available_actions,
            )
            if plan is not None and plan.actions:
                first_actions.append(plan.actions[0])
                if best_plan is None or len(plan.actions) < len(best_plan.actions):
                    best_plan = plan

            # Budget check on rollouts
            if i >= self.n_rollouts:
                break

        if not first_actions:
            return None

        # Pick the most common first action (robust choice)
        from collections import Counter
        action_counts = Counter(first_actions)
        robust_first_action = action_counts.most_common(1)[0][0]

        # Return a 1-step plan with the robust first action
        # (MCTS is typically used step-by-step, replanning each step)
        return Plan(
            actions=[robust_first_action],
            click_coords=[None],
            expected_goal_achievement=goal.confidence * 0.5,  # lower confidence due to uncertainty
            planner_name="MCTS",
            notes=f"robust first action from {len(first_actions)} rollouts",
            n_steps=1,
        )

    def _sample_world_model(self, world_model: WorldModel) -> WorldModel:
        """Create a perturbed copy of the world model by sampling rules."""
        import copy
        sampled = WorldModel(confidence_threshold=0.3)  # lower threshold for sampling
        for rule in world_model.rules:
            # Include each rule with probability proportional to its confidence
            if self.rng.random() < rule.confidence:
                sampled.rules.append(copy.deepcopy(rule))
        return sampled


# =============================================================================
# Planner 3: Policy Planner (stub)
# =============================================================================

class PolicyPlanner:
    """
    Use when prior levels solved similar goals. Retrieves a learned policy
    from semantic memory.

    STUB: returns None (not yet implemented — requires Module 12 Context Compressor).
    """

    def plan(
        self,
        scene: SceneGraph,
        agent_state: AgentState | None,
        goal: GoalHypothesis | None,
        world_model: WorldModel,
        budget_remaining: int,
        available_actions: list[int],
    ) -> Plan | None:
        return None  # not yet implemented


# =============================================================================
# Planner 4: LLM Planner (stub — would use the reasoning engine)
# =============================================================================

class LLMPlanner:
    """
    Use when novel goal, world model has gaps. Asks the reasoning engine
    to generate a plan.

    STUB: returns a random valid action (placeholder until reasoning engine
    integration in Phase 4/5).
    """

    def __init__(self, rng_seed: int | None = None):
        self.rng = random.Random(rng_seed)

    def plan(
        self,
        scene: SceneGraph,
        agent_state: AgentState | None,
        goal: GoalHypothesis | None,
        world_model: WorldModel,
        budget_remaining: int,
        available_actions: list[int],
    ) -> Plan | None:
        if not available_actions:
            return None
        # Pick a random valid action (placeholder)
        action = self.rng.choice(available_actions)
        return Plan(
            actions=[action],
            click_coords=[None],
            expected_goal_achievement=0.1,
            planner_name="LLM",
            notes="placeholder (reasoning engine not yet integrated)",
            n_steps=1,
        )


# =============================================================================
# Planner 5: Click Planner (for click-only games)
# =============================================================================

class ClickPlanner:
    """
    For click-only games (available_actions = [6] or [6,7]).
    Picks click targets by trying each object in the scene, using the
    world model to predict which click achieves the goal.
    """

    def __init__(self, max_clicks: int = 10):
        self.max_clicks = max_clicks

    def plan(
        self,
        scene: SceneGraph,
        agent_state: AgentState | None,
        goal: GoalHypothesis | None,
        world_model: WorldModel,
        budget_remaining: int,
        available_actions: list[int],
    ) -> Plan | None:
        if goal is None or not scene.objects:
            return None

        # Try clicking each object, predict outcome, check if goal achieved
        best_plan: Plan | None = None
        best_score = -1.0

        for obj in scene.objects.values():
            # Click on object centroid
            xs = [x for x, _ in obj.cells]
            ys = [y for _, y in obj.cells]
            cx = sum(xs) // len(xs)
            cy = sum(ys) // len(ys)

            # Predict outcome of clicking here
            prediction = world_model.predict(scene, agent_state, ActionId.CLICK)

            # Check if goal would be achieved
            goal_achieved = evaluate_goal_predicate(
                goal.terminal_condition, goal.args,
                prediction.predicted_scene, prediction.predicted_agent_state,
            )

            score = goal.confidence if goal_achieved else 0.0
            if score > best_score:
                best_score = score
                best_plan = Plan(
                    actions=[ActionId.CLICK],
                    click_coords=[(cx, cy)],
                    expected_goal_achievement=score,
                    planner_name="Click",
                    notes=f"click on obj {obj.id} (color={obj.color})",
                    n_steps=1,
                )

        return best_plan


# =============================================================================
# Planner 6: Transformation Planner (for state-matching goals)
# =============================================================================

class TransformationPlanner:
    """
    For state-matching goals (LS20-style: agent must match target color,
    shape, orientation). Searches over transformation sequences rather
    than spatial paths.

    Strategy: find a sequence of actions that transforms the agent's
    state to match the target, using the world model to predict
    transformations.
    """

    def __init__(self, max_depth: int = 15, max_nodes: int = 300):
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def plan(
        self,
        scene: SceneGraph,
        agent_state: AgentState | None,
        goal: GoalHypothesis | None,
        world_model: WorldModel,
        budget_remaining: int,
        available_actions: list[int],
    ) -> Plan | None:
        if agent_state is None or goal is None:
            return None

        # BFS over transformation sequences
        import heapq

        counter = 0
        # State = (agent_color, agent_orientation, agent_shape, position)
        start_state = self._encode_state(agent_state)

        pq: list[tuple[int, list[int], list, SceneGraph, AgentState]] = [
            (0, [], [], scene, agent_state)
        ]
        visited: set = {start_state}

        while pq and counter < self.max_nodes:
            cost, actions, coords, curr_scene, curr_agent = heapq.heappop(pq)
            counter += 1

            # Check if goal achieved
            if evaluate_goal_predicate(
                goal.terminal_condition, goal.args, curr_scene, curr_agent
            ):
                return Plan(
                    actions=actions,
                    click_coords=coords,
                    expected_goal_achievement=goal.confidence,
                    planner_name="Transformation",
                    n_steps=len(actions),
                )

            if len(actions) >= self.max_depth:
                continue
            if len(actions) > 0.6 * budget_remaining:
                continue

            # Try each action
            for action_id in available_actions:
                if action_id == ActionId.RESET:
                    continue

                prediction = world_model.predict(curr_scene, curr_agent, action_id)
                new_agent = prediction.predicted_agent_state
                if new_agent is None:
                    continue

                new_state = self._encode_state(new_agent)
                if new_state in visited:
                    continue
                visited.add(new_state)

                new_actions = actions + [action_id]
                new_coords = coords + [None]
                heapq.heappush(pq, (
                    len(new_actions), new_actions, new_coords,
                    prediction.predicted_scene, new_agent,
                ))

        return None

    def _encode_state(self, agent: AgentState) -> tuple:
        """Encode agent state for visited-set dedup."""
        return (
            tuple(sorted(agent.color)),
            agent.orientation,
            agent.shape,
            agent.position,
        )


# =============================================================================
# Planner Selection
# =============================================================================

def select_planner(
    world_model: WorldModel,
    goal_module: GoalInferenceModule,
    available_actions: list[int],
    has_prior: bool = False,
) -> Planner:
    """
    Select the appropriate planner based on world model confidence
    profile and goal type.

    Args:
        world_model: the agent's world model
        goal_module: the goal inference module (for goal type)
        available_actions: list of valid action IDs for this game
        has_prior: True if semantic memory has similar prior levels

    Returns:
        A Planner instance.
    """
    # Click-only games → ClickPlanner
    if available_actions == [ActionId.CLICK] or (
        ActionId.CLICK in available_actions
        and not any(a in (ActionId.UP, ActionId.DOWN, ActionId.LEFT, ActionId.RIGHT)
                    for a in available_actions)
    ):
        return ClickPlanner()

    wm_conf = world_model.avg_rule_confidence()
    rule_uncertainty = world_model.rule_entropy()
    top_goal = goal_module.get_top_goal()

    # State-matching goals → TransformationPlanner
    if top_goal and top_goal.terminal_condition in (
        GoalPredicateType.AGENT_STATE_MATCHES.value,
        GoalPredicateType.AGENT_COLOR_MATCHES.value,
        GoalPredicateType.AGENT_SHAPE_MATCHES.value,
        GoalPredicateType.AGENT_ORIENTATION_MATCHES.value,
    ):
        return TransformationPlanner()

    # High confidence, low uncertainty → A*
    if wm_conf > 0.8 and rule_uncertainty < 0.2:
        return AStarPlanner()

    # Rule uncertainty → MCTS
    if rule_uncertainty >= 0.2:
        return MCTSPlanner()

    # Has prior → Policy
    if has_prior:
        return PolicyPlanner()

    # Fallback → LLM
    return LLMPlanner()


# =============================================================================
# Planning result with metadata
# =============================================================================

@dataclass
class PlanningResult:
    """Result of a planning attempt."""
    plan: Plan | None
    planner_name: str
    selected_by: str  # description of selection rule
    fallback_to_exploration: bool = False  # True if no plan found
    notes: str = ""


def plan_action(
    scene: SceneGraph,
    agent_state: AgentState | None,
    goal_module: GoalInferenceModule,
    world_model: WorldModel,
    budget_remaining: int,
    available_actions: list[int],
    has_prior: bool = False,
) -> PlanningResult:
    """
    High-level planning entry point. Selects a planner and generates a plan.

    Returns:
        PlanningResult with the plan (or None if no plan found).
    """
    # Select planner
    planner = select_planner(world_model, goal_module, available_actions, has_prior)
    planner_name = planner.__class__.__name__

    # Get top goal
    top_goal = goal_module.get_top_goal()

    # Selection reason
    if isinstance(planner, ClickPlanner):
        selected_by = "click-only game"
    elif isinstance(planner, TransformationPlanner):
        selected_by = f"state-matching goal ({top_goal.terminal_condition if top_goal else 'None'})"
    elif isinstance(planner, AStarPlanner):
        selected_by = f"high WM confidence ({world_model.avg_rule_confidence():.2f})"
    elif isinstance(planner, MCTSPlanner):
        selected_by = f"rule uncertainty (entropy={world_model.rule_entropy():.2f})"
    elif isinstance(planner, PolicyPlanner):
        selected_by = "has prior level"
    else:
        selected_by = "fallback (novel goal)"

    # Check if ready to plan
    if not goal_module.is_ready_to_plan():
        return PlanningResult(
            plan=None,
            planner_name=planner_name,
            selected_by=selected_by,
            fallback_to_exploration=True,
            notes="goal confidence below planning threshold",
        )

    # Generate plan
    if top_goal is None:
        return PlanningResult(
            plan=None,
            planner_name=planner_name,
            selected_by=selected_by,
            fallback_to_exploration=True,
            notes="no goal hypothesis",
        )

    plan = planner.plan(
        scene=scene,
        agent_state=agent_state,
        goal=top_goal,
        world_model=world_model,
        budget_remaining=budget_remaining,
        available_actions=available_actions,
    )

    if plan is None:
        return PlanningResult(
            plan=None,
            planner_name=planner_name,
            selected_by=selected_by,
            fallback_to_exploration=True,
            notes="planner returned no plan",
        )

    return PlanningResult(
        plan=plan,
        planner_name=planner_name,
        selected_by=selected_by,
        fallback_to_exploration=False,
    )
