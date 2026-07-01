"""
Phase 3 integration test — full explore→exploit loop with planning.

Wires together the complete agent:
  Perception → Tracker → Cascade → Hypothesis → Goal → NearMiss → Planner
  → WorldModel → Verifier → (back to Hypothesis on mismatch)

Shows the agent transitioning from exploration to exploitation as it
learns the game mechanics and infers the goal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from alphamoo.agent_tracker import AgentStateTracker
from alphamoo.cascade_interpreter import classify_event, diff_grids, interpret_cascade
from alphamoo.experiment_planner import ExperimentPlanner
from alphamoo.goal_inference import GoalInferenceModule
from alphamoo.hypothesis_generator import HypothesisGenerator
from alphamoo.near_miss_tracker import NearMissTracker
from alphamoo.perception import detect_background_color, perceive
from alphamoo.planner_interface import plan_action
from alphamoo.schemas import GameState
from alphamoo.verifier import Verifier
from alphamoo.vtx_reader import load_replays_from_dir
from alphamoo.world_model import WorldModel


def format_plan(plan) -> str:
    """Format a Plan for display."""
    if plan is None:
        return "None"
    actions_str = []
    for _i, (a, c) in enumerate(zip(plan.actions, plan.click_coords)):
        if c:
            actions_str.append(f"{a}@{c}")
        else:
            actions_str.append(str(a))
    return f"[{', '.join(actions_str)}] ({plan.planner_name}, {plan.n_steps} steps)"


def run_phase3_loop(replay, max_steps=100, verbose=False):
    """Run the full Phase 3 explore→exploit loop on one replay."""
    tracker = AgentStateTracker()
    hyp_gen = HypothesisGenerator()
    goal_module = GoalInferenceModule()
    near_miss = NearMissTracker()
    exploration_planner = ExperimentPlanner(ig_floor=0.05, epsilon=0.1, rng_seed=42)
    world_model = WorldModel()
    verifier = Verifier(world_model)

    n = min(max_steps, len(replay.records))
    prev_grid = None
    prev_record = None

    exploration_steps = 0
    planning_steps = 0
    planning_successes = 0
    planner_types_used = {}

    for i in range(n):
        record = replay.records[i]
        grid = np.array(record.final_grid, dtype=np.int8)
        bg = detect_background_color(grid)
        scene = perceive(grid.tolist(), background_color=bg)
        agent_state, _ = tracker.update(
            grid, record.action_input.id,
            background_color=bg,
            available_actions=record.available_actions,
            action_input=record.action_input,
        )

        # Detect events
        events = []
        if record.n_subframes > 1:
            with __import__("contextlib").suppress(Exception):
                _, events = interpret_cascade(record.frame)
        elif prev_grid is not None:
            diff = diff_grids(prev_grid, grid)
            events = classify_event(diff, None, None, timestep=0)

        # 1. Update world model from confirmed hypotheses
        confirmed = [h for h in hyp_gen.hypotheses if h.confidence > 0.7]
        world_model.update_from_hypotheses(confirmed)

        # 2. If we had a prediction from last step, verify it
        if hasattr(run_phase3_loop, '_last_prediction') and run_phase3_loop._last_prediction is not None:
            verifier.verify(
                run_phase3_loop._last_prediction,
                scene, events, agent_state,
            )
            run_phase3_loop._last_prediction = None

        # 3. Update hypothesis generator
        hyp_gen.observe(scene, agent_state, events, record.action_input.id)

        # 4. Update goal inference
        goal_module.observe_step(scene, agent_state, record.action_input.id)
        if prev_record and record.levels_completed > prev_record.levels_completed:
            goal_module.observe_win(scene, agent_state)
            near_miss.reset()
        elif record.state == GameState.WIN:
            goal_module.observe_win(scene, agent_state)
        elif record.state == GameState.GAME_OVER:
            near_miss_predicates = near_miss.on_episode_end("GAME_OVER")
            if near_miss_predicates:
                goal_module.observe_lose(scene, agent_state, near_miss_predicates)
            near_miss.reset()
        else:
            near_miss.record_step(scene, agent_state)

        # 5. Decide: explore or exploit?
        if goal_module.is_ready_to_plan() and world_model.rules:
            # Try to plan
            planning_result = plan_action(
                scene=scene, agent_state=agent_state,
                goal_module=goal_module, world_model=world_model,
                budget_remaining=n - i,
                available_actions=record.available_actions,
            )
            planning_steps += 1
            planner_name = planning_result.planner_name
            planner_types_used[planner_name] = planner_types_used.get(planner_name, 0) + 1

            if planning_result.plan is not None:
                planning_successes += 1
                # Predict the outcome of the first action
                first_action = planning_result.plan.actions[0]
                run_phase3_loop._last_prediction = world_model.predict(
                    scene, agent_state, first_action
                )
            else:
                # Plan failed — fall back to exploration
                exploration_steps += 1
        else:
            # Not ready to plan — explore
            exploration_steps += 1
            top_hyps = hyp_gen.get_top_hypotheses(k=10)
            chosen = exploration_planner.select_action(
                available_actions=record.available_actions,
                scene=scene, agent_state=agent_state,
                hypotheses=top_hyps,
            )
            # Predict outcome for verification
            run_phase3_loop._last_prediction = world_model.predict(
                scene, agent_state, chosen.action_id
            )

        prev_grid = grid
        prev_record = record

        if verbose and i % 25 == 0:
            mode = "PLANNING" if (goal_module.is_ready_to_plan() and world_model.rules) else "EXPLORING"
            print(f"\n  Step {i} [{mode}]:")
            print(f"    WM rules: {len(world_model.rules)}, avg_conf={world_model.avg_rule_confidence():.2f}")
            print(f"    Hypotheses: {len(hyp_gen.hypotheses)} ({len(confirmed)} confirmed)")
            print(f"    Goal ready: {goal_module.is_ready_to_plan()}")
            top_goal = goal_module.get_top_goal()
            if top_goal:
                print(f"    Top goal: {top_goal.terminal_condition} conf={top_goal.confidence:.2f}")
            print(f"    Verifier: {verifier.get_stats()['match_rate']*100:.0f}% match rate")

    return {
        "steps": n,
        "exploration_steps": exploration_steps,
        "planning_steps": planning_steps,
        "planning_successes": planning_successes,
        "planner_types_used": planner_types_used,
        "hypothesis_stats": hyp_gen.get_stats(),
        "goal_stats": goal_module.get_stats(),
        "world_model_stats": world_model.get_stats(),
        "verifier_stats": verifier.get_stats(),
    }


def main():
    data_dir = Path("/home/z/my-project/alphamoo/data")
    print("Loading all replays...")
    replays = load_replays_from_dir(data_dir)

    test_games = ["ls20", "ft09", "r11l"]

    for game_id in test_games:
        if game_id not in replays:
            continue
        replay = replays[game_id]
        print(f"\n{'='*80}")
        print(f"Game: {game_id} ({replay.n_actions} actions, available: {replay.available_actions})")
        print(f"{'='*80}")

        max_steps = min(100, len(replay.records))
        print(f"Running full Phase 3 loop on first {max_steps} steps...")

        results = run_phase3_loop(replay, max_steps=max_steps, verbose=True)

        print("\n--- Results ---")
        print(f"Total steps: {results['steps']}")
        print(f"Exploration steps: {results['exploration_steps']}")
        print(f"Planning steps: {results['planning_steps']}")
        print(f"Planning successes: {results['planning_successes']}")
        print(f"Planner types used: {results['planner_types_used']}")

        print("\nHypothesis stats:")
        for k, v in results["hypothesis_stats"].items():
            print(f"  {k}: {v}")

        print("\nGoal inference stats:")
        for k, v in results["goal_stats"].items():
            print(f"  {k}: {v}")

        print("\nWorld model stats:")
        for k, v in results["world_model_stats"].items():
            print(f"  {k}: {v}")

        print("\nVerifier stats:")
        for k, v in results["verifier_stats"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
