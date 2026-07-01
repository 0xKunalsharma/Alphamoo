"""
Phase 2 integration test — the full discovery loop.

Wires together:
  Perception → AgentTracker → CascadeInterpreter →
  HypothesisGenerator → GoalInference → NearMissTracker → ExperimentPlanner

Runs against real replay data and shows the agent actually doing
discovery: forming hypotheses, inferring goals, learning from deaths,
and choosing informative actions.
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
from alphamoo.schemas import GameState
from alphamoo.vtx_reader import load_replays_from_dir


def format_hypothesis(hyp) -> str:
    conds = []
    for cond in hyp.trigger.conditions:
        neg = "NOT " if cond.negated else ""
        args_str = ", ".join(f"{k}={v}" for k, v in cond.args.items())
        conds.append(f"{neg}{cond.predicate}({args_str})")
    trigger_str = " AND ".join(conds) if conds else "(always)"
    eff_args = ", ".join(f"{k}={v}" for k, v in hyp.effect.get("args", {}).items())
    effect_str = f"{hyp.effect['type']}({eff_args})"
    return (f"IF {trigger_str} THEN {effect_str} "
            f"[conf={hyp.confidence:.2f}, support={hyp.support}]")


def format_goal(goal) -> str:
    args_str = ", ".join(f"{k}={v}" for k, v in goal.args.items())
    return (f"{goal.terminal_condition}({args_str}) "
            f"[conf={goal.confidence:.2f}, support={goal.support}, "
            f"near_miss={goal.near_miss_support}]")


def run_phase2_loop(replay, max_steps=100, verbose=False):
    """Run the full Phase 2 discovery loop on one replay."""
    tracker = AgentStateTracker()
    hyp_gen = HypothesisGenerator()
    goal_module = GoalInferenceModule()
    near_miss = NearMissTracker()
    planner = ExperimentPlanner(ig_floor=0.05, epsilon=0.1, rng_seed=42)

    n = min(max_steps, len(replay.records))
    prev_grid = None
    prev_record = None

    planner_actions = []
    wins_detected = 0
    loses_detected = 0

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

        # Update hypothesis generator
        hyp_gen.observe(scene, agent_state, events, record.action_input.id)

        # Update goal inference
        goal_module.observe_step(scene, agent_state, record.action_input.id)

        # Check for level transition / win
        if prev_record and record.levels_completed > prev_record.levels_completed:
            goal_module.observe_win(scene, agent_state)
            wins_detected += 1
            # Reset near-miss tracker for new "episode"
            near_miss.reset()
        elif record.state == GameState.WIN:
            goal_module.observe_win(scene, agent_state)
            wins_detected += 1
        elif record.state == GameState.GAME_OVER:
            near_miss_predicates = near_miss.on_episode_end("GAME_OVER")
            if near_miss_predicates:
                goal_module.observe_lose(scene, agent_state, near_miss_predicates)
            loses_detected += 1
            near_miss.reset()
        else:
            near_miss.record_step(scene, agent_state)

        # Experiment planner chooses next action (we don't execute it; we measure)
        if i < n - 1:
            top_hyps = hyp_gen.get_top_hypotheses(k=10)
            chosen = planner.select_action(
                available_actions=record.available_actions,
                scene=scene,
                agent_state=agent_state,
                hypotheses=top_hyps,
                goal_module=goal_module,
            )
            planner_actions.append(chosen.action_id)
            planner.record_visited_state(scene, agent_state, chosen.action_id)

        prev_grid = grid
        prev_record = record

        if verbose and i % 25 == 0:
            print(f"\n  Step {i}:")
            print(f"    Scene: {len(scene.objects)} objects, agent: {agent_state.position if agent_state else 'None'}")
            print(f"    Events: {len(events)}")
            print(f"    Hypotheses: {len(hyp_gen.hypotheses)} (top conf: {max((h.confidence for h in hyp_gen.hypotheses), default=0):.2f})")
            print(f"    Goal hypotheses: {len(goal_module.hypotheses)} (top conf: {goal_module.get_top_goal().confidence if goal_module.get_top_goal() else 0:.2f})")
            print(f"    Ready to plan: {goal_module.is_ready_to_plan()}")

    return {
        "steps": n,
        "wins_detected": wins_detected,
        "loses_detected": loses_detected,
        "planner_actions": planner_actions,
        "hypothesis_stats": hyp_gen.get_stats(),
        "goal_stats": goal_module.get_stats(),
        "near_miss_stats": near_miss.get_stats(),
        "planner_stats": planner.get_stats(),
        "top_hypotheses": hyp_gen.get_top_hypotheses(k=5),
        "top_goals": goal_module.get_top_goals(k=5),
    }


def main():
    data_dir = Path("/home/z/my-project/alphamoo/data")
    print("Loading all replays...")
    replays = load_replays_from_dir(data_dir)

    # Test on 3 representative games
    test_games = ["ls20", "ft09", "r11l"]

    for game_id in test_games:
        if game_id not in replays:
            continue
        replay = replays[game_id]
        print(f"\n{'='*80}")
        print(f"Game: {game_id} ({replay.n_actions} actions, available: {replay.available_actions})")
        print(f"{'='*80}")

        max_steps = min(100, len(replay.records))
        print(f"Running full Phase 2 loop on first {max_steps} steps...")

        results = run_phase2_loop(replay, max_steps=max_steps, verbose=True)

        print("\n--- Results ---")
        print(f"Steps: {results['steps']}")
        print(f"Wins detected: {results['wins_detected']}")
        print(f"Loses detected: {results['loses_detected']}")
        print(f"Planner actions: {results['planner_actions'][:20]}...")

        print("\nHypothesis stats:")
        for k, v in results["hypothesis_stats"].items():
            print(f"  {k}: {v}")

        print("\nGoal inference stats:")
        for k, v in results["goal_stats"].items():
            print(f"  {k}: {v}")

        print("\nNear-miss stats:")
        for k, v in results["near_miss_stats"].items():
            print(f"  {k}: {v}")

        print("\nPlanner stats:")
        for k, v in results["planner_stats"].items():
            print(f"  {k}: {v}")

        print("\nTop 5 mechanics hypotheses:")
        for i, hyp in enumerate(results["top_hypotheses"], 1):
            print(f"  [{i}] {format_hypothesis(hyp)}")

        print("\nTop 5 goal hypotheses:")
        for i, goal in enumerate(results["top_goals"], 1):
            print(f"  [{i}] {format_goal(goal)}")

        print(f"\nReady to plan: {results['goal_stats']['ready_to_plan']}")


if __name__ == "__main__":
    main()
