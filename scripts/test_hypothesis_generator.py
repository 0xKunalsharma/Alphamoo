"""
Test the Hypothesis Generator on real replay data.

For each game:
  - Run perception + cascade + tracker + hypothesis generator
  - Print the top 10 discovered hypotheses
  - Print overall stats
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alphamoo.hypothesis_generator import run_on_replay
from alphamoo.vtx_reader import load_replays_from_dir


def format_hypothesis(hyp) -> str:
    """Format a hypothesis for display."""
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


def main():
    data_dir = Path("/home/z/my-project/alphamoo/data")
    print("Loading all replays...")
    replays = load_replays_from_dir(data_dir)

    # Test on a representative sample: 1 movement game, 1 click game, 1 mixed
    test_games = ["r11l", "ls20", "ft09"]

    for game_id in test_games:
        if game_id not in replays:
            continue
        replay = replays[game_id]
        print(f"\n{'='*80}")
        print(f"Game: {game_id} ({replay.n_actions} actions, available: {replay.available_actions})")
        print(f"{'='*80}")

        # Cap at 100 steps for speed
        max_steps = min(100, len(replay.records))
        print(f"Running on first {max_steps} steps...")

        gen, stats = run_on_replay(replay, max_steps=max_steps)

        print("\nStats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        print("\nTop 10 hypotheses:")
        top = gen.get_top_hypotheses(k=10)
        for i, hyp in enumerate(top, 1):
            print(f"  [{i}] {format_hypothesis(hyp)}")

        confirmed = gen.get_confirmed_hypotheses(threshold=0.5)
        if confirmed:
            print(f"\nConfirmed hypotheses (conf >= 0.5): {len(confirmed)}")
            for hyp in confirmed[:5]:
                print(f"  ✓ {format_hypothesis(hyp)}")
        else:
            print("\nNo confirmed hypotheses yet (need more observations)")


if __name__ == "__main__":
    main()
