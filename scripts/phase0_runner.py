"""
AlphaMoo v4.1 — Phase 0 Runner.

Runs the agent loop on multiple games with multiple latency models, produces
a ComputeProfile report that determines which model + quant configuration
fits the 9-hour Kaggle budget.

Usage:
    python scripts/phase0_runner.py [--games ls20 r11l ft09] [--max-steps 100]
    python scripts/phase0_runner.py --all  # run on all 25 games
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alphamoo.agent_loop import run_phase0_on_game
from alphamoo.llm_stub import LATENCY_MODELS

# Kaggle eval constraints (verified from rules)
KAGGLE_BUDGET_HOURS = 9
KAGGLE_BUDGET_SEC = KAGGLE_BUDGET_HOURS * 3600  # 32400 sec
ESTIMATED_HIDDEN_ACTIONS = 14800  # same as public set, conservative estimate


def fmt_ms(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms/1000:.2f}s"


def run_one(game_id: str, latency_model: str, max_steps: int | None, verbose: bool):
    """Run Phase 0 on one game with one latency model. Returns (loop_stats, llm_stats)."""
    try:
        loop_stats, llm_stats = run_phase0_on_game(
            game_id=game_id,
            latency_model=latency_model,
            max_output_tokens=80,
            max_steps=max_steps,
            verbose=verbose,
        )
        return loop_stats, llm_stats
    except Exception as e:
        print(f"  [ERROR] {game_id} with {latency_model}: {e}")
        return None, None


def print_per_game_table(results: list[dict]):
    """Print per-game results table."""
    print(f"\n{'='*120}")
    print(f"{'Game':<8} {'Model':<22} {'Steps':>6} {'Avg ms':>10} {'P50 ms':>10} "
          f"{'P95 ms':>10} {'P99 ms':>10} {'Max ms':>10} {'LLM %':>7}")
    print("-" * 120)
    for r in results:
        ls = r["loop_stats"]
        if ls is None:
            continue
        print(f"{r['game']:<8} {r['latency_model']:<22} {ls.n_steps:>6} "
              f"{fmt_ms(ls.avg_step_ms):>10} {fmt_ms(ls.p50_step_ms):>10} "
              f"{fmt_ms(ls.p95_step_ms):>10} {fmt_ms(ls.p99_step_ms):>10} "
              f"{fmt_ms(ls.max_step_ms):>10} {ls.avg_llm_pct:>6.1f}%")
    print("=" * 120)


def compute_budget_assessment(results: list[dict]) -> dict:
    """Compute whether each latency model fits the Kaggle budget."""
    by_model: dict[str, list] = {}
    for r in results:
        if r["loop_stats"] is None:
            continue
        by_model.setdefault(r["latency_model"], []).append(r)

    assessments = {}
    for model, runs in by_model.items():
        total_actions = sum(r["loop_stats"].n_steps for r in runs)
        if total_actions == 0:
            continue
        # Extrapolate: avg_step_ms × ESTIMATED_HIDDEN_ACTIONS = total wall-clock for hidden set
        # (assumes hidden set is similar in difficulty to public)
        avg_step_ms = sum(r["loop_stats"].avg_step_ms * r["loop_stats"].n_steps
                          for r in runs) / total_actions
        projected_total_sec = avg_step_ms * ESTIMATED_HIDDEN_ACTIONS / 1000
        fits = projected_total_sec <= KAGGLE_BUDGET_SEC
        headroom_pct = (1 - projected_total_sec / KAGGLE_BUDGET_SEC) * 100 if fits else 0
        assessments[model] = {
            "total_actions_measured": total_actions,
            "avg_step_ms": avg_step_ms,
            "projected_total_sec": projected_total_sec,
            "projected_total_hr": projected_total_sec / 3600,
            "fits_budget": fits,
            "headroom_pct": headroom_pct,
            "max_actions_allowed": int(KAGGLE_BUDGET_SEC * 1000 / avg_step_ms) if avg_step_ms > 0 else 0,
        }
    return assessments


def write_report(results: list[dict], assessments: dict, output_path: Path):
    """Write a markdown ComputeProfile report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# AlphaMoo v4.1 — Phase 0 ComputeProfile Report",
        "",
        f"**Kaggle eval budget:** {KAGGLE_BUDGET_HOURS} hours ({KAGGLE_BUDGET_SEC:,} sec)",
        f"**Estimated hidden set actions:** {ESTIMATED_HIDDEN_ACTIONS:,}",
        f"**Games measured:** {len({r['game'] for r in results})}",
        f"**Latency models tested:** {len({r['latency_model'] for r in results})}",
        "",
        "## Per-Game Results",
        "",
        "| Game | Latency Model | Steps | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | LLM % |",
        "|------|---------------|-------|----------|----------|----------|----------|----------|-------|",
    ]
    for r in results:
        ls = r["loop_stats"]
        if ls is None:
            continue
        lines.append(
            f"| {r['game']} | {r['latency_model']} | {ls.n_steps} | "
            f"{ls.avg_step_ms:.1f} | {ls.p50_step_ms:.1f} | "
            f"{ls.p95_step_ms:.1f} | {ls.p99_step_ms:.1f} | "
            f"{ls.max_step_ms:.1f} | {ls.avg_llm_pct:.1f}% |"
        )

    lines.extend([
        "",
        "## Budget Assessment",
        "",
        "Each latency model is assessed against the 9-hour Kaggle budget,",
        "extrapolated to the estimated 14,800-action hidden set.",
        "",
        "| Latency Model | Avg Step (ms) | Projected Total | Fits Budget? | Headroom | Max Actions |",
        "|---------------|---------------|-----------------|--------------|----------|-------------|",
    ])
    for model, a in sorted(assessments.items(), key=lambda x: x[1]["avg_step_ms"]):
        fits = "✅ YES" if a["fits_budget"] else "❌ NO"
        headroom = f"{a['headroom_pct']:.1f}%" if a["fits_budget"] else "OVER"
        lines.append(
            f"| {model} | {a['avg_step_ms']:.1f} | "
            f"{a['projected_total_hr']:.2f}h ({a['projected_total_sec']:.0f}s) | "
            f"{fits} | {headroom} | {a['max_actions_allowed']:,} |"
        )

    lines.extend([
        "",
        "## Pipeline Cost Breakdown",
        "",
        "Average time spent in each pipeline stage, across all runs:",
        "",
        "| Stage | Total Time | % of Total | Notes |",
        "|-------|------------|------------|-------|",
    ])

    # Aggregate across all results
    total_perception = sum(r["loop_stats"].total_perception_ms for r in results if r["loop_stats"])
    total_cascade = sum(r["loop_stats"].total_cascade_ms for r in results if r["loop_stats"])
    total_tracker = sum(r["loop_stats"].total_tracker_ms for r in results if r["loop_stats"])
    total_prompt = sum(r["loop_stats"].total_prompt_build_ms for r in results if r["loop_stats"])
    total_llm = sum(r["loop_stats"].total_llm_ms for r in results if r["loop_stats"])
    grand_total = total_perception + total_cascade + total_tracker + total_prompt + total_llm

    if grand_total > 0:
        lines.append(f"| Perception | {total_perception/1000:.2f}s | {total_perception/grand_total*100:.1f}% | CCL + object extraction |")
        lines.append(f"| Cascade | {total_cascade/1000:.2f}s | {total_cascade/grand_total*100:.1f}% | Multi-subframe diff (only when N>1) |")
        lines.append(f"| Tracker | {total_tracker/1000:.2f}s | {total_tracker/grand_total*100:.1f}% | Agent State Tracker |")
        lines.append(f"| Prompt Build | {total_prompt/1000:.2f}s | {total_prompt/grand_total*100:.1f}% | Scene + agent → prompt string |")
        lines.append(f"| LLM | {total_llm/1000:.2f}s | {total_llm/grand_total*100:.1f}% | Stub with realistic latency model |")
        lines.append(f"| **Total** | **{grand_total/1000:.2f}s** | **100%** | |")

    lines.extend([
        "",
        "## Token Statistics",
        "",
        f"- Total prompt tokens generated: {sum(r['loop_stats'].total_prompt_tokens for r in results if r['loop_stats']):,}",
        f"- Total output tokens generated: {sum(r['loop_stats'].total_output_tokens for r in results if r['loop_stats']):,}",
        f"- Average prompt tokens/step: {sum(r['loop_stats'].avg_prompt_tokens * r['loop_stats'].n_steps for r in results if r['loop_stats']) / max(1, sum(r['loop_stats'].n_steps for r in results if r['loop_stats'])):.0f}",
        f"- Average output tokens/step: {sum(r['loop_stats'].avg_output_tokens * r['loop_stats'].n_steps for r in results if r['loop_stats']) / max(1, sum(r['loop_stats'].n_steps for r in results if r['loop_stats'])):.0f}",
        "",
        "## Recommendations",
        "",
    ])

    # Generate recommendations based on which models fit
    fitting = [m for m, a in assessments.items() if a["fits_budget"]]
    if fitting:
        best = min(fitting, key=lambda m: assessments[m]["avg_step_ms"])
        lines.append(f"### ✅ Recommended configuration: `{best}`")
        lines.append("")
        lines.append(f"- Avg step time: {assessments[best]['avg_step_ms']:.1f}ms")
        lines.append(f"- Projected total for hidden set: {assessments[best]['projected_total_hr']:.2f}h")
        lines.append(f"- Headroom: {assessments[best]['headroom_pct']:.1f}%")
        lines.append(f"- Max actions allowed: {assessments[best]['max_actions_allowed']:,}")
        lines.append("")
        if "qwen2.5-0.5b-4bit" in fitting:
            lines.append("- **0.5B is the safe default.** Use this for Phase 1+ development.")
        if "qwen2.5-1.5b-4bit" in fitting:
            lines.append("- **1.5B is the stretch goal.** Better reasoning quality if it fits.")
        if "vibethinker-3b-4bit" in fitting:
            lines.append("- **VibeThinker-3B is the alternative.** Test A/B against 1.5B before committing.")
    else:
        lines.append("### ❌ No tested configuration fits the budget")
        lines.append("")
        lines.append("Mitigation options:")
        lines.append("- Reduce `max_output_tokens` (currently 80)")
        lines.append("- Use shorter prompts (cap objects in scene description)")
        lines.append("- Batch actions across multiple games in parallel")
        lines.append("- Use continuous batching in vLLM")

    lines.extend([
        "",
        "## Methodology",
        "",
        "- Phase 0 uses a **replay-driven simulator**: we walk through ground-truth",
        "  replays frame by frame, measuring pipeline cost at each step.",
        "- The LLM is a **stub** with realistic latency model calibrated against",
        "  published 4-bit Qwen2.5 benchmarks on H100/RTX 6000-class GPUs.",
        "- Real LLM performance may vary ±30% from the stub.",
        "- Agent quality is intentionally not measured here — Phase 0 is about",
        "  pipeline cost, not intelligence. Phase 2+ measures agent quality.",
        "",
        "## Next Steps",
        "",
        "1. If 0.5B fits: proceed with Qwen2.5-0.5B-Instruct as default reasoning engine",
        "2. If 1.5B also fits: A/B test 0.5B vs 1.5B on a few games for quality",
        "3. Build Module 5 (Hypothesis Generator) — start of Phase 2",
        "4. Build Module 6 (Goal Inference) — Phase 2",
        "5. Build Module 11 (Planner Interface) — Phase 3",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 0 compute measurement")
    parser.add_argument("--games", nargs="+", default=["ls20", "r11l", "ft09", "wa30"],
                        help="Game IDs to test (default: ls20 r11l ft09 wa30)")
    parser.add_argument("--all", action="store_true",
                        help="Run on all 25 games (slow)")
    parser.add_argument("--models", nargs="+",
                        default=["qwen2.5-0.5b-4bit", "qwen2.5-1.5b-4bit", "vibethinker-3b-4bit"],
                        choices=list(LATENCY_MODELS.keys()),
                        help="Latency models to test")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap steps per game (for quick testing)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", default="/home/z/my-project/alphamoo/download/phase0/compute_profile.md",
                        help="Output report path")
    args = parser.parse_args()

    # Determine games to run
    if args.all:
        from alphamoo.vtx_reader import load_replays_from_dir
        replays = load_replays_from_dir("/home/z/my-project/alphamoo/data")
        games = sorted(replays.keys())
    else:
        games = args.games

    print("Phase 0 ComputeProfile Measurement")
    print(f"  Games: {games}")
    print(f"  Latency models: {args.models}")
    print(f"  Max steps per game: {args.max_steps or 'unlimited'}")
    print()

    results = []
    for game in games:
        for model in args.models:
            print(f"Running {game} with {model}...")
            loop_stats, llm_stats = run_one(game, model, args.max_steps, args.verbose)
            results.append({
                "game": game,
                "latency_model": model,
                "loop_stats": loop_stats,
                "llm_stats": llm_stats,
            })
            if loop_stats:
                print(f"  → {loop_stats.n_steps} steps, avg {loop_stats.avg_step_ms:.1f}ms/step, "
                      f"total {loop_stats.total_wall_clock_sec:.1f}s")

    # Print summary table
    print_per_game_table(results)

    # Compute budget assessment
    assessments = compute_budget_assessment(results)
    print(f"\n=== Budget Assessment (9hr Kaggle, {ESTIMATED_HIDDEN_ACTIONS:,} actions) ===")
    print(f"{'Model':<22} {'Avg ms':>10} {'Projected':>12} {'Fits?':>8} {'Headroom':>10}")
    print("-" * 65)
    for model, a in sorted(assessments.items(), key=lambda x: x[1]["avg_step_ms"]):
        fits = "✅ YES" if a["fits_budget"] else "❌ NO"
        headroom = f"{a['headroom_pct']:.1f}%" if a["fits_budget"] else "OVER"
        print(f"{model:<22} {a['avg_step_ms']:>10.1f} "
              f"{a['projected_total_hr']:>10.2f}h {fits:>8} {headroom:>10}")

    # Write report
    output_path = Path(args.output)
    write_report(results, assessments, output_path)

    print(f"\nDone. Report: {output_path}")


if __name__ == "__main__":
    main()
