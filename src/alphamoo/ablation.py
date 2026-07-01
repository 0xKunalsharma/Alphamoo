"""
AlphaMoo v4.1 — Ablation Framework (Phase 5).

Runs the agent with different modules enabled/disabled to measure
each module's contribution to performance.

Ablations:
  1. Full agent (all modules)
  2. No hypothesis generator
  3. No goal inference
  4. No near-miss tracker
  5. No world model
  6. No verifier
  7. No context compressor (no cross-level transfer)
  8. No planner interface (exploration only)
  9. No experiment planner (random exploration)

Each ablation runs on the 25 public demo games and reports:
  - Win rate
  - Avg actions per level
  - Goal inference accuracy
  - Planning success rate
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .agent_runner import AgentConfig, AlphaMooAgent

# =============================================================================
# Ablation definitions
# =============================================================================

@dataclass
class AblationResult:
    """Result of one ablation run."""
    name: str
    description: str
    config: AgentConfig
    games_played: int = 0
    total_wins: int = 0
    total_loses: int = 0
    total_actions: int = 0
    avg_actions_per_level: float = 0.0
    win_rate: float = 0.0
    planning_success_rate: float = 0.0
    avg_goal_confidence: float = 0.0
    avg_wm_rules: float = 0.0
    notes: str = ""


ABLATIONS: list[tuple[str, str, dict]] = [
    (
        "full_agent",
        "All modules enabled (baseline)",
        {},
    ),
    (
        "no_hypothesis_generator",
        "Disable Module 5 (Hypothesis Generator)",
        {"enable_hypothesis_generator": False},
    ),
    (
        "no_goal_inference",
        "Disable Module 6 (Goal Inference)",
        {"enable_goal_inference": False},
    ),
    (
        "no_near_miss_tracker",
        "Disable Module 7 (Near-Miss Tracker)",
        {"enable_near_miss_tracker": False},
    ),
    (
        "no_world_model",
        "Disable Module 9 (World Model) — no planning possible",
        {"enable_world_model": False, "enable_planner_interface": False},
    ),
    (
        "no_verifier",
        "Disable Module 10 (Verifier) — no rule downgrade",
        {"enable_verifier": False},
    ),
    (
        "no_context_compressor",
        "Disable Module 12 (Context Compressor) — no cross-level transfer",
        {"enable_context_compressor": False},
    ),
    (
        "no_planner_interface",
        "Disable Module 11 (Planner Interface) — exploration only",
        {"enable_planner_interface": False},
    ),
    (
        "no_experiment_planner",
        "Disable Module 8 (Experiment Planner) — random exploration",
        {"enable_experiment_planner": False},
    ),
]


# =============================================================================
# Ablation runner
# =============================================================================

def run_ablation(
    ablation_name: str,
    ablation_config_overrides: dict,
    games_to_test: list[str],
    data_dir: Path,
    max_steps_per_game: int = 100,
    use_real_llm: bool = False,
) -> AblationResult:
    """
    Run one ablation on a set of games.

    Args:
        ablation_name: name of the ablation
        ablation_config_overrides: dict of config overrides
        games_to_test: list of game IDs to test on
        data_dir: path to data directory
        max_steps_per_game: cap on steps per game
        use_real_llm: if True, use real LLM (requires GPU)

    Returns:
        AblationResult with aggregated stats.
    """
    from .schemas import GameState
    from .vtx_reader import load_replays_from_dir

    # Build config
    config = AgentConfig(
        use_real_llm=use_real_llm,
        **ablation_config_overrides,
    )

    # Load replays
    replays = load_replays_from_dir(data_dir)

    result = AblationResult(
        name=ablation_name,
        description=next(
            (desc for name, desc, _ in ABLATIONS if name == ablation_name),
            ablation_name,
        ),
        config=config,
    )

    goal_confidences = []
    wm_rule_counts = []

    for game_id in games_to_test:
        if game_id not in replays:
            continue

        replay = replays[game_id]
        agent = AlphaMooAgent(config)
        agent.start_game(game_id)

        n_steps = min(max_steps_per_game, len(replay.records))

        for i in range(n_steps):
            record = replay.records[i]
            step_result = agent.step(
                grid=record.final_grid,
                action_id_taken=record.action_input.id,
                available_actions=record.available_actions,
                n_subframes=record.n_subframes,
                subframes=record.frame if record.n_subframes > 1 else None,
                state=record.state,
                levels_completed=record.levels_completed,
                win_levels=record.win_levels,
            )

            if record.state == GameState.WIN:
                result.total_wins += 1
                break
            elif record.state == GameState.GAME_OVER:
                result.total_loses += 1
                break

            result.total_actions += 1
            goal_confidences.append(step_result.goal_confidence)
            wm_rule_counts.append(step_result.world_model_rules)

        agent.end_game()
        result.games_played += 1

    # Compute aggregates
    if result.games_played > 0:
        result.win_rate = result.total_wins / result.games_played
        result.avg_actions_per_level = result.total_actions / max(1, result.games_played)
        result.avg_goal_confidence = sum(goal_confidences) / max(1, len(goal_confidences))
        result.avg_wm_rules = sum(wm_rule_counts) / max(1, len(wm_rule_counts))

    stats = agent.get_stats()
    result.planning_success_rate = stats.get("planning_success_rate", 0.0)

    return result


def run_all_ablations(
    games_to_test: list[str],
    data_dir: Path,
    max_steps_per_game: int = 100,
    use_real_llm: bool = False,
) -> list[AblationResult]:
    """Run all defined ablations."""
    results: list[AblationResult] = []
    for name, _, overrides in ABLATIONS:
        print(f"\nRunning ablation: {name}")
        result = run_ablation(
            ablation_name=name,
            ablation_config_overrides=overrides,
            games_to_test=games_to_test,
            data_dir=data_dir,
            max_steps_per_game=max_steps_per_game,
            use_real_llm=use_real_llm,
        )
        results.append(result)
        print(f"  Win rate: {result.win_rate:.0%}")
        print(f"  Avg actions: {result.avg_actions_per_level:.0f}")
        print(f"  Avg goal conf: {result.avg_goal_confidence:.2f}")
    return results


def write_ablation_report(results: list[AblationResult], output_path: Path) -> None:
    """Write ablation results as markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# AlphaMoo v4.1 — Ablation Report\n",
        f"Games tested: {results[0].games_played if results else 0}",
        "Max steps per game: 100\n",
        "## Results\n",
        "| Ablation | Win Rate | Avg Actions | Goal Conf | WM Rules | Planning Success |",
        "|----------|----------|-------------|-----------|----------|------------------|",
    ]

    for r in results:
        lines.append(
            f"| {r.name} | {r.win_rate:.0%} | {r.avg_actions_per_level:.0f} | "
            f"{r.avg_goal_confidence:.2f} | {r.avg_wm_rules:.1f} | "
            f"{r.planning_success_rate:.0%} |"
        )

    lines.extend([
        "\n## Interpretation\n",
        "- Compare each ablation to `full_agent` (baseline)",
        "- Large drops in win rate indicate the module is critical",
        "- Small drops indicate the module is helpful but not essential",
        "- No change suggests the module isn't contributing (investigate why)",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nAblation report written to: {output_path}")
