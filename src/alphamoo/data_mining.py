"""
AlphaMoo v4.1 — Data mining pipeline.

Mines the 14,798 ground-truth actions across 25 public demo games for
patterns that bootstrap Phase 1 sub-modules:

  - Action distribution per game (which actions are used how often)
  - Color co-occurrence (which colors appear together — for AffordanceNet)
  - Object lifetime events (appear/disappear — for Type Inferencer)
  - Action → state-change pairs (for Hypothesis Generator priors)
  - Cascade statistics (how often do cascades happen, how big)
  - Per-level action counts (RHAE denominators — for Planner budgeting)
  - Action efficiency by game (which games humans find hard)

Output: structured JSON report + human-readable markdown.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .schemas import MAX_COLOR
from .vtx_reader import Replay, iter_actions, load_replays_from_dir

# =============================================================================
# Mining data structures
# =============================================================================

@dataclass
class GameStats:
    game_id: str
    n_records: int
    n_actions: int
    n_levels: int
    won: bool
    available_actions: list[int]
    action_counts: dict[int, int]  # action_id -> count
    cascade_counts: dict[int, int]  # n_subframes -> count of records
    color_distribution: dict[int, int]  # color_idx -> total cells across all frames
    actions_per_level: dict[int, int]  # level -> action count (if level data exists)
    avg_actions_per_record: float
    max_cascade_size: int


@dataclass
class CrossGameStats:
    n_games: int
    total_actions: int
    total_levels: int
    won_count: int
    action_distribution: dict[int, int]  # action_id -> total count across all games
    action_per_game: dict[str, dict[int, int]]  # game_id -> action_counts
    games_by_action_set: dict[str, list[str]]  # action_set_signature -> [game_ids]
    color_usage_per_game: dict[str, dict[int, int]]  # game_id -> color_counts
    rare_colors: list[int]  # colors used in <50% of games
    common_colors: list[int]  # colors used in >80% of games
    avg_actions_per_game: float
    hardest_games: list[tuple[str, int]]  # (game_id, n_actions), top 5
    easiest_games: list[tuple[str, int]]  # (game_id, n_actions), bottom 5
    cascade_stats: dict[str, float]  # mean, median, max, pct_with_cascade


# =============================================================================
# Mining functions
# =============================================================================

def _color_distribution_for_record(record) -> dict[int, int]:
    """Count color occurrences in a record's final grid."""
    grid = np.array(record.final_grid)
    counts = {int(i): int((grid == i).sum())
              for i in range(MAX_COLOR + 1)
              if (grid == i).sum() > 0}
    return counts


def mine_game(replay: Replay) -> GameStats:
    """Mine stats for a single game replay."""
    action_counts: Counter = Counter()
    cascade_counts: Counter = Counter()
    color_counts: Counter = Counter()
    actions_per_level: dict[int, int] = {}

    prev_level = 0
    prev_action_idx = 0

    for i, record in enumerate(replay.records):
        # Action distribution
        action_id = record.action_input.id
        action_counts[action_id] += 1

        # Cascade stats
        cascade_counts[record.n_subframes] += 1

        # Color distribution
        for color, count in _color_distribution_for_record(record).items():
            color_counts[color] += count

        # Per-level action counts (if we have level info)
        if record.levels_completed > prev_level:
            for level in range(prev_level + 1, record.levels_completed + 1):
                actions_per_level[level] = i - prev_action_idx
            prev_level = record.levels_completed
            prev_action_idx = i

    # Compute averages
    n_records = len(replay.records)
    n_actions = replay.n_actions
    avg_actions_per_record = (n_actions / n_records) if n_records > 0 else 0.0
    max_cascade = max(cascade_counts.keys()) if cascade_counts else 1

    return GameStats(
        game_id=replay.game_id,
        n_records=n_records,
        n_actions=n_actions,
        n_levels=replay.n_levels,
        won=replay.won,
        available_actions=replay.available_actions,
        action_counts=dict(action_counts),
        cascade_counts=dict(cascade_counts),
        color_distribution=dict(color_counts),
        actions_per_level=actions_per_level,
        avg_actions_per_record=avg_actions_per_record,
        max_cascade_size=max_cascade,
    )


def mine_all_games(replays: dict[str, Replay]) -> tuple[list[GameStats], CrossGameStats]:
    """Mine stats across all games."""
    per_game = [mine_game(r) for r in replays.values()]

    # Cross-game aggregates
    action_distribution: Counter = Counter()
    action_per_game: dict[str, dict[int, int]] = {}
    games_by_action_set: dict[str, list[str]] = defaultdict(list)
    color_usage_per_game: dict[str, dict[int, int]] = {}
    games_using_color: dict[int, set[str]] = defaultdict(set)
    total_actions = 0
    total_levels = 0
    won_count = 0
    all_cascade_sizes: list[int] = []

    for gs in per_game:
        action_distribution.update(gs.action_counts)
        action_per_game[gs.game_id] = gs.action_counts
        action_set_sig = ",".join(str(a) for a in sorted(gs.available_actions))
        games_by_action_set[action_set_sig].append(gs.game_id)
        color_usage_per_game[gs.game_id] = gs.color_distribution
        for color in gs.color_distribution:
            games_using_color[color].add(gs.game_id)
        total_actions += gs.n_actions
        total_levels += gs.n_levels
        if gs.won:
            won_count += 1
        for n_sub, count in gs.cascade_counts.items():
            all_cascade_sizes.extend([n_sub] * count)

    n_games = len(per_game)
    avg_actions = total_actions / n_games if n_games > 0 else 0

    # Color commonality
    common_threshold = int(n_games * 0.8)
    rare_threshold = int(n_games * 0.5)
    common_colors = sorted([c for c, games in games_using_color.items()
                            if len(games) >= common_threshold])
    rare_colors = sorted([c for c, games in games_using_color.items()
                          if len(games) < rare_threshold])

    # Hardest / easiest games
    by_difficulty = sorted(per_game, key=lambda g: g.n_actions, reverse=True)
    hardest = [(g.game_id, g.n_actions) for g in by_difficulty[:5]]
    easiest = [(g.game_id, g.n_actions) for g in by_difficulty[-5:]]

    # Cascade stats
    if all_cascade_sizes:
        cascade_stats = {
            "mean": float(np.mean(all_cascade_sizes)),
            "median": float(np.median(all_cascade_sizes)),
            "max": int(np.max(all_cascade_sizes)),
            "pct_with_cascade": float(
                sum(1 for s in all_cascade_sizes if s > 1) / len(all_cascade_sizes)
            ),
        }
    else:
        cascade_stats = {"mean": 1.0, "median": 1, "max": 1, "pct_with_cascade": 0.0}

    cross = CrossGameStats(
        n_games=n_games,
        total_actions=total_actions,
        total_levels=total_levels,
        won_count=won_count,
        action_distribution=dict(action_distribution),
        action_per_game=action_per_game,
        games_by_action_set=dict(games_by_action_set),
        color_usage_per_game=color_usage_per_game,
        rare_colors=rare_colors,
        common_colors=common_colors,
        avg_actions_per_game=avg_actions,
        hardest_games=hardest,
        easiest_games=easiest,
        cascade_stats=cascade_stats,
    )

    return per_game, cross


# =============================================================================
# Object-event mining (deeper — for Hypothesis Generator priors)
# =============================================================================

@dataclass
class ObjectEvent:
    """One appearance or disappearance of an object between two frames."""
    game_id: str
    timestep: int
    event_type: str  # "appearance" | "disappearance"
    color: int
    n_cells: int
    bbox: tuple[int, int, int, int]  # (xmin, ymin, xmax, ymax)


def mine_object_events(replay: Replay,
                       background_colors: set[int] | None = None) -> list[ObjectEvent]:
    """
    Detect object appearance/disappearance events across a replay.

    Args:
        replay: Loaded Replay.
        background_colors: Colors to treat as background (not objects).
            Defaults to {0} (black).

    Returns:
        List of ObjectEvent, one per appearance/disappearance.
    """
    if background_colors is None:
        background_colors = {0}

    events: list[ObjectEvent] = []
    prev_grid: np.ndarray | None = None

    for i, record in enumerate(replay.records):
        curr_grid = np.array(record.final_grid, dtype=np.int8)

        if prev_grid is not None:
            # Find cells that changed
            curr_grid - prev_grid
            np.zeros_like(curr_grid, dtype=bool)
            np.zeros_like(curr_grid, dtype=bool)

            for color in range(MAX_COLOR + 1):
                if color in background_colors:
                    continue
                was = prev_grid == color
                now = curr_grid == color
                appeared = now & ~was & ~np.isin(prev_grid, list(background_colors))
                disappeared = was & ~now & ~np.isin(curr_grid, list(background_colors))
                if appeared.any():
                    ys, xs = np.where(appeared)
                    if len(xs) > 0:
                        events.append(ObjectEvent(
                            game_id=replay.game_id,
                            timestep=i,
                            event_type="appearance",
                            color=color,
                            n_cells=len(xs),
                            bbox=(int(xs.min()), int(ys.min()),
                                  int(xs.max()), int(ys.max())),
                        ))
                if disappeared.any():
                    ys, xs = np.where(disappeared)
                    if len(xs) > 0:
                        events.append(ObjectEvent(
                            game_id=replay.game_id,
                            timestep=i,
                            event_type="disappearance",
                            color=color,
                            n_cells=len(xs),
                            bbox=(int(xs.min()), int(ys.min()),
                                  int(xs.max()), int(ys.max())),
                        ))

        prev_grid = curr_grid

    return events


# =============================================================================
# Action-outcome correlation (for AffordanceNet training data)
# =============================================================================

@dataclass
class ActionOutcome:
    """One (state, action, next_state) triple for training sub-modules."""
    game_id: str
    timestep: int
    action_id: int
    click_coords: tuple[int, int] | None
    levels_completed_before: int
    levels_completed_after: int
    state_before: str
    state_after: str
    n_subframes: int
    cells_changed: int  # how many cells changed between final grids


def mine_action_outcomes(replay: Replay) -> list[ActionOutcome]:
    """Extract all (state, action, outcome) tuples from a replay."""
    outcomes: list[ActionOutcome] = []

    for (before, after) in iter_actions(replay):
        before_grid = np.array(before.final_grid, dtype=np.int8)
        after_grid = np.array(after.final_grid, dtype=np.int8)
        cells_changed = int((before_grid != after_grid).sum())

        outcomes.append(ActionOutcome(
            game_id=replay.game_id,
            timestep=replay.records.index(after),
            action_id=after.action_input.id,
            click_coords=after.action_input.click_coords,
            levels_completed_before=before.levels_completed,
            levels_completed_after=after.levels_completed,
            state_before=before.state,
            state_after=after.state,
            n_subframes=after.n_subframes,
            cells_changed=cells_changed,
        ))

    return outcomes


# =============================================================================
# Report generation
# =============================================================================

def write_report(per_game: list[GameStats],
                 cross: CrossGameStats,
                 output_path: Path,
                 object_events: dict[str, list[ObjectEvent]],
                 action_outcomes: dict[str, list[ActionOutcome]]) -> None:
    """Write a comprehensive markdown report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# AlphaMoo v4.1 — Data Mining Report\n")
    lines.append(f"Source: 25 public demo replays, {cross.total_actions:,} actions\n")
    lines.append("---\n")

    # Cross-game summary
    lines.append("## Cross-Game Summary\n")
    lines.append(f"- Games: {cross.n_games}")
    lines.append(f"- Total actions: {cross.total_actions:,}")
    lines.append(f"- Total levels: {cross.total_levels}")
    lines.append(f"- Won: {cross.won_count}/{cross.n_games}")
    lines.append(f"- Avg actions/game: {cross.avg_actions_per_game:.1f}")
    lines.append(f"- Cascade stats: mean={cross.cascade_stats['mean']:.2f}, "
                 f"max={cross.cascade_stats['max']}, "
                 f"% with cascade={cross.cascade_stats['pct_with_cascade']*100:.1f}%\n")

    # Action distribution
    lines.append("## Action Distribution (across all games)\n")
    lines.append("| Action ID | Name | Count | % of total |")
    lines.append("|-----------|------|-------|------------|")
    action_names = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                    5: "INTERACT", 6: "CLICK", 7: "UNDO"}
    total = sum(cross.action_distribution.values())
    for aid in sorted(cross.action_distribution.keys()):
        name = action_names.get(aid, f"?{aid}")
        count = cross.action_distribution[aid]
        pct = count / total * 100 if total else 0
        lines.append(f"| {aid} | {name} | {count:,} | {pct:.1f}% |")
    lines.append("")

    # Games by action set
    lines.append("## Games Grouped by Action Set\n")
    lines.append("Each row is a distinct subset of available actions.\n")
    lines.append("| Action Set | Games |")
    lines.append("|------------|-------|")
    for action_set, games in sorted(cross.games_by_action_set.items(),
                                     key=lambda x: -len(x[1])):
        lines.append(f"| {action_set} | {', '.join(games)} ({len(games)}) |")
    lines.append("")

    # Color usage
    lines.append("## Color Usage\n")
    lines.append(f"- **Common colors** (in ≥80% of games): {cross.common_colors}")
    lines.append(f"- **Rare colors** (in <50% of games): {cross.rare_colors}\n")

    lines.append("### Color usage per game (top 5 colors by cell count)\n")
    lines.append("| Game | Top colors (idx:cells) |")
    lines.append("|------|------------------------|")
    for game_id, color_counts in cross.color_usage_per_game.items():
        top = sorted(color_counts.items(), key=lambda x: -x[1])[:5]
        top_str = ", ".join(f"{c}:{n:,}" for c, n in top)
        lines.append(f"| {game_id} | {top_str} |")
    lines.append("")

    # Per-game table
    lines.append("## Per-Game Stats\n")
    lines.append("| Game | Acts | Lvls | Won | Max Cascade | Top Action | Top Color |")
    lines.append("|------|------|------|-----|-------------|------------|-----------|")
    for gs in sorted(per_game, key=lambda g: g.game_id):
        top_action = max(gs.action_counts.items(), key=lambda x: x[1])[0] if gs.action_counts else -1
        top_action_name = action_names.get(top_action, f"?{top_action}")
        top_color = max(gs.color_distribution.items(), key=lambda x: x[1])[0] if gs.color_distribution else -1
        won = "Y" if gs.won else "N"
        lines.append(f"| {gs.game_id} | {gs.n_actions:,} | {gs.n_levels} | {won} | "
                     f"{gs.max_cascade_size} | {top_action_name} | {top_color} |")
    lines.append("")

    # Hardest / easiest
    lines.append("## Difficulty Ranking\n")
    lines.append("### Hardest games (most actions)\n")
    for game_id, n in cross.hardest_games:
        lines.append(f"- **{game_id}**: {n:,} actions")
    lines.append("\n### Easiest games (fewest actions)\n")
    for game_id, n in cross.easiest_games:
        lines.append(f"- **{game_id}**: {n:,} actions")
    lines.append("")

    # Object events
    lines.append("## Object Event Mining\n")
    total_events = sum(len(events) for events in object_events.values())
    total_appearances = sum(1 for events in object_events.values()
                            for e in events if e.event_type == "appearance")
    total_disappearances = sum(1 for events in object_events.values()
                               for e in events if e.event_type == "disappearance")
    lines.append(f"- Total events detected: {total_events:,}")
    lines.append(f"- Appearances: {total_appearances:,}")
    lines.append(f"- Disappearances: {total_disappearances:,}\n")

    # Per-game event counts
    lines.append("### Events per game\n")
    lines.append("| Game | Appearances | Disappearances | Total |")
    lines.append("|------|-------------|----------------|-------|")
    for game_id, events in sorted(object_events.items()):
        apps = sum(1 for e in events if e.event_type == "appearance")
        dis = sum(1 for e in events if e.event_type == "disappearance")
        lines.append(f"| {game_id} | {apps} | {dis} | {apps + dis} |")
    lines.append("")

    # Action outcomes
    lines.append("## Action-Outcome Mining\n")
    total_outcomes = sum(len(o) for o in action_outcomes.values())
    level_advancing = sum(1 for outcomes in action_outcomes.values()
                          for o in outcomes
                          if o.levels_completed_after > o.levels_completed_before)
    state_changes = sum(1 for outcomes in action_outcomes.values()
                        for o in outcomes if o.cells_changed > 0)
    no_op = sum(1 for outcomes in action_outcomes.values()
                for o in outcomes if o.cells_changed == 0)
    lines.append(f"- Total (state, action, outcome) tuples: {total_outcomes:,}")
    lines.append(f"- Level-advancing actions: {level_advancing:,}")
    lines.append(f"- State-changing actions: {state_changes:,}")
    lines.append(f"- No-op actions (no cell change): {no_op:,}\n")

    # Write
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {output_path}")


def write_json_dump(per_game: list[GameStats],
                    cross: CrossGameStats,
                    object_events: dict[str, list[ObjectEvent]],
                    action_outcomes: dict[str, list[ActionOutcome]],
                    output_path: Path) -> None:
    """Write the full mining output as JSON for downstream tools."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dump = {
        "cross_game": asdict(cross),
        "per_game": [asdict(gs) for gs in per_game],
        "object_events": {
            game_id: [asdict(e) for e in events]
            for game_id, events in object_events.items()
        },
        "action_outcomes": {
            game_id: [asdict(o) for o in outcomes]
            for game_id, outcomes in action_outcomes.items()
        },
    }
    output_path.write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
    print(f"JSON dump written to: {output_path}")


# =============================================================================
# Main entry point
# =============================================================================

def run(data_dir: Path, output_dir: Path) -> None:
    """Run the full mining pipeline."""
    print(f"Loading replays from {data_dir}...")
    replays = load_replays_from_dir(data_dir)
    print(f"Loaded {len(replays)} replays")

    print("\nMining per-game and cross-game stats...")
    per_game, cross = mine_all_games(replays)

    print("Mining object events (appearances/disappearances)...")
    object_events: dict[str, list[ObjectEvent]] = {}
    for game_id, replay in replays.items():
        object_events[game_id] = mine_object_events(replay)
        print(f"  {game_id}: {len(object_events[game_id])} events")

    print("\nMining action-outcome pairs...")
    action_outcomes: dict[str, list[ActionOutcome]] = {}
    for game_id, replay in replays.items():
        action_outcomes[game_id] = mine_action_outcomes(replay)

    print("\nWriting reports...")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_report(per_game, cross, output_dir / "mining_report.md",
                 object_events, action_outcomes)
    write_json_dump(per_game, cross, object_events, action_outcomes,
                    output_dir / "mining_dump.json")

    print(f"\nDone. Outputs in {output_dir}/")
    print("  - mining_report.md (human-readable)")
    print("  - mining_dump.json (machine-readable)")


if __name__ == "__main__":
    import sys
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/z/my-project/alphamoo/data")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/home/z/my-project/alphamoo/download/mining")
    run(data_dir, output_dir)
