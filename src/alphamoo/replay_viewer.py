"""
AlphaMoo v4.1 — Replay Viewer CLI.

Lets you step through a replay and inspect what's happening at each step.

Usage:
    python -m alphamoo.replay_viewer <game_id> [--start N] [--end N] [--step N]
    python -m alphamoo.replay_viewer ls20 --start 0 --end 30 --step 1
    python -m alphamoo.replay_viewer ls20 --cascades-only
    python -m alphamoo.replay_viewer ls20 --level-transitions-only

For each step, prints:
    - Action taken (id, click coords if any)
    - State change (levels_completed delta)
    - Cascade size (N subframes)
    - Top 5 detected objects (color, n_cells, topology)
    - Top 3 detected events (if cascade)
"""
from __future__ import annotations

import argparse
import sys

from .cascade_interpreter import cascade_summary, interpret_cascade
from .perception import perceive_with_diagnostics
from .schemas import GameState
from .vtx_reader import load_replays_from_dir

ACTION_NAMES = {
    0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "INTERACT", 6: "CLICK", 7: "UNDO",
}


def print_step(record, prev_record=None, verbose=False):
    """Print one step's summary."""
    action_id = record.action_input.id
    action_name = ACTION_NAMES.get(action_id, f"?{action_id}")
    click = record.action_input.click_coords

    state_marker = ""
    if record.state == GameState.WIN:
        state_marker = " 🏆 WIN"
    elif record.state == GameState.GAME_OVER:
        state_marker = " 💀 GAME_OVER"

    level_delta = ""
    if prev_record and record.levels_completed > prev_record.levels_completed:
        level_delta = f"  ↑ LEVEL UP: {prev_record.levels_completed}→{record.levels_completed}"

    cascade_info = ""
    if record.n_subframes > 1:
        cascade_info = f"  [CASCADE x{record.n_subframes}]"

    click_info = f" at {click}" if click else ""
    print(f"\n[Step {record.action_input.id and 'action' or 'init'}] "
          f"{action_name}{click_info} → state={record.state}{state_marker}{level_delta}{cascade_info}")

    # Perceive the final grid
    diag = perceive_with_diagnostics(record.final_grid)
    scene = diag["scene_graph"]

    print(f"  Background: color={diag['background_color']}")
    print(f"  Objects: {len(scene.objects)}  Relations: {len(scene.edges)}")

    # Top 5 objects by size
    sorted_objs = sorted(scene.objects.values(),
                         key=lambda o: -len(o.cells))[:5]
    if sorted_objs:
        print("  Top objects (by size):")
        for obj in sorted_objs:
            print(f"    {obj.id}: color={obj.color}, cells={len(obj.cells)}, "
                  f"topology={obj.topology}, bbox={obj.bounding_box}")

    # Cascade events
    if record.n_subframes > 1:
        _, events = interpret_cascade(record.frame)
        summary = cascade_summary(events)
        print(f"  Cascade events: {summary['total_events']}")
        if verbose and events:
            for e in events[:5]:
                print(f"    {e.type}: target_color={e.target_color}, "
                      f"before={e.before}, after={e.after}")

    elif prev_record:
        # Single-subframe step: still diff against previous
        _, events = interpret_cascade(record.frame, prev_scene=scene)
        if events:
            print(f"  Events vs previous: {len(events)}")
            if verbose:
                for e in events[:3]:
                    print(f"    {e.type}: {e.before} → {e.after}")


def main():
    parser = argparse.ArgumentParser(
        description="Step through an ARC-AGI-3 replay"
    )
    parser.add_argument("game_id", help="Game ID (e.g. ls20)")
    parser.add_argument("--data-dir", default="/home/z/my-project/alphamoo/data",
                        help="Directory containing .vtx files")
    parser.add_argument("--start", type=int, default=0,
                        help="Start record index (default 0)")
    parser.add_argument("--end", type=int, default=None,
                        help="End record index (default: all)")
    parser.add_argument("--step", type=int, default=1,
                        help="Step size (default 1)")
    parser.add_argument("--cascades-only", action="store_true",
                        help="Only show records with N>1 subframes")
    parser.add_argument("--level-transitions-only", action="store_true",
                        help="Only show records where levels_completed increased")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show event details")
    args = parser.parse_args()

    # Load all replays, find the requested one
    replays = load_replays_from_dir(args.data_dir)
    if args.game_id not in replays:
        print(f"Game '{args.game_id}' not found. Available:")
        for gid in sorted(replays.keys()):
            print(f"  {gid}")
        sys.exit(1)

    replay = replays[args.game_id]
    print(f"=== Replay: {args.game_id} ===")
    print(f"Records: {len(replay.records)}  Actions: {replay.n_actions}  "
          f"Levels: {replay.n_levels}  Won: {replay.won}")
    print(f"Available actions: {replay.available_actions}")

    end = args.end if args.end is not None else len(replay.records)
    prev = None
    shown = 0

    for i in range(args.start, min(end, len(replay.records))):
        record = replay.records[i]

        if args.cascades_only and record.n_subframes <= 1:
            prev = record
            continue
        if args.level_transitions_only and (
            prev is None or record.levels_completed <= prev.levels_completed
        ):
            prev = record
            continue
        if (i - args.start) % args.step != 0:
            prev = record
            continue

        print_step(record, prev, verbose=args.verbose)
        shown += 1
        prev = record

    print(f"\n=== Shown {shown} records ===")


if __name__ == "__main__":
    main()
