"""
Quick smoke test for vtx_reader.

Loads one replay and prints stats. Run with:
    python /home/z/my-project/alphamoo/scripts/test_reader.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alphamoo.vtx_reader import load_replay, load_replays_from_dir

DATA_DIR = Path("/home/z/my-project/alphamoo/data")

def test_single_replay():
    # Use ls20 (smallest of the well-known ones, 1.5MB)
    vtx_files = list(DATA_DIR.glob("ls20-*.vtx"))
    assert vtx_files, "No ls20 replay found"

    print(f"Loading: {vtx_files[0].name}")
    replay = load_replay(vtx_files[0])

    print("\n=== Replay summary ===")
    print(f"Game ID: {replay.game_id}")
    print(f"GUID: {replay.guid}")
    print(f"Records: {len(replay.records)}")
    print(f"Actions: {replay.n_actions}")
    print(f"Levels: {replay.n_levels}")
    print(f"Won: {replay.won}")
    print(f"Available actions: {replay.available_actions}")

    # First record (initial state)
    first = replay.records[0]
    print("\n=== First record ===")
    print(f"Timestamp: {first.timestamp}")
    print(f"State: {first.state}")
    print(f"Levels completed: {first.levels_completed}")
    print(f"Action input: id={first.action_input.id}, data={first.action_input.data}")
    print(f"N subframes: {first.n_subframes}")
    print(f"Grid shape: {len(first.final_grid)}x{len(first.final_grid[0])}")

    # Sample a mid-game action
    mid_idx = len(replay.records) // 2
    mid = replay.records[mid_idx]
    print(f"\n=== Mid-game record (index {mid_idx}) ===")
    print(f"State: {mid.state}")
    print(f"Levels completed: {mid.levels_completed}")
    print(f"Action: id={mid.action_input.id}")
    if mid.action_input.is_click:
        print(f"Click coords: {mid.action_input.click_coords}")
    print(f"N subframes: {mid.n_subframes}")

    # Final record
    last = replay.records[-1]
    print("\n=== Final record ===")
    print(f"State: {last.state}")
    print(f"Levels completed: {last.levels_completed}")
    print(f"Total actions: {replay.summary.total_actions}")

    # Summary
    print("\n=== Summary ===")
    print(f"Total plays: {replay.summary.total_plays}")
    print(f"Final state: {replay.summary.final_states}")
    if replay.summary.actions_by_level:
        print("Actions by level:")
        for level_sum in replay.summary.actions_by_level[0]:
            print(f"  Level {level_sum.level}: {level_sum.cumulative_actions} cumulative actions")

    return replay


def test_all_replays():
    print("\n\n=== Loading all 25 replays ===")
    replays = load_replays_from_dir(DATA_DIR)

    print(f"\nLoaded {len(replays)} replays")
    print(f"\n{'Game':<8} {'Recs':>5} {'Acts':>5} {'Lvls':>5} {'Won':>5} {'Actions Available':<25}")
    print("-" * 65)

    total_actions = 0
    total_levels = 0
    won_count = 0

    for game_id, replay in sorted(replays.items()):
        n_actions = replay.n_actions
        n_levels = replay.n_levels
        won = "Y" if replay.won else "N"
        avail = ",".join(str(a) for a in replay.available_actions)
        print(f"{game_id:<8} {len(replay.records):>5} {n_actions:>5} {n_levels:>5} {won:>5} {avail:<25}")
        total_actions += n_actions
        total_levels += n_levels
        if replay.won:
            won_count += 1

    print("-" * 65)
    print(f"{'TOTAL':<8} {'':>5} {total_actions:>5} {total_levels:>5} {won_count}/25:>5")
    print(f"\nTotal actions across all replays: {total_actions:,}")
    print(f"Games won: {won_count}/{len(replays)}")


if __name__ == "__main__":
    test_single_replay()
    test_all_replays()
