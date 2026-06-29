"""
Test the cascade interpreter on real cascades from the replay data.

For each game:
  - Find all records with N>1 subframes (cascade records)
  - Run interpret_cascade on them
  - Print event type distribution
  - Save annotated PNGs of the most interesting cascades
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


from alphamoo.cascade_interpreter import interpret_cascade
from alphamoo.frame_renderer import save_cascade_png
from alphamoo.vtx_reader import load_replays_from_dir


def main():
    data_dir = Path("/home/z/my-project/alphamoo/data")
    out_dir = Path("/home/z/my-project/alphamoo/download/cascade_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading all replays...")
    replays = load_replays_from_dir(data_dir)

    print(f"\n{'Game':<8} {'Cascades':>10} {'MaxN':>5} {'Appear':>7} {'Disappear':>10} {'Move':>6} {'ColorChg':>9} {'LevelTr':>9}")
    print("-" * 75)

    total_cascade_records = 0
    total_event_counts = defaultdict(int)
    biggest_cascades = []  # (n_subframes, game_id, record_idx)

    for game_id, replay in sorted(replays.items()):
        cascade_count = 0
        event_counts = defaultdict(int)
        max_n = 1

        for i, record in enumerate(replay.records):
            if record.n_subframes > 1:
                cascade_count += 1
                max_n = max(max_n, record.n_subframes)
                biggest_cascades.append((record.n_subframes, game_id, i))

                # Interpret the cascade
                _, events = interpret_cascade(record.frame)
                for e in events:
                    event_counts[e.type] += 1


        total_cascade_records += cascade_count
        for k, v in event_counts.items():
            total_event_counts[k] += v

        print(f"{game_id:<8} {cascade_count:>10} {max_n:>5} "
              f"{event_counts.get('appearance', 0):>7} "
              f"{event_counts.get('disappearance', 0):>10} "
              f"{event_counts.get('move', 0):>6} "
              f"{event_counts.get('color_change', 0):>9} "
              f"{event_counts.get('level_transition', 0):>9}")

    print("-" * 75)
    print(f"{'TOTAL':<8} {total_cascade_records:>10}")
    print("\nEvent type distribution across all cascades:")
    for event_type, count in sorted(total_event_counts.items()):
        print(f"  {event_type}: {count}")

    # Save the 5 biggest cascades as PNGs
    biggest_cascades.sort(reverse=True)
    print("\n=== Saving 5 biggest cascades as PNGs ===")
    for n_sub, game_id, rec_idx in biggest_cascades[:5]:
        replay = replays[game_id]
        record = replay.records[rec_idx]
        out_path = out_dir / f"{game_id}_cascade_record{rec_idx}_n{n_sub}.png"
        save_cascade_png(record.frame, out_path, scale=2)
        print(f"  Saved {out_path.name}  ({n_sub} subframes, action id={record.action_input.id})")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
