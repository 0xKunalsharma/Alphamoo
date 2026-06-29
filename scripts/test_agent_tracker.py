"""
Test the Agent State Tracker on all 25 games.

For each game:
  - Run the tracker over every record
  - Print tracking stats (detection rate, color changes, shape changes)
  - Save annotated PNGs showing the tracked agent at key moments:
      * First detection
      * First color change
      * First shape change
      * Level transitions
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw

from alphamoo.agent_tracker import track_replay
from alphamoo.frame_renderer import grid_to_rgb
from alphamoo.perception import detect_background_color
from alphamoo.schemas import GRID_SIZE
from alphamoo.vtx_reader import load_replays_from_dir


def draw_agent_overlay(grid, agent_state, diag, scale=10):
    """Render grid with the tracked agent highlighted in green."""
    rgb = grid_to_rgb(grid)
    img = Image.fromarray(rgb, mode="RGB").resize(
        (GRID_SIZE * scale, GRID_SIZE * scale), Image.NEAREST
    )
    draw = ImageDraw.Draw(img, "RGBA")

    if agent_state and agent_state.position:
        # Highlight the agent's bounding box (approximate from position)
        cx, cy = agent_state.position
        # Find actual agent cells by checking colors in a region around centroid
        grid_np = np.array(grid, dtype=np.int8)
        bg = detect_background_color(grid_np)
        # Search 10x10 region around centroid
        x0 = max(0, cx - 5)
        y0 = max(0, cy - 5)
        x1 = min(GRID_SIZE - 1, cx + 5)
        y1 = min(GRID_SIZE - 1, cy + 5)
        agent_cells = []
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                c = int(grid_np[y, x])
                if c in agent_state.color and c != bg:
                    agent_cells.append((x, y))
        # Draw cells with green overlay
        for x, y in agent_cells:
            draw.rectangle(
                [x * scale, y * scale, (x + 1) * scale, (y + 1) * scale],
                fill=(0, 255, 0, 100),
                outline=(0, 255, 0, 255),
            )

        # Bounding box
        if agent_cells:
            xs = [c[0] for c in agent_cells]
            ys = [c[1] for c in agent_cells]
            draw.rectangle(
                [min(xs) * scale, min(ys) * scale,
                 (max(xs) + 1) * scale, (max(ys) + 1) * scale],
                outline=(255, 255, 0, 255), width=2
            )

    # Text overlay
    info_lines = [
        f"Method: {diag.detection_method}",
        f"Cells: {diag.n_agent_cells}",
        f"Colors: {diag.agent_colors}",
        f"Moved: {diag.moved} ({diag.displacement})",
        f"ColorChg: {diag.color_changed}",
        f"ShapeChg: {diag.shape_changed}",
    ]
    for i, line in enumerate(info_lines):
        draw.text((4, 4 + i * 14), line, fill=(255, 255, 255, 255))

    return img


def main():
    data_dir = Path("/home/z/my-project/alphamoo/data")
    out_dir = Path("/home/z/my-project/alphamoo/download/agent_tracker_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading all replays...")
    replays = load_replays_from_dir(data_dir)

    print(f"\n{'Game':<8} {'Acts':>5} {'Detected':>9} {'Failed':>7} {'ColorChg':>9} {'ShapeChg':>9} {'ClickMode':>10}")
    print("-" * 65)

    total_detected = 0
    total_failed = 0
    total_color_chg = 0
    total_shape_chg = 0

    interesting_moments = []  # (game_id, record_idx, reason, state, diag)

    # Cache per-game results so we don't re-run track_replay for visualization
    per_game_results: dict[str, list] = {}

    for game_id, replay in sorted(replays.items()):
        per_step, stats = track_replay(replay)
        per_game_results[game_id] = per_step

        detected = stats["detection_count"]
        failed = stats["failed_detection_count"]
        color_chg = stats["color_change_count"]
        shape_chg = stats["shape_change_count"]
        click_mode = "Y" if stats["click_game_mode"] else "N"

        total_detected += detected
        total_failed += failed
        total_color_chg += color_chg
        total_shape_chg += shape_chg

        detection_pct = detected / max(1, len(replay.records)) * 100
        print(f"{game_id:<8} {replay.n_actions:>5} {detected:>5} ({detection_pct:.0f}%) "
              f"{failed:>7} {color_chg:>9} {shape_chg:>9} {click_mode:>10}")

        # Find first detection and first color change for visualization
        first_detection_idx = None
        first_color_change_idx = None
        first_shape_change_idx = None
        for i, (_state, diag) in enumerate(per_step):
            if first_detection_idx is None and diag.detected and diag.detection_method == "movement":
                first_detection_idx = i
            if first_color_change_idx is None and diag.color_changed:
                first_color_change_idx = i
            if first_shape_change_idx is None and diag.shape_changed:
                first_shape_change_idx = i

        if first_detection_idx is not None:
            state, diag = per_step[first_detection_idx]
            interesting_moments.append((game_id, first_detection_idx, "first_detection", state, diag))
        if first_color_change_idx is not None:
            state, diag = per_step[first_color_change_idx]
            interesting_moments.append((game_id, first_color_change_idx, "first_color_change", state, diag))
        if first_shape_change_idx is not None:
            state, diag = per_step[first_shape_change_idx]
            interesting_moments.append((game_id, first_shape_change_idx, "first_shape_change", state, diag))

    print("-" * 65)
    total_actions = sum(r.n_actions for r in replays.values())
    overall_pct = total_detected / max(1, total_actions) * 100
    print(f"{'TOTAL':<8} {total_actions:>5} {total_detected:>5} ({overall_pct:.0f}%) "
          f"{total_failed:>7} {total_color_chg:>9} {total_shape_chg:>9}")

    print(f"\n=== Saving {len(interesting_moments)} annotated PNGs ===")
    for game_id, rec_idx, reason, state, diag in interesting_moments[:30]:
        replay = replays[game_id]
        record = replay.records[rec_idx]
        img = draw_agent_overlay(record.final_grid, state, diag, scale=10)
        out_path = out_dir / f"{game_id}_{reason}_step{rec_idx}.png"
        img.save(out_path, format="PNG")
        print(f"  Saved {out_path.name}")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
