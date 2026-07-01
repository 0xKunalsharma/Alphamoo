"""
Investigate click-only games to understand their agent structure.

Click-only games (available_actions = [6] or [6,7]):
  - r11l, lp85, s5i5, tn36, vc33 (and su15 is [6,7])

For each, sample frames across the replay and ask:
  1. Is there a stationary object that transforms over time?
  2. Are clicks happening at consistent locations (cursor-like)?
  3. What changes between consecutive frames?
"""
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw

from alphamoo.frame_renderer import grid_to_rgb
from alphamoo.perception import detect_background_color, perceive
from alphamoo.schemas import GRID_SIZE
from alphamoo.vtx_reader import load_replays_from_dir

CLICK_GAMES = ["r11l", "lp85", "s5i5", "tn36", "vc33", "su15"]


def analyze_click_patterns(replay):
    """For a click-only replay, analyze where the clicks happen."""
    click_locations = []
    for record in replay.records:
        if record.action_input.is_click:
            coords = record.action_input.click_coords
            if coords:
                click_locations.append(coords)
    return click_locations


def find_stationary_transforming_objects(replay, n_samples=20):
    """
    Find objects that stay at the same bbox but change color/shape over time.
    These are candidate "agent" objects in click games.
    """
    n_records = len(replay.records)
    if n_records < 2:
        return []

    # Sample frame indices evenly
    sample_indices = list(range(0, n_records, max(1, n_records // n_samples)))
    if len(sample_indices) > n_samples:
        sample_indices = sample_indices[:n_samples]

    # Track objects by their bounding box (identity heuristic)
    bbox_history = defaultdict(list)  # bbox -> [(color, shape_hash, n_cells)]

    for idx in sample_indices:
        record = replay.records[idx]
        grid = np.array(record.final_grid, dtype=np.int8)
        bg = detect_background_color(grid)
        scene = perceive(grid.tolist(), background_color=bg)

        for obj in scene.objects.values():
            # Skip very large objects (likely background/walls)
            if len(obj.cells) > 200:
                continue
            bbox_history[obj.bounding_box].append({
                "color": obj.color,
                "shape": obj.shape_hash,
                "n_cells": len(obj.cells),
                "sample_idx": idx,
            })

    # Score each bbox by how much it changes
    candidates = []
    for bbox, history in bbox_history.items():
        if len(history) < 3:
            continue  # need at least 3 observations

        # Count color changes
        color_changes = sum(1 for i in range(1, len(history))
                            if history[i]["color"] != history[i-1]["color"])
        # Count shape changes
        shape_changes = sum(1 for i in range(1, len(history))
                           if history[i]["shape"] != history[i-1]["shape"])
        # Count size changes
        size_changes = sum(1 for i in range(1, len(history))
                          if history[i]["n_cells"] != history[i-1]["n_cells"])

        total_changes = color_changes + shape_changes + size_changes
        change_rate = total_changes / max(1, len(history) - 1)

        if total_changes > 0:
            candidates.append({
                "bbox": bbox,
                "history_len": len(history),
                "color_changes": color_changes,
                "shape_changes": shape_changes,
                "size_changes": size_changes,
                "total_changes": total_changes,
                "change_rate": change_rate,
                "first_color": history[0]["color"],
                "last_color": history[-1]["color"],
                "avg_n_cells": sum(h["n_cells"] for h in history) / len(history),
            })

    # Sort by total changes (most-transforming first)
    candidates.sort(key=lambda c: c["total_changes"], reverse=True)
    return candidates


def render_click_visualization(replay, game_id, out_dir, n_frames=8):
    """Render a multi-frame PNG showing the click game over time."""
    n_records = len(replay.records)
    sample_indices = list(range(0, n_records, max(1, n_records // n_frames)))[:n_frames]

    scale = 4
    cols = min(4, n_frames)
    rows = (n_frames + cols - 1) // cols
    cell_px = GRID_SIZE * scale
    img = Image.new("RGB", (cols * cell_px, rows * cell_px + 30 * rows), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for i, idx in enumerate(sample_indices):
        record = replay.records[idx]
        grid = np.array(record.final_grid, dtype=np.int8)
        rgb = grid_to_rgb(grid)
        tile = Image.fromarray(rgb, mode="RGB").resize(
            (cell_px, cell_px), Image.NEAREST
        )

        col = i % cols
        row = i // cols
        img.paste(tile, (col * cell_px, row * (cell_px + 30)))

        # Annotate
        action_id = record.action_input.id
        click = record.action_input.click_coords
        state = record.state
        levels = record.levels_completed
        label = f"step {idx} act={action_id} click={click} lvl={levels} {state}"
        draw.text((col * cell_px + 4, row * (cell_px + 30) + cell_px + 4),
                  label, fill=(255, 255, 255))

        # Draw click location if present
        if click:
            cx, cy = click
            x0 = col * cell_px + cx * scale
            y0 = row * (cell_px + 30) + cy * scale
            draw.rectangle(
                [x0 - 2, y0 - 2, x0 + scale + 2, y0 + scale + 2],
                outline=(255, 0, 0), width=2
            )

    out_path = out_dir / f"{game_id}_click_analysis.png"
    img.save(out_path, format="PNG")
    return out_path


def main():
    data_dir = Path("/home/z/my-project/alphamoo/data")
    out_dir = Path("/home/z/my-project/alphamoo/download/click_game_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading all replays...")
    replays = load_replays_from_dir(data_dir)

    for game_id in CLICK_GAMES:
        if game_id not in replays:
            print(f"\n{game_id}: not found")
            continue

        replay = replays[game_id]
        print(f"\n{'='*70}")
        print(f"Game: {game_id}")
        print(f"  Actions: {replay.n_actions}, Available: {replay.available_actions}")

        # Click pattern analysis
        click_locs = analyze_click_patterns(replay)
        print(f"  Total clicks: {len(click_locs)}")
        if click_locs:
            xs = [c[0] for c in click_locs]
            ys = [c[1] for c in click_locs]
            unique_locs = len(set(click_locs))
            print(f"  Click X range: {min(xs)}-{max(xs)}, Y range: {min(ys)}-{max(ys)}")
            print(f"  Unique click locations: {unique_locs}/{len(click_locs)} ({unique_locs/len(click_locs)*100:.0f}%)")
            print(f"  → {'Cursor-like (many unique locations)' if unique_locs > len(click_locs) * 0.5 else 'Targeted (few unique locations)'}")

        # Stationary transforming objects
        candidates = find_stationary_transforming_objects(replay)
        print("\n  Stationary transforming objects (top 5):")
        if candidates:
            for i, c in enumerate(candidates[:5]):
                print(f"    [{i+1}] bbox={c['bbox']} changes={c['total_changes']} "
                      f"(color:{c['color_changes']} shape:{c['shape_changes']} size:{c['size_changes']}) "
                      f"first_color={c['first_color']} last_color={c['last_color']} "
                      f"avg_cells={c['avg_n_cells']:.0f}")
        else:
            print("    (none found)")

        # Render visualization
        out_path = render_click_visualization(replay, game_id, out_dir)
        print(f"  Saved: {out_path.name}")

    print(f"\n{'='*70}")
    print(f"All visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
