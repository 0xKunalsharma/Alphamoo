"""
Test the perception module on real frames from all 25 games.

For each game:
  - Run perceive() on the initial frame
  - Print diagnostic stats
  - Save an annotated PNG showing detected objects (color-coded)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw

from alphamoo.perception import perceive_with_diagnostics
from alphamoo.schemas import GRID_SIZE
from alphamoo.vtx_reader import load_replays_from_dir


def render_annotated(grid, scene_diag, scale=8):
    """Render grid with bounding boxes drawn around detected objects."""
    from alphamoo.frame_renderer import grid_to_rgb
    rgb = grid_to_rgb(grid)
    img = Image.fromarray(rgb, mode="RGB").resize(
        (GRID_SIZE * scale, GRID_SIZE * scale), Image.NEAREST
    )
    draw = ImageDraw.Draw(img, "RGBA")

    for obj in scene_diag["objects_summary"]:
        x0, y0, x1, y1 = obj["bbox"]
        # Draw a contrasting border
        border_color = (255, 0, 0, 180)  # red, semi-transparent
        draw.rectangle(
            [x0 * scale, y0 * scale, (x1 + 1) * scale, (y1 + 1) * scale],
            outline=border_color, width=2
        )
        # Label with object id and color
        label = f"{obj['id']} c{obj['color']} ({obj['topology']})"
        draw.text((x0 * scale + 2, y0 * scale + 2), label, fill=(255, 255, 255, 255))

    return img


def main():
    data_dir = Path("/home/z/my-project/alphamoo/data")
    out_dir = Path("/home/z/my-project/alphamoo/download/perception_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading all replays...")
    replays = load_replays_from_dir(data_dir)
    print(f"Loaded {len(replays)} replays\n")

    print(f"{'Game':<8} {'BG':>3} {'#Obj':>5} {'#Rel':>5} {'Percept ms':>12} {'Topologies':<35}")
    print("-" * 75)

    total_percept_ms = 0.0
    total_objects = 0
    topology_counts = {}

    for game_id, replay in sorted(replays.items()):
        first_grid = replay.records[0].final_grid
        grid_np = np.array(first_grid, dtype=np.int8)

        diag = perceive_with_diagnostics(grid_np)
        scene = diag["scene_graph"]

        # Topology breakdown
        topo_count = {}
        for obj in scene.objects.values():
            topo_count[obj.topology] = topo_count.get(obj.topology, 0) + 1
        for t, c in topo_count.items():
            topology_counts[t] = topology_counts.get(t, 0) + c
        topo_str = ", ".join(f"{t}:{c}" for t, c in sorted(topo_count.items()))

        percept_ms = diag["full_perception_ms"]
        total_percept_ms += percept_ms
        total_objects += len(scene.objects)

        print(f"{game_id:<8} {diag['background_color']:>3} "
              f"{len(scene.objects):>5} {len(scene.edges):>5} "
              f"{percept_ms:>10.2f}ms {topo_str:<35}")

        # Save annotated image for first few games
        if len(scene.objects) > 0:
            img = render_annotated(grid_np, diag, scale=10)
            img.save(out_dir / f"{game_id}_perception.png", format="PNG")

    print("-" * 75)
    print(f"{'TOTAL':<8} {'':>3} {total_objects:>5} {'':>5} {total_percept_ms:>10.2f}ms")
    print(f"\nAvg perception time: {total_percept_ms/len(replays):.2f} ms")
    print(f"Avg objects per frame: {total_objects/len(replays):.1f}")
    print("\nTopology distribution across all games:")
    for topo, count in sorted(topology_counts.items()):
        print(f"  {topo}: {count}")

    # Speed check: can we perceive in <50ms? (need ~1.85s per action budget)
    print("\n=== Speed check ===")
    print(f"Total perception time for 25 initial frames: {total_percept_ms:.2f}ms")
    print(f"Avg per frame: {total_percept_ms/len(replays):.2f}ms")
    print("Budget per action (9hr, 14800 acts): ~1850ms")
    print(f"Perception budget fraction: {(total_percept_ms/len(replays))/1850*100:.2f}%")


if __name__ == "__main__":
    main()
