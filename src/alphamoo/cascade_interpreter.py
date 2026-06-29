"""
AlphaMoo v4.1 — Cascade Interpreter.

When an agent takes an action, the engine may emit multiple subframes
(observed: 1 to 372 subframes per action). The Cascade Interpreter:

  1. Takes [N][64][64] subframes from one action
  2. Computes pairwise diffs between consecutive subframes
  3. Emits discrete CascadeEvents (appearance, disappearance, move,
     color_change, shape_change, level_transition)
  4. Returns the final steady-state SceneGraph + the event list

This is critical because:
  - v4 assumed 1 frame per turn; reality is variable
  - The hypothesis generator needs discrete events to update priors
  - The world model needs to know WHAT changed, not just the final state

Performance target: <50ms per cascade (most cascades are size 1, the
worst case is level transitions which can be 100+ frames).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .perception import perceive
from .schemas import GRID_SIZE, CascadeEvent, SceneGraph

# =============================================================================
# Diff computation
# =============================================================================

def diff_grids(grid_a: np.ndarray, grid_b: np.ndarray) -> dict:
    """
    Compute the cell-level diff between two grids.

    Returns:
        {
            "cells_changed": int,
            "appearances": {color: [(x, y), ...]},
            "disappearances": {color: [(x, y), ...]},
            "color_changes": [(x, y, old_color, new_color), ...],
        }
    """
    if grid_a.shape != grid_b.shape:
        raise ValueError(f"Shape mismatch: {grid_a.shape} vs {grid_b.shape}")

    diff_mask = grid_a != grid_b
    n_changed = int(diff_mask.sum())

    if n_changed == 0:
        return {
            "cells_changed": 0,
            "appearances": {},
            "disappearances": {},
            "color_changes": [],
        }

    # Get changed coordinates
    ys, xs = np.where(diff_mask)
    old_colors = grid_a[ys, xs]
    new_colors = grid_b[ys, xs]

    appearances: dict[int, list[tuple[int, int]]] = defaultdict(list)
    disappearances: dict[int, list[tuple[int, int]]] = defaultdict(list)
    color_changes: list[tuple[int, int, int, int]] = []

    for i in range(len(ys)):
        x, y = int(xs[i]), int(ys[i])
        old_c = int(old_colors[i])
        new_c = int(new_colors[i])

        if old_c == 0:  # was background
            appearances[new_c].append((x, y))
        elif new_c == 0:  # now background
            disappearances[old_c].append((x, y))
        else:
            color_changes.append((x, y, old_c, new_c))

    return {
        "cells_changed": n_changed,
        "appearances": dict(appearances),
        "disappearances": dict(disappearances),
        "color_changes": color_changes,
    }


# =============================================================================
# Event classification
# =============================================================================

def classify_event(diff: dict, prev_scene: SceneGraph | None,
                   curr_scene: SceneGraph | None,
                   timestep: int) -> list[CascadeEvent]:
    """
    Turn a raw grid diff into a list of discrete CascadeEvents.

    Heuristics:
      - Large coordinated appearance (10+ cells of same color) → "appearance"
      - Large coordinated disappearance → "disappearance"
      - Small moves (cells shift by 1-2 positions) → "move"
      - Color change of existing cells → "color_change"
      - Level transition (huge diff > 1000 cells) → "level_transition"
    """
    events: list[CascadeEvent] = []
    n_changed = diff["cells_changed"]

    if n_changed == 0:
        return events

    # Level transition: massive change
    if n_changed > GRID_SIZE * GRID_SIZE * 0.25:  # >25% of grid
        events.append(CascadeEvent(
            type="level_transition",
            before={"n_cells": n_changed},
            after={"state": "new_level"},
            timestep=timestep,
        ))
        return events  # don't try to break down level transitions

    # Appearances
    for color, cells in diff["appearances"].items():
        if len(cells) >= 3:
            # Determine bounding box
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            events.append(CascadeEvent(
                type="appearance",
                target_color=color,
                before={},
                after={
                    "n_cells": len(cells),
                    "bbox": bbox,
                    "cells_sample": cells[:10],  # cap for memory
                },
                timestep=timestep,
            ))

    # Disappearances
    for color, cells in diff["disappearances"].items():
        if len(cells) >= 3:
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            events.append(CascadeEvent(
                type="disappearance",
                target_color=color,
                before={
                    "n_cells": len(cells),
                    "bbox": bbox,
                    "cells_sample": cells[:10],
                },
                after={},
                timestep=timestep,
            ))

    # Color changes (treat as transformation of existing object)
    if len(diff["color_changes"]) >= 3:
        # Group by (old_color, new_color) pair
        groups: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for x, y, old_c, new_c in diff["color_changes"]:
            groups[(old_c, new_c)].append((x, y))

        for (old_c, new_c), cells in groups.items():
            if len(cells) >= 3:
                events.append(CascadeEvent(
                    type="color_change",
                    target_color=new_c,
                    before={"old_color": old_c, "n_cells": len(cells)},
                    after={"new_color": new_c, "n_cells": len(cells)},
                    timestep=timestep,
                ))

    # Moves: detect by checking if disappeared cells reappear nearby
    if diff["disappearances"] and diff["appearances"]:
        for dis_color, dis_cells in diff["disappearances"].items():
            for app_color, app_cells in diff["appearances"].items():
                if dis_color != app_color:
                    continue
                if len(dis_cells) != len(app_cells):
                    continue
                # Check if the displacement is consistent (within 1-3 cells)
                displacements = [
                    (a[0] - d[0], a[1] - d[1])
                    for d, a in zip(sorted(dis_cells), sorted(app_cells))
                ]
                unique_disps = set(displacements)
                if len(unique_disps) == 1:
                    dx, dy = unique_disps.pop()
                    if 0 < abs(dx) + abs(dy) <= 5:
                        events.append(CascadeEvent(
                            type="move",
                            target_color=dis_color,
                            before={"bbox": _bbox_of(dis_cells)},
                            after={"bbox": _bbox_of(app_cells),
                                   "displacement": (dx, dy)},
                            timestep=timestep,
                        ))

    return events


def _bbox_of(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return (min(xs), min(ys), max(xs), max(ys))


# =============================================================================
# Main entry point
# =============================================================================

def interpret_cascade(subframes: list[list[list[int]]],
                      prev_scene: SceneGraph | None = None,
                      ) -> tuple[SceneGraph, list[CascadeEvent]]:
    """
    Take [N][64][64] subframes from one action, return:
      - The final steady-state scene graph
      - A list of CascadeEvents that occurred during the cascade

    Args:
        subframes: list of 64x64 grids (the `frame` field of a FrameRecord).
        prev_scene: optional SceneGraph from before this action (for context).

    Returns:
        (final_scene, events)
    """
    if not subframes:
        raise ValueError("Empty subframes list")

    # Convert to numpy arrays
    grids = [np.array(sf, dtype=np.int8) for sf in subframes]
    final_grid = grids[-1]

    # Perceive the final scene
    final_scene = perceive(final_grid.tolist())

    # If only one subframe, no cascade to interpret
    if len(grids) == 1:
        # Still emit events vs the previous scene if provided
        if prev_scene is not None and prev_scene.grid is not None:
            prev_grid = np.array(prev_scene.grid, dtype=np.int8)
            diff = diff_grids(prev_grid, final_grid)
            events = classify_event(diff, prev_scene, final_scene, timestep=0)
            return final_scene, events
        return final_scene, []

    # Multi-subframe cascade: diff each consecutive pair
    all_events: list[CascadeEvent] = []
    for i in range(len(grids) - 1):
        diff = diff_grids(grids[i], grids[i + 1])
        events = classify_event(diff, None, None, timestep=i)
        all_events.extend(events)

    return final_scene, all_events


def interpret_record(prev_record_grid: list[list[int]] | None,
                     curr_record) -> tuple[SceneGraph, list[CascadeEvent]]:
    """
    Convenience wrapper: takes the previous record's final grid and the
    current FrameRecord, returns the interpreted scene + events.

    This handles the full case: cascade between subframes of the current
    record, PLUS the diff from the previous record's final grid.
    """
    subframes = curr_record.frame
    prev_scene = None
    if prev_record_grid is not None:
        prev_scene = perceive(prev_record_grid)

    final_scene, cascade_events = interpret_cascade(subframes, prev_scene)

    return final_scene, cascade_events


# =============================================================================
# Diagnostics
# =============================================================================

def cascade_summary(events: list[CascadeEvent]) -> dict:
    """Summarize a list of CascadeEvents for logging/debugging."""
    by_type: dict[str, int] = defaultdict(int)
    for e in events:
        by_type[e.type] += 1
    return {
        "total_events": len(events),
        "by_type": dict(by_type),
        "timesteps_covered": max((e.timestep for e in events), default=0) + 1 if events else 0,
    }
