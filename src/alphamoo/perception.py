"""
AlphaMoo v4.1 — Perception module.

Converts raw 64x64 grids (ints 0-15) into symbolic scene graphs.

Pipeline:
  1. Background detection — find the dominant color (treat as background)
  2. Connected Component Labeling (CCL) — group same-color adjacent cells
  3. Feature extraction — bounding box, topology, shape hash
  4. Relation extraction — adjacency, containment, blocking
  5. Scene graph construction

This is the foundation everything else builds on. No ML, pure symbolic.
Uses scipy.ndimage for fast CCL on the 64x64 grid.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np
from scipy import ndimage

from .schemas import MAX_COLOR, GameObject, SceneGraph

# =============================================================================
# Configuration
# =============================================================================

# Minimum object size (in cells) — smaller than this is noise
MIN_OBJECT_SIZE = 2

# Colors that are always treated as background (never objects)
# Color 0 (black) is traditionally background in ARC
DEFAULT_BACKGROUND_COLORS: frozenset[int] = frozenset({0})


# =============================================================================
# Background detection
# =============================================================================

def detect_background_color(grid: np.ndarray,
                             background_candidates: frozenset[int] = DEFAULT_BACKGROUND_COLORS
                             ) -> int:
    """
    Detect the dominant background color of a grid.

    Strategy:
      1. If any of the default background candidates covers >50% of the grid,
         use the largest one.
      2. Otherwise, use whichever single color covers the most cells.

    Returns:
        Color index (0-15) identified as background.
    """
    counts = Counter(grid.flatten())
    total = sum(counts.values())

    # Try default candidates first
    candidate_counts = {c: counts.get(c, 0) for c in background_candidates}
    best_candidate = max(candidate_counts.items(), key=lambda x: x[1])
    if best_candidate[1] > 0.5 * total:
        return best_candidate[0]

    # Fall back to most common color overall
    return counts.most_common(1)[0][0]


# =============================================================================
# Connected Component Labeling
# =============================================================================

def _ccl_for_color(grid: np.ndarray, color: int) -> tuple[np.ndarray, int]:
    """
    Run connected component labeling on the binary mask `grid == color`.

    Connectivity: 8-way (including diagonals). This matches typical ARC
    object structure where diagonal cells can belong to the same object.

    Returns:
        (labeled_array, n_components) — labeled_array has int labels 1..N,
        0 = background.
    """
    mask = (grid == color).astype(np.int32)
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], dtype=np.int32)
    labeled, n = ndimage.label(mask, structure=structure)
    return labeled, n


def extract_objects(grid: list[list[int]] | np.ndarray,
                    background_color: int | None = None,
                    min_size: int = MIN_OBJECT_SIZE,
                    ) -> list[GameObject]:
    """
    Extract all objects from a grid via CCL.

    Args:
        grid: 2D grid of color indices. Works with any square size (tests use 8x8,
              real games use 64x64).
        background_color: If None, auto-detect. Otherwise treat this color
            as background (skip CCL on it).
        min_size: minimum cell count for an object.

    Returns:
        List of GameObject instances.
    """
    if isinstance(grid, list):
        grid = np.array(grid, dtype=np.int8)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError(f"Expected square 2D grid, got shape {grid.shape}")

    if background_color is None:
        background_color = detect_background_color(grid)

    objects: list[GameObject] = []
    obj_counter = 0

    for color in range(MAX_COLOR + 1):
        if color == background_color:
            continue
        if not (grid == color).any():
            continue

        labeled, n_components = _ccl_for_color(grid, color)
        if n_components == 0:
            continue

        for label_id in range(1, n_components + 1):
            mask = labeled == label_id
            n_cells = int(mask.sum())
            if n_cells < min_size:
                continue

            obj_counter += 1
            obj = _build_object(
                obj_id=f"obj_{obj_counter:03d}",
                color=color,
                mask=mask,
                grid=grid,
            )
            objects.append(obj)

    return objects


def _build_object(obj_id: str, color: int, mask: np.ndarray,
                  grid: np.ndarray) -> GameObject:
    """Build a GameObject from a boolean mask + the source grid."""
    ys, xs = np.where(mask)
    cells = list(zip(xs.tolist(), ys.tolist()))
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    # Topology
    topology = _classify_topology(mask)

    # Shape hash — rotation/translation invariant
    shape_hash = _compute_shape_hash(mask)

    # Secondary colors (within bounding box but outside this object's cells)
    bbox_region = grid[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
    bbox_mask = mask[bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]
    secondary = []
    if bbox_region.size > bbox_mask.sum():
        other_colors = bbox_region[~bbox_mask]
        if len(other_colors) > 0:
            other_counts = Counter(other_colors.tolist())
            # Top 3 secondary colors
            secondary = [c for c, _ in other_counts.most_common(3)]

    return GameObject(
        id=obj_id,
        color=int(color),
        secondary_colors=secondary,
        cells=cells,
        bounding_box=bbox,
        topology=topology,
        shape_hash=shape_hash,
        is_agent=False,
    )


def _classify_topology(mask: np.ndarray) -> str:
    """
    Classify an object's topology as solid, hollow, or fragmented.

    - solid: filled (no holes inside the bounding box)
    - hollow: has interior holes (like a ring)
    - fragmented: multiple disconnected pieces within the same CCL label
      (shouldn't happen with our CCL setup, but kept for safety)
    """
    # Use scipy to find holes: invert the mask, run CCL on the inverted,
    # if any component is fully surrounded by the mask, it's a hole.
    inverted = ~mask
    structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    inv_labeled, inv_n = ndimage.label(inverted, structure=structure)

    if inv_n <= 1:
        # Only one "background" component (outside the object) — solid
        return "solid"

    # Check if any inverted component is enclosed (not touching the border)
    border_labels = set()
    border_labels.update(inv_labeled[0, :].tolist())
    border_labels.update(inv_labeled[-1, :].tolist())
    border_labels.update(inv_labeled[:, 0].tolist())
    border_labels.update(inv_labeled[:, -1].tolist())
    border_labels.discard(0)

    for label_id in range(1, inv_n + 1):
        if label_id not in border_labels:
            return "hollow"

    return "solid"


def _compute_shape_hash(mask: np.ndarray) -> str:
    """
    Compute a rotation/translation-invariant hash of the object's shape.

    Method:
      1. Crop to the bounding box (translation invariance)
      2. Generate all 4 rotations + their mirrors (8 variants)
      3. Sort the variants lexicographically
      4. Hash the smallest variant (rotation/reflection invariance)

    Returns:
        16-char hex string.
    """
    # Crop to bounding box
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return "empty"
    cropped = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    arr = cropped.astype(np.uint8)

    # Generate 8 variants (4 rotations × 2 reflections)
    variants = []
    for refl in [arr, np.fliplr(arr)]:
        v = refl.copy()
        variants.append(v)
        for _ in range(3):
            v = np.rot90(v)
            variants.append(v.copy())

    # Sort by flattened bytes and take the smallest (canonical form)
    variant_bytes = [v.tobytes() for v in variants]
    canonical = min(variant_bytes)
    return hashlib.md5(canonical).hexdigest()[:16]


# =============================================================================
# Relation extraction
# =============================================================================

def extract_relations(objects: list[GameObject]) -> set[tuple[str, str, str]]:
    """
    Extract typed relational edges between objects.

    Relation types:
      - "adjacent_to": bounding boxes touch or overlap by ≤1 cell
      - "inside": one object's bounding box is fully inside another's
      - "overlapping": bounding boxes overlap (rare but possible)
      - "same_color": objects share the same primary color
      - "same_shape": objects share the same shape_hash
    """
    relations: set[tuple[str, str, str]] = set()

    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i >= j:
                continue
            # Adjacency (bounding box touching)
            if _bboxes_adjacent(a.bounding_box, b.bounding_box):
                relations.add((a.id, b.id, "adjacent_to"))
                relations.add((b.id, a.id, "adjacent_to"))

            # Containment
            if _bbox_contains(a.bounding_box, b.bounding_box):
                relations.add((b.id, a.id, "inside"))
            elif _bbox_contains(b.bounding_box, a.bounding_box):
                relations.add((a.id, b.id, "inside"))

            # Same color
            if a.color == b.color:
                relations.add((a.id, b.id, "same_color"))
                relations.add((b.id, a.id, "same_color"))

            # Same shape
            if a.shape_hash == b.shape_hash and a.shape_hash != "empty":
                relations.add((a.id, b.id, "same_shape"))
                relations.add((b.id, a.id, "same_shape"))

    return relations


def _bboxes_adjacent(a: tuple[int, int, int, int],
                     b: tuple[int, int, int, int],
                     tolerance: int = 1) -> bool:
    """Check if two bounding boxes are within `tolerance` cells of each other."""
    ax_min, ay_min, ax_max, ay_max = a
    bx_min, by_min, bx_max, by_max = b
    # Check x overlap/gap
    if ax_max + tolerance < bx_min or bx_max + tolerance < ax_min:
        return False
    return not (ay_max + tolerance < by_min or by_max + tolerance < ay_min)


def _bbox_contains(outer: tuple[int, int, int, int],
                   inner: tuple[int, int, int, int]) -> bool:
    """Check if outer bbox fully contains inner bbox."""
    ox_min, oy_min, ox_max, oy_max = outer
    ix_min, iy_min, ix_max, iy_max = inner
    return (ox_min <= ix_min and oy_min <= iy_min and
            ox_max >= ix_max and oy_max >= iy_max)


# =============================================================================
# Scene graph construction
# =============================================================================

def perceive(grid: list[list[int]] | np.ndarray,
             background_color: int | None = None,
             min_object_size: int = MIN_OBJECT_SIZE) -> SceneGraph:
    """
    Full perception pipeline: grid → SceneGraph.

    Args:
        grid: 64x64 grid of color indices (0-15).
        background_color: If None, auto-detect.
        min_object_size: minimum cell count for an object.

    Returns:
        SceneGraph with all detected objects and their relations.
    """
    if isinstance(grid, list):
        grid = np.array(grid, dtype=np.int8)

    if background_color is None:
        background_color = detect_background_color(grid)

    objects_list = extract_objects(grid, background_color, min_object_size)
    relations = extract_relations(objects_list)

    # Build dict
    objects_dict = {obj.id: obj for obj in objects_list}

    # Compute hash for revisit detection
    grid_hash = hashlib.md5(grid.tobytes()).hexdigest()[:16]

    return SceneGraph(
        objects=objects_dict,
        edges=relations,
        agent_id=None,  # set by Agent State Tracker (Phase 1, later)
        grid=grid.tolist(),
        hash=grid_hash,
    )


def perceive_with_diagnostics(grid: list[list[int]] | np.ndarray) -> dict:
    """
    Run perceive() and return diagnostic info alongside the SceneGraph.

    Useful for debugging and for Phase 0 measurement.
    """
    if isinstance(grid, list):
        grid = np.array(grid, dtype=np.int8)

    import time
    t0 = time.perf_counter()
    bg_color = detect_background_color(grid)
    t1 = time.perf_counter()
    scene = perceive(grid, background_color=bg_color)
    t2 = time.perf_counter()

    return {
        "scene_graph": scene,
        "background_color": bg_color,
        "n_objects": len(scene.objects),
        "n_relations": len(scene.edges),
        "bg_detection_ms": (t1 - t0) * 1000,
        "full_perception_ms": (t2 - t0) * 1000,
        "objects_summary": [
            {
                "id": obj.id,
                "color": obj.color,
                "n_cells": len(obj.cells),
                "bbox": obj.bounding_box,
                "topology": obj.topology,
                "shape_hash": obj.shape_hash,
                "secondary_colors": obj.secondary_colors,
            }
            for obj in scene.objects.values()
        ],
    }
