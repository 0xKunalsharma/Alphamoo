"""
AlphaMoo v4.1 — Agent State Tracker (Module 16).

Identifies and tracks the agent across timesteps. The agent can:
  - Be composite (multiple colors, like LS20's red+blue)
  - Transform (color, shape, orientation all mutate)
  - Move via directional actions (1-4)
  - Be stationary in click-only games (r11l, lp85, s5i5, vc33, tn36)

Two detection strategies:

  Strategy A — Movement Correlation (for movement games):
    When the agent takes action UP/DOWN/LEFT/RIGHT, the agent cells shift
    by exactly 1 cell in the action's direction. Find the set of cells in
    prev_frame whose color matches the cell one step in the action direction
    in curr_frame. That's the agent.

  Strategy B — Initial Heuristic (for click-only or first-frame):
    In click-only games, the agent either:
      (a) doesn't exist (you're a cursor), or
      (b) exists but doesn't move — identified as the smallest distinctive
          object that changes state across the replay
    For first-frame detection in movement games, defer until the first
    directional action occurs, then propagate backwards.

Once identified, the tracker maintains the AgentState across timesteps,
detecting color/shape/orientation mutations.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .schemas import GRID_SIZE, MAX_COLOR, AgentState

# Action → expected displacement (dx, dy) when the agent moves
# (x grows right, y grows down — top-left origin)
ACTION_DISPLACEMENTS: dict[int, tuple[int, int]] = {
    1: (0, -1),   # UP: y decreases
    2: (0, +1),   # DOWN: y increases
    3: (-1, 0),   # LEFT: x decreases
    4: (+1, 0),   # RIGHT: x increases
}

# Minimum cells for a valid agent detection
MIN_AGENT_CELLS = 3

# Cells that match within this threshold counts as "the agent moved here"
COLOR_MATCH_TOLERANCE = 0  # exact match


# =============================================================================
# Core detection: movement correlation
# =============================================================================

def detect_agent_by_movement(
    prev_grid: np.ndarray,
    curr_grid: np.ndarray,
    action_id: int,
    background_color: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    """
    Detect the agent by finding cells that moved consistently with the action.

    Args:
        prev_grid: 64x64 grid before the action.
        curr_grid: 64x64 grid after the action.
        action_id: 1 (UP), 2 (DOWN), 3 (LEFT), or 4 (RIGHT).
        background_color: treat this color as background (not agent).

    Returns:
        (prev_agent_cells, curr_agent_cells) if detection succeeds, else None.
        Both sets contain (x, y) tuples.
    """
    if action_id not in ACTION_DISPLACEMENTS:
        return None

    dx, dy = ACTION_DISPLACEMENTS[action_id]

    # For each cell (x, y) in prev that's not background:
    #   Check if cell (x+dx, y+dy) in curr has the SAME color.
    # If yes, this cell likely moved with the agent.
    #
    # We do this efficiently with array slicing.
    h, w = prev_grid.shape  # 64, 64

    # Compute the slice ranges for prev and curr
    # We want pairs where prev[y, x] == curr[y+dy, x+dx] (and both != background)
    if dy >= 0:
        prev_y_slice = slice(0, h - dy)
        curr_y_slice = slice(dy, h)
    else:
        prev_y_slice = slice(-dy, h)
        curr_y_slice = slice(0, h + dy)

    if dx >= 0:
        prev_x_slice = slice(0, w - dx)
        curr_x_slice = slice(dx, w)
    else:
        prev_x_slice = slice(-dx, w)
        curr_x_slice = slice(0, w + dx)

    prev_view = prev_grid[prev_y_slice, prev_x_slice]
    curr_view = curr_grid[curr_y_slice, curr_x_slice]

    # Match: same color, both not background
    color_match = (prev_view == curr_view)
    not_bg_prev = (prev_view != background_color)
    not_bg_curr = (curr_view != background_color)
    match_mask = color_match & not_bg_prev & not_bg_curr

    if not match_mask.any():
        return None

    # Get the (x, y) coordinates in the original grid
    ys_prev, xs_prev = np.where(match_mask)
    # Convert slice-local coordinates back to global
    prev_y_offset = prev_y_slice.start if prev_y_slice.start is not None else 0
    prev_x_offset = prev_x_slice.start if prev_x_slice.start is not None else 0
    prev_cells = set()
    curr_cells = set()
    for ly, lx in zip(ys_prev, xs_prev):
        py = int(ly) + prev_y_offset
        px = int(lx) + prev_x_offset
        cy = py + dy
        cx = px + dx
        prev_cells.add((px, py))
        curr_cells.add((cx, cy))

    if len(prev_cells) < MIN_AGENT_CELLS:
        return None

    return prev_cells, curr_cells


def detect_agent_initial(
    grid: np.ndarray,
    background_color: int,
) -> set[tuple[int, int]] | None:
    """
    Initial-frame heuristic for movement games (used when we don't yet
    have a movement to correlate against).

    Strategy: pick the most distinctive multi-color object near the center.
    Falls back to None if no good candidate.
    """
    # Find all non-background objects via CCL on each color
    candidates: list[tuple[int, set[tuple[int, int]]]] = []
    for color in range(MAX_COLOR + 1):
        if color == background_color:
            continue
        mask = (grid == color)
        if not mask.any():
            continue
        structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        labeled, n = ndimage.label(mask, structure=structure)
        for label_id in range(1, n + 1):
            comp_mask = labeled == label_id
            ys, xs = np.where(comp_mask)
            cells = set(zip(xs.tolist(), ys.tolist()))
            if len(cells) >= MIN_AGENT_CELLS:
                # Compute center distance from grid center
                cx = sum(x for x, _ in cells) / len(cells)
                cy = sum(y for _, y in cells) / len(cells)
                center_dist = abs(cx - 32) + abs(cy - 32)
                candidates.append((center_dist, cells, color))

    if not candidates:
        return None

    # Prefer objects closer to center
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


# =============================================================================
# Composite agent detection (multiple colors)
# =============================================================================

def expand_to_composite(
    base_cells: set[tuple[int, int]],
    grid: np.ndarray,
    background_color: int,
    max_extra_colors: int = 3,
) -> set[tuple[int, int]]:
    """
    If the detected agent is part of a composite (multi-color) agent,
    expand to include all adjacent non-background cells that move together.

    Strategy: flood-fill from the base cells, including any non-background
    cell that's 4-adjacent. Stop at background boundaries.
    """
    if not base_cells:
        return base_cells

    expanded = set(base_cells)
    frontier = list(base_cells)
    while frontier:
        x, y = frontier.pop()
        for nx, ny in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                continue
            if (nx, ny) in expanded:
                continue
            color = int(grid[ny, nx])
            if color == background_color:
                continue
            expanded.add((nx, ny))
            frontier.append((nx, ny))

    return expanded


# =============================================================================
# Agent State Tracker class
# =============================================================================

@dataclass
class TrackerDiagnostics:
    """Diagnostic info from one tracking step."""
    detected: bool = False
    detection_method: str = ""  # "movement" | "initial_heuristic" | "propagated" | "click_game"
    n_agent_cells: int = 0
    agent_colors: list[int] = field(default_factory=list)
    bbox: tuple[int, int, int, int] | None = None
    moved: bool = False
    displacement: tuple[int, int] = (0, 0)
    color_changed: bool = False
    shape_changed: bool = False
    notes: str = ""


class AgentStateTracker:
    """
    Maintains AgentState across timesteps.

    Usage:
        tracker = AgentStateTracker()
        for record in replay.records:
            agent_state, diag = tracker.update(record.final_grid,
                                                record.action_input.id,
                                                background_color)
            # agent_state is updated AgentState (or None if not yet detected)
    """

    def __init__(self, background_color: int | None = None):
        self.background_color = background_color
        self.agent_state: AgentState | None = None
        self.prev_grid: np.ndarray | None = None
        self.prev_action_id: int | None = None
        self.click_game_mode: bool = False  # set if all actions are clicks
        self._detection_count: int = 0
        self._failed_detection_count: int = 0
        self._color_change_count: int = 0
        self._shape_change_count: int = 0

    def update(self, grid: list[list[int]] | np.ndarray,
               action_id: int,
               background_color: int | None = None,
               ) -> tuple[AgentState | None, TrackerDiagnostics]:
        """
        Process one timestep. Updates internal state.

        Args:
            grid: 64x64 grid (current frame).
            action_id: the action that produced this frame.
            background_color: optional override; uses stored value if None.

        Returns:
            (agent_state, diagnostics)
        """
        if isinstance(grid, list):
            grid = np.array(grid, dtype=np.int8)
        if background_color is not None:
            self.background_color = background_color
        if self.background_color is None:
            from .perception import detect_background_color
            self.background_color = detect_background_color(grid)

        diag = TrackerDiagnostics()

        # First frame: no previous to compare against
        if self.prev_grid is None:
            self.prev_grid = grid
            self.prev_action_id = action_id
            # Try initial heuristic
            initial_cells = detect_agent_initial(grid, self.background_color)
            if initial_cells is not None:
                self.agent_state = self._build_state_from_cells(initial_cells, grid)
                diag.detected = True
                diag.detection_method = "initial_heuristic"
                diag.n_agent_cells = len(initial_cells)
                diag.agent_colors = list(self.agent_state.color)
                diag.bbox = self._bbox(initial_cells)
                self._detection_count += 1
            return self.agent_state, diag

        # Movement-based detection
        new_state: AgentState | None = None

        if action_id in ACTION_DISPLACEMENTS and self.prev_grid is not None:
            result = detect_agent_by_movement(
                self.prev_grid, grid, action_id, self.background_color
            )
            if result is not None:
                prev_cells, curr_cells = result

                # Expand to composite (multi-color agent)
                curr_cells = expand_to_composite(
                    curr_cells, grid, self.background_color
                )

                new_state = self._build_state_from_cells(curr_cells, grid)

                # Detect changes vs previous state
                if self.agent_state is not None:
                    diag.moved = True
                    diag.displacement = ACTION_DISPLACEMENTS[action_id]
                    if set(new_state.color) != set(self.agent_state.color):
                        diag.color_changed = True
                        self._color_change_count += 1
                    if new_state.shape != self.agent_state.shape and new_state.shape:
                        diag.shape_changed = True
                        self._shape_change_count += 1

                diag.detected = True
                diag.detection_method = "movement"
                diag.n_agent_cells = len(curr_cells)
                diag.agent_colors = list(new_state.color)
                diag.bbox = self._bbox(curr_cells)
                self._detection_count += 1

        # Fallback: click games — agent doesn't move, but might change state
        if new_state is None and self.agent_state is not None:
            # Propagate previous agent position; check for color/shape changes
            # at the same location
            propagated_cells = set(self.agent_state.object_ids_to_cells()) if hasattr(self.agent_state, 'object_ids_to_cells') else set()
            # Actually we need to store the cells; let's use a simpler approach:
            # extract the current grid at the previous agent's bounding box
            if self.agent_state and self.agent_state.color:
                # Find cells in current grid matching previous agent colors
                # within previous bbox
                bbox = self._get_state_bbox(self.agent_state)
                if bbox:
                    x_min, y_min, x_max, y_max = bbox
                    subgrid = grid[y_min:y_max + 1, x_min:x_max + 1]
                    # Find non-background cells in this region
                    non_bg_mask = (subgrid != self.background_color)
                    if non_bg_mask.any():
                        ys, xs = np.where(non_bg_mask)
                        propagated_cells = {
                            (int(x) + x_min, int(y) + y_min)
                            for x, y in zip(xs, ys)
                        }

                if propagated_cells:
                    new_state = self._build_state_from_cells(propagated_cells, grid)
                    if set(new_state.color) != set(self.agent_state.color):
                        diag.color_changed = True
                        self._color_change_count += 1
                    if new_state.shape != self.agent_state.shape and new_state.shape:
                        diag.shape_changed = True
                        self._shape_change_count += 1

                    diag.detected = True
                    diag.detection_method = "propagated"
                    diag.n_agent_cells = len(propagated_cells)
                    diag.agent_colors = list(new_state.color)
                    diag.bbox = self._bbox(propagated_cells)
                    self._detection_count += 1
                    self.click_game_mode = (action_id == 6)

        if new_state is not None:
            self.agent_state = new_state
        else:
            self._failed_detection_count += 1
            diag.notes = "no detection; agent state unchanged"

        # Update history
        self.prev_grid = grid
        self.prev_action_id = action_id

        return self.agent_state, diag

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _build_state_from_cells(
        self,
        cells: set[tuple[int, int]],
        grid: np.ndarray,
    ) -> AgentState:
        """Build an AgentState from a set of (x, y) cells."""
        if not cells:
            return AgentState()

        # Color distribution
        color_counts = Counter(int(grid[y, x]) for x, y in cells)
        color_counts.most_common(1)[0][0]
        all_colors = sorted(color_counts.keys())

        # Position (centroid)
        cx = sum(x for x, _ in cells) / len(cells)
        cy = sum(y for _, y in cells) / len(cells)

        # Bounding box
        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        # Orientation: compute based on bounding box aspect ratio
        # and the distribution of cells. Simple heuristic:
        #   - If bbox is wider than tall → 0 (horizontal, default)
        #   - If bbox is taller than wide → 1 (vertical, rotated 90°)
        #   - Square → check for diagonal patterns
        w = bbox[2] - bbox[0] + 1
        h = bbox[3] - bbox[1] + 1
        if w > h * 1.5:
            orientation = 0  # horizontal
        elif h > w * 1.5:
            orientation = 1  # vertical
        else:
            orientation = 2  # square / other

        # Shape hash
        shape_hash = self._compute_shape_hash(cells)

        return AgentState(
            object_ids=[],  # we're tracking by cells, not by object IDs
            position=(int(cx), int(cy)),
            orientation=orientation,
            shape=shape_hash,
            color=all_colors,
            energy=None,  # TODO: detect energy bar
            inventory=[],
        )

    def _compute_shape_hash(self, cells: set[tuple[int, int]]) -> str:
        """Rotation/translation-invariant hash of the agent's shape."""
        import hashlib
        if not cells:
            return "empty"

        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        # Build binary mask
        w = x_max - x_min + 1
        h = y_max - y_min + 1
        mask = np.zeros((h, w), dtype=np.uint8)
        for x, y in cells:
            mask[y - y_min, x - x_min] = 1

        # Generate 8 variants
        variants = []
        for refl in [mask, np.fliplr(mask)]:
            v = refl.copy()
            variants.append(v)
            for _ in range(3):
                v = np.rot90(v)
                variants.append(v.copy())

        canonical = min(v.tobytes() for v in variants)
        return hashlib.md5(canonical).hexdigest()[:16]

    def _bbox(self, cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]
        return (min(xs), min(ys), max(xs), max(ys))

    def _get_state_bbox(self, state: AgentState) -> tuple[int, int, int, int] | None:
        """Recover bbox from state. We don't store it explicitly, so we
        approximate using position. Best-effort — only useful for click games."""
        if not state.position:
            return None
        cx, cy = state.position
        # Approximate bbox as 5x5 around centroid (rough heuristic)
        return (cx - 2, cy - 2, cx + 2, cy + 2)

    def get_stats(self) -> dict:
        """Return tracking statistics for diagnostics."""
        return {
            "detection_count": self._detection_count,
            "failed_detection_count": self._failed_detection_count,
            "color_change_count": self._color_change_count,
            "shape_change_count": self._shape_change_count,
            "click_game_mode": self.click_game_mode,
        }


# =============================================================================
# Top-level convenience: track an entire replay
# =============================================================================

def track_replay(replay) -> tuple[list[tuple[AgentState | None, TrackerDiagnostics]], dict]:
    """
    Run the AgentStateTracker over an entire replay.

    Returns:
        (per_step_states, summary_stats)
        per_step_states: list of (agent_state, diag) tuples, one per record.
        summary_stats: dict with overall tracking stats.
    """
    import numpy as np

    from .perception import detect_background_color

    tracker = AgentStateTracker()
    per_step: list[tuple[AgentState | None, TrackerDiagnostics]] = []

    for record in replay.records:
        grid = np.array(record.final_grid, dtype=np.int8)
        bg_color = detect_background_color(grid)
        state, diag = tracker.update(
            grid,
            record.action_input.id,
            background_color=bg_color,
        )
        per_step.append((state, diag))

    return per_step, tracker.get_stats()
