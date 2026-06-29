"""Unit tests for cascade_interpreter.py."""
import numpy as np
import pytest

from alphamoo.cascade_interpreter import (
    classify_event,
    diff_grids,
    interpret_cascade,
)
from alphamoo.schemas import GRID_SIZE


def make_empty_grid(color=0):
    return np.full((GRID_SIZE, GRID_SIZE), color, dtype=np.int8)


class TestDiffGrids:
    def test_no_change(self):
        a = make_empty_grid(0)
        b = make_empty_grid(0)
        diff = diff_grids(a, b)
        assert diff["cells_changed"] == 0
        assert diff["appearances"] == {}
        assert diff["disappearances"] == {}

    def test_appearance(self):
        a = make_empty_grid(0)
        b = make_empty_grid(0)
        # Add a 3x3 block of color 1 in b
        b[10:13, 10:13] = 1
        diff = diff_grids(a, b)
        assert diff["cells_changed"] == 9
        assert 1 in diff["appearances"]
        assert len(diff["appearances"][1]) == 9

    def test_disappearance(self):
        a = make_empty_grid(0)
        b = make_empty_grid(0)
        # Remove (set to 0) a 3x3 block in a
        a[10:13, 10:13] = 5
        diff = diff_grids(a, b)
        assert diff["cells_changed"] == 9
        assert 5 in diff["disappearances"]
        assert len(diff["disappearances"][5]) == 9

    def test_color_change(self):
        a = make_empty_grid(0)
        b = make_empty_grid(0)
        # Same cells, different colors
        a[10:13, 10:13] = 3
        b[10:13, 10:13] = 7
        diff = diff_grids(a, b)
        assert diff["cells_changed"] == 9
        assert len(diff["color_changes"]) == 9
        # No appearances or disappearances
        assert diff["appearances"] == {}
        assert diff["disappearances"] == {}


class TestClassifyEvent:
    def test_appearance_event(self):
        diff = {
            "cells_changed": 9,
            "appearances": {1: [(10, 10), (10, 11), (10, 12),
                                 (11, 10), (11, 11), (11, 12),
                                 (12, 10), (12, 11), (12, 12)]},
            "disappearances": {},
            "color_changes": [],
        }
        events = classify_event(diff, None, None, timestep=0)
        assert len(events) >= 1
        assert any(e.type == "appearance" for e in events)
        assert events[0].target_color == 1

    def test_level_transition_event(self):
        """A huge change should be classified as level_transition."""
        diff = {
            "cells_changed": GRID_SIZE * GRID_SIZE,  # all 4096 cells
            "appearances": {},
            "disappearances": {},
            "color_changes": [],
        }
        events = classify_event(diff, None, None, timestep=0)
        assert len(events) == 1
        assert events[0].type == "level_transition"


class TestInterpretCascade:
    def test_single_subframe_no_events(self):
        grid = make_empty_grid(0)
        grid[10, 10] = 1  # one cell of color 1
        subframes = [grid.tolist()]
        scene, events = interpret_cascade(subframes)
        assert scene is not None
        # Single subframe vs no prior context → no events
        assert events == []

    def test_multi_subframe_emits_events(self):
        # Frame 1: empty
        f1 = make_empty_grid(0)
        # Frame 2: add 3x3 block of color 1
        f2 = make_empty_grid(0)
        f2[10:13, 10:13] = 1
        subframes = [f1.tolist(), f2.tolist()]
        scene, events = interpret_cascade(subframes)
        # Should detect the appearance
        assert any(e.type == "appearance" for e in events)

    def test_final_scene_is_last_subframe(self):
        f1 = make_empty_grid(0)
        f2 = make_empty_grid(0)
        f2[5, 5] = 3
        f3 = make_empty_grid(0)
        f3[10, 10] = 7
        subframes = [f1.tolist(), f2.tolist(), f3.tolist()]
        scene, _ = interpret_cascade(subframes)
        # Final grid should have color 7 at (10, 10), not color 3 at (5, 5)
        grid = np.array(scene.grid)
        assert grid[10, 10] == 7
        assert grid[5, 5] == 0

    def test_empty_subframes_raises(self):
        with pytest.raises(ValueError, match="Empty subframes"):
            interpret_cascade([])
