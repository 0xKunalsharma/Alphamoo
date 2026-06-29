"""Test fixtures shared across the test suite."""
import sys
from pathlib import Path

import pytest

# Make src importable
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture
def sample_grid():
    """A small 8x8 test grid with a few distinct objects.

    Layout:
      0 0 0 0 0 0 0 0
      0 1 1 0 0 2 0 0
      0 1 1 0 0 2 0 0
      0 0 0 0 0 0 0 0
      0 0 3 3 3 0 0 0
      0 0 3 3 3 0 0 0
      0 0 0 0 0 0 0 0
      0 0 0 0 0 0 0 0

    Object 1: 2x2 square of color 1 at (1,1)
    Object 2: 2x1 vertical bar of color 2 at (5,1)
    Object 3: 2x3 rectangle of color 3 at (2,4)
    Background: 0
    """
    grid = [[0] * 8 for _ in range(8)]
    # Object 1
    for y in range(1, 3):
        for x in range(1, 3):
            grid[y][x] = 1
    # Object 2
    for y in range(1, 3):
        grid[y][5] = 2
    # Object 3
    for y in range(4, 6):
        for x in range(2, 5):
            grid[y][x] = 3
    return grid


@pytest.fixture
def small_replay_path():
    """Path to the smallest known replay for fast tests."""
    # r11l is 167 actions, smallest of the 25
    p = Path("/home/z/my-project/alphamoo/data/r11l-7bda9483-2c94-4d34-9657-e6fae7aa82ef.vtx")
    if not p.exists():
        pytest.skip(f"Test data not available: {p}")
    return p
