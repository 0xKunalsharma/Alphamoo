"""
AlphaMoo v4.1 — Frame renderer.

Renders ARC-AGI-3 64x64 grids (ints 0-15) to PNGs using the standard ARC
palette. Useful for:
  - Visual inspection of replays
  - Debugging perception module output
  - Generating training data images for AffordanceNet

The palette is the canonical ARC 16-color RGB set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .schemas import ARC_PALETTE, GRID_SIZE, MAX_COLOR


def grid_to_rgb(grid: list[list[int]] | np.ndarray) -> np.ndarray:
    """
    Convert a 64x64 grid of color indices (0-15) to an RGB uint8 array.

    Returns:
        np.ndarray of shape (64, 64, 3), dtype uint8.
    """
    if isinstance(grid, list):
        grid = np.array(grid, dtype=np.int8)
    assert grid.shape == (GRID_SIZE, GRID_SIZE), f"Expected 64x64, got {grid.shape}"

    rgb = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    for color_idx in range(MAX_COLOR + 1):
        mask = grid == color_idx
        rgb[mask] = ARC_PALETTE[color_idx]
    return rgb


def render_grid(grid: list[list[int]],
                scale: int = 8,
                grid_lines: bool = False) -> Image.Image:
    """
    Render a 64x64 grid as a PIL Image.

    Args:
        grid: 64x64 list of ints 0-15.
        scale: Pixel size of each cell (default 8 → 512x512 image).
        grid_lines: If True, draw thin lines between cells.

    Returns:
        PIL Image (RGB).
    """
    rgb = grid_to_rgb(grid)
    img = Image.fromarray(rgb, mode="RGB")
    if scale != 1:
        img = img.resize((GRID_SIZE * scale, GRID_SIZE * scale),
                         Image.NEAREST)  # NEAREST preserves crisp edges
    if grid_lines:
        _draw_grid_lines(img, scale)
    return img


def _draw_grid_lines(img: Image.Image, scale: int) -> None:
    """Draw faint grid lines on the image (in-place)."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    line_color = (50, 50, 50)
    for i in range(0, GRID_SIZE + 1):
        x = i * scale
        draw.line([(x, 0), (x, GRID_SIZE * scale)], fill=line_color, width=1)
        draw.line([(0, x), (GRID_SIZE * scale, x)], fill=line_color, width=1)


def render_subframes(subframes: list[list[list[int]]],
                     scale: int = 4,
                     cols: int | None = None) -> Image.Image:
    """
    Render multiple subframes (a cascade) as a single tiled image.

    Args:
        subframes: list of 64x64 grids.
        scale: pixel size per cell.
        cols: number of columns in the tile grid. Defaults to sqrt(len).

    Returns:
        PIL Image with all subframes arranged in a grid.
    """
    n = len(subframes)
    if n == 0:
        raise ValueError("No subframes to render")
    if cols is None:
        cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    cell_px = GRID_SIZE * scale
    img_w = cols * cell_px
    img_h = rows * cell_px
    out = Image.new("RGB", (img_w, img_h), (0, 0, 0))

    for i, subframe in enumerate(subframes):
        tile = render_grid(subframe, scale=scale)
        col = i % cols
        row = i // cols
        out.paste(tile, (col * cell_px, row * cell_px))

    return out


def save_frame_png(grid: list[list[int]],
                   path: str | Path,
                   scale: int = 8,
                   grid_lines: bool = False) -> Path:
    """Render a grid and save as PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = render_grid(grid, scale=scale, grid_lines=grid_lines)
    img.save(path, format="PNG")
    return path


def save_cascade_png(subframes: list[list[list[int]]],
                     path: str | Path,
                     scale: int = 4,
                     cols: int | None = None) -> Path:
    """Render a cascade (multiple subframes) and save as a tiled PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = render_subframes(subframes, scale=scale, cols=cols)
    img.save(path, format="PNG")
    return path


def color_distribution(grid: list[list[int]]) -> dict[int, int]:
    """Return {color_idx: cell_count} for the grid."""
    flat = np.array(grid).flatten()
    counts = {i: int((flat == i).sum()) for i in range(MAX_COLOR + 1)}
    return {k: v for k, v in counts.items() if v > 0}


def palette_legend() -> Image.Image:
    """Render a legend image showing all 16 ARC colors with their indices."""
    cell = 64
    cols = 4
    rows = 4
    img = Image.new("RGB", (cols * cell, rows * cell), (0, 0, 0))
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except OSError:
        font = ImageFont.load_default()

    for idx in range(MAX_COLOR + 1):
        col = idx % cols
        row = idx // cols
        x0 = col * cell
        y0 = row * cell
        rgb = ARC_PALETTE[idx]
        draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=rgb)
        # Pick text color for contrast
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        text_color = (0, 0, 0) if lum > 128 else (255, 255, 255)
        text = str(idx)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x0 + (cell - tw) // 2, y0 + (cell - th) // 2 - 4),
                  text, fill=text_color, font=font)

    return img
