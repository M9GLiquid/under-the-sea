#!/usr/bin/env python3
"""
Grid inspector tool - open a saved grid image and display the cell indices under the cursor.

Usage: python tools/grid_inspector.py <grid_image_path>

The tool tries to parse the grid image filename pattern NAME_grid_{cols}x{rows}.png
and auto-detect the corresponding arena transform JSON to get bounds. If the
transform is not found, it assumes the grid covers the full image.

Controls:
- 'q' : quit

Notes:
- Cell indices are shown as (col, row) with origin at the top-left cell of the arena rectangle.
"""

import os
import re
import argparse
from typing import Optional, Tuple

import cv2
import numpy as np
import json


def ensure_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def parse_cols_rows_from_filename(path: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"_grid_(\d+)x(\d+)\.png$", os.path.basename(path))
    if not m:
        return None
    cols = int(m.group(1))
    rows = int(m.group(2))
    return cols, rows


def guess_grids_json_path(project_root: str, grid_image_path: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(grid_image_path))[0]
    # Strip the _grid_{cols}x{rows} suffix
    base = re.sub(r"_grid_\d+x\d+$", "", base)
    candidate = os.path.join(project_root, 'data', f"{base}_grids.json")
    return candidate if os.path.exists(candidate) else None


def load_arena_rect_from_grids_json(grids_json_path: Optional[str], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = image_shape
    if not grids_json_path:
        return 0, 0, w - 1, h - 1
    try:
        with open(grids_json_path, 'r') as f:
            data = json.load(f)
        bounds = data.get('arena_bounds')
        if bounds:
            left = int(bounds['left'])
            top = int(bounds['top'])
            right = int(bounds['right'])
            bottom = int(bounds['bottom'])
            # Clamp to image bounds
            left = max(0, min(w - 1, left))
            right = max(0, min(w - 1, right))
            top = max(0, min(h - 1, top))
            bottom = max(0, min(h - 1, bottom))
            return left, top, right, bottom
    except Exception:
        pass
    return 0, 0, w - 1, h - 1


def main():
    parser = argparse.ArgumentParser(description="Grid inspector tool")
    parser.add_argument("grid_image", help="Path to an output/*_grid_{cols}x{rows}.png image")
    args = parser.parse_args()

    project_root = ensure_project_root()
    img = cv2.imread(args.grid_image)
    if img is None:
        print(f"Error: could not load image: {args.grid_image}")
        return 1

    h, w = img.shape[:2]
    parsed = parse_cols_rows_from_filename(args.grid_image)
    if not parsed:
        print("Error: could not parse cols/rows from filename. Expected *_grid_{cols}x{rows}.png")
        return 1
    cols, rows = parsed

    tpath = guess_grids_json_path(project_root, args.grid_image)
    left, top, right, bottom = load_arena_rect_from_grids_json(tpath, (h, w))
    span_x = max(1, right - left)
    span_y = max(1, bottom - top)

    window = "Grid Inspector"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, min(1600, w), min(900, h))

    # Helper: compute cell index from mouse position
    def compute_cell(px: int, py: int) -> Optional[Tuple[int, int]]:
        if px < left or px > right or py < top or py > bottom:
            return None
        # Map mouse position to cell index using the same rounding as the grid renderer
        # Grid lines are at: left + round(i * span_x / cols) for i in range(cols + 1)
        # So cell boundaries are between lines i and i+1
        fx = (px - left) / float(span_x)
        fy = (py - top) / float(span_y)
        
        # Find which cell the mouse is in by checking against grid line positions
        cx = 0
        for i in range(cols):
            line_x = left + round(i * span_x / cols)
            next_line_x = left + round((i + 1) * span_x / cols)
            if line_x <= px < next_line_x:
                cx = i
                break
            elif px >= next_line_x and i == cols - 1:
                cx = i
                break
        
        cy = 0
        for j in range(rows):
            line_y = top + round(j * span_y / rows)
            next_line_y = top + round((j + 1) * span_y / rows)
            if line_y <= py < next_line_y:
                cy = j
                break
            elif py >= next_line_y and j == rows - 1:
                cy = j
                break
        
        return cx, cy

    while True:
        # Exit if window closed
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        display = img.copy()

        # Show crosshair and label for current mouse
        x, y = None, None
        # Use getMouseCallback state by polling mouse position from events
        # OpenCV doesn't provide direct getMousePos; capture via event callback storing globals
        # We'll set a static variable on the function object for simplicity
        if not hasattr(main, "_mouse_pos"):
            main._mouse_pos = (w // 2, h // 2)

        def on_mouse(event, mx, my, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                main._mouse_pos = (mx, my)

        cv2.setMouseCallback(window, on_mouse)
        mx, my = main._mouse_pos
        cell = compute_cell(mx, my)

        # Draw overlay text
        label = ""
        if cell is not None:
            label = f"Cell ({cell[0]}, {cell[1]})"
        else:
            label = "Outside arena"

        # Semi-transparent banner
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (10 + 280, 50), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, display, 0.6, 0, display)
        cv2.putText(display, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window, display)
        key = cv2.waitKey(16) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
