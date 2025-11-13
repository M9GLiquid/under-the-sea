#!/usr/bin/env python3
"""
Grid inspector tool - open a saved grid image and display the cell indices under the cursor.

Usage: python tools/Tool_5_Grid_Inspector.py <grid_image_path>

The tool loads grid configuration from the corresponding JSON file (e.g., GPS-Real_grid.json).

Controls:
- 'q' : quit

Notes:
- Cell indices are shown as (col, row) with origin at the top-left cell of the arena rectangle.
- Grid configuration is loaded from the JSON file created by Tool 4.
"""

import os
import argparse
from typing import Optional, Tuple

import cv2
import numpy as np
import json


def ensure_project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def guess_grid_json_path(project_root: str, grid_image_path: str) -> Optional[str]:
    # The JSON file has the same name as the image file but with .json extension
    base = os.path.splitext(grid_image_path)[0]
    candidate = f"{base}.json"
    return candidate if os.path.exists(candidate) else None


def load_arena_rect_from_grid_json(grid_json_path: Optional[str], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    h, w = image_shape
    if not grid_json_path:
        return 0, 0, w - 1, h - 1
    try:
        with open(grid_json_path, 'r') as f:
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
    parser.add_argument("grid_image", help="Path to grid image (e.g., output/GPS-Real_grid.png) created by Tool 4")
    args = parser.parse_args()

    project_root = ensure_project_root()
    img = cv2.imread(args.grid_image)
    if img is None:
        print(f"Error: could not load image: {args.grid_image}")
        return 1

    h, w = img.shape[:2]

    # Load grid configuration from JSON file
    tpath = guess_grid_json_path(project_root, args.grid_image)
    if not tpath or not os.path.exists(tpath):
        print(f"Error: Grid JSON file not found: {tpath}")
        print("Make sure Tool 4 has been run to create the grid configuration.")
        print(f"Expected: {os.path.basename(args.grid_image).replace('.png', '.json')}")
        return 1

    try:
        with open(tpath, 'r') as f:
            grid_data = json.load(f)
        current_grid = grid_data.get("current_grid")
        if not current_grid:
            print("Error: Grid JSON missing 'current_grid' configuration.")
            print("The grid JSON file should contain grid configuration from Tool 4.")
            print("Make sure you're using a grid file created by the updated Tool 4.")
            return 1

        cols = current_grid["cols"]
        rows = current_grid["rows"]
        print(f"Loaded grid configuration: {cols}x{rows} from {os.path.relpath(tpath, project_root)}")

    except Exception as e:
        print(f"Error loading grid configuration: {e}")
        return 1

    left, top, right, bottom = load_arena_rect_from_grid_json(tpath, (h, w))
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
