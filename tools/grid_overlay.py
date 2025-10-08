#!/usr/bin/env python3
"""
Grid overlay tool - draws a dashed grid within the arena bounds.

Usage: python tools/grid_overlay.py <image_path>

Controls:
- '+' / '=' : increase grid density (more cells, smaller cells) up to 4 steps
- '-'       : decrease grid density (fewer cells, larger cells) up to 4 steps
- 's'       : save overlaid image and grid settings
- 'q'       : quit

Notes:
- The grid is drawn as thin dashed red lines with ~30% transparency.
- The grid fits inside the arena rectangle derived from the corners (if available).
- Cells are near-uniform; rounding is distributed so borders align exactly and interior lines
  may vary by ~1–2 pixels when dimensions do not divide evenly.
- Outputs: image -> output/, settings -> data/
"""

import os
import json
import argparse
from typing import Tuple, Optional, List
import math

import cv2
import numpy as np


def ensure_output_dirs() -> Tuple[str, str, str]:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    output_dir = os.path.join(project_root, 'output')
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    return project_root, output_dir, data_dir


def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=12, alpha=0.3):
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))
    if length == 0:
        return
    ux = dx / length
    uy = dy / length
    overlay = img.copy()
    current = 0.0
    draw_segment = True
    while current < length:
        if draw_segment:
            sx = int(round(x1 + ux * current))
            sy = int(round(y1 + uy * current))
            end = min(current + dash_length, length)
            ex = int(round(x1 + ux * end))
            ey = int(round(y1 + uy * end))
            cv2.line(overlay, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        current += dash_length
        draw_segment = not draw_segment
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def get_divisors(n: int) -> list:
    if n <= 0:
        return [1]
    small, large = [], []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i != n // i:
                large.append(n // i)
        i += 1
    return small + large[::-1]


def common_divisors(a: int, b: int) -> List[int]:
    da = set(get_divisors(max(1, a)))
    db = set(get_divisors(max(1, b)))
    cds = sorted(list(da & db))
    if not cds:
        return [1]
    return cds


def select_nearest(divs: List[int], target: float) -> int:
    best_i = 0
    best_diff = abs(divs[0] - target)
    for i in range(1, len(divs)):
        d = abs(divs[i] - target)
        if d < best_diff:
            best_i = i
            best_diff = d
    return best_i


def render_grid(base_img: np.ndarray, cols: int, rows: int, rect: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    h, w = base_img.shape[:2]
    out = base_img.copy()
    color = (0, 0, 255)  # red
    thickness = 1
    dash = 12
    if rect is None:
        # Full-frame grid
        left, top, right, bottom = 0, 0, w - 1, h - 1
    else:
        left, top, right, bottom = rect
        left = max(0, min(w - 1, left))
        right = max(0, min(w - 1, right))
        top = max(0, min(h - 1, top))
        bottom = max(0, min(h - 1, bottom))

    cols = max(2, int(cols))
    rows = max(2, int(rows))
    span_x = max(1, right - left)
    span_y = max(1, bottom - top)

    # Vertical lines: place at x = left + round(i * span_x / cols)
    for i in range(cols + 1):
        x = left + int(round(i * span_x / cols))
        draw_dashed_line(out, (x, top), (x, bottom), color, thickness, dash, 0.3)

    # Horizontal lines: place at y = top + round(j * span_y / rows)
    for j in range(rows + 1):
        y = top + int(round(j * span_y / rows))
        draw_dashed_line(out, (left, y), (right, y), color, thickness, dash, 0.3)
    return out


def guess_transform_path(project_root: str, image_path: str) -> Optional[str]:
    base = os.path.splitext(os.path.basename(image_path))[0]
    if base.endswith('_rectified_oriented'):
        base = base[: -len('_rectified_oriented')]
    candidate = os.path.join(project_root, 'data', f"{base}_transform.json")
    return candidate if os.path.exists(candidate) else None


def load_arena_rect_from_transform(transform_json_path: str, image_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    try:
        with open(transform_json_path, 'r') as f:
            data = json.load(f)
        W = int(data['output_size']['width'])
        H = int(data['output_size']['height'])
        min_x = float(data['warp_bounds']['min_x'])
        min_y = float(data['warp_bounds']['min_y'])
        # Rectangle in canvas coordinates
        left = int(round(0 - min_x))
        top = int(round(0 - min_y))
        right = left + W - 1
        bottom = top + H - 1
        # Clamp to image bounds
        h, w = image_shape
        left = max(0, min(w - 1, left))
        right = max(0, min(w - 1, right))
        top = max(0, min(h - 1, top))
        bottom = max(0, min(h - 1, bottom))
        return left, top, right, bottom
    except Exception:
        return None


def run(image_path: str) -> int:
    project_root, output_dir, data_dir = ensure_output_dirs()

    base = cv2.imread(image_path)
    if base is None:
        print(f"Error: could not load image: {image_path}")
        return 1

    h, w = base.shape[:2]

    # Determine arena rectangle bounds if transform is provided or can be guessed
    rect_bounds: Optional[Tuple[int, int, int, int]] = None
    tpath = guess_transform_path(project_root, image_path)
    if tpath:
        rect_bounds = load_arena_rect_from_transform(tpath, (h, w))

    # Compute automatic perfect spacings that fit exactly inside bounds
    if rect_bounds:
        left, top, right, bottom = rect_bounds
        span_x = max(1, right - left)
        span_y = max(1, bottom - top)
    else:
        left, top, right, bottom = 0, 0, w - 1, h - 1
        span_x = max(1, right - left)
        span_y = max(1, bottom - top)

    # Base grid density: aim ~5 cells along the smaller axis, respect min cell size (>=8px)
    MIN_CELL_PX = 8
    small_span = min(span_x, span_y)
    max_small_cells = max(2, small_span // MIN_CELL_PX)
    base_small_cells = min(5, max_small_cells)
    offset = 0  # steps from base, limited to [-4, +4]

    def compute_counts(current_offset: int) -> Tuple[int, int, int]:
        # number of cells along smaller dimension
        cells_small = max(2, min(max_small_cells, base_small_cells + current_offset))
        # approximate cell size in pixels along smaller dimension
        s = max(1.0, small_span / float(cells_small))
        cols = max(2, int(round(span_x / s)))
        rows = max(2, int(round(span_y / s)))
        return cells_small, cols, rows

    cells_small, cols, rows = compute_counts(offset)

    window = "Grid Overlay"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, min(1600, w), min(900, h))

    # Track mouse for crosshair overlay
    mouse_xy = [w // 2, h // 2]

    def on_mouse(event, mx, my, _flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            mouse_xy[0] = mx
            mouse_xy[1] = my

    cv2.setMouseCallback(window, on_mouse)

    while True:
        # Exit cleanly if the window X is pressed (check BEFORE imshow to avoid re-creating the window)
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        overlaid = render_grid(base, cols, rows, rect_bounds)
        # Draw crosshair to ensure a consistent cursor visual
        cx, cy = int(mouse_xy[0]), int(mouse_xy[1])
        h_img, w_img = overlaid.shape[:2]
        cx = max(0, min(w_img - 1, cx))
        cy = max(0, min(h_img - 1, cy))
        size = 10
        color = (255, 255, 255)
        thickness = 1
        cv2.line(overlaid, (max(0, cx - size), cy), (min(w_img - 1, cx + size), cy), color, thickness, cv2.LINE_AA)
        cv2.line(overlaid, (cx, max(0, cy - size)), (cx, min(h_img - 1, cy + size)), color, thickness, cv2.LINE_AA)
        cv2.imshow(window, overlaid)
        key = cv2.waitKey(16) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('+'), ord('=')):
            # Step to more cells (smaller cells), up to 4 steps smaller
            if offset > -4:
                proposed = offset - 1
                new_cells_small, new_cols, new_rows = compute_counts(proposed)
                if new_cols >= 2 and new_rows >= 2:
                    offset = proposed
                    cells_small, cols, rows = new_cells_small, new_cols, new_rows
            cw = int(round(span_x / cols))
            rh = int(round(span_y / rows))
            print(f"Grid: ~{cw}x{rh}px cells → {cols}x{rows}")
        elif key == ord('-'):
            # Step to fewer cells (larger cells), up to 4 steps larger
            if offset < 4:
                proposed = offset + 1
                new_cells_small, new_cols, new_rows = compute_counts(proposed)
                if new_cols >= 2 and new_rows >= 2:
                    offset = proposed
                    cells_small, cols, rows = new_cells_small, new_cols, new_rows
            cw = int(round(span_x / cols))
            rh = int(round(span_y / rows))
            print(f"Grid: ~{cw}x{rh}px cells → {cols}x{rows}")
        elif key == ord('s'):
            # Save all candidate grids with cell sizes between 30px and 1/3 of arena size (both axes)
            MIN_SAVE_CELL = 30
            max_cell_allow = int(min(span_x, span_y) // 3)
            if max_cell_allow < MIN_SAVE_CELL:
                print("No grids meet the 30px..(span/3) constraint; saving current grid only.")
                candidates = [(cols, rows)]
            else:
                candidates = []
                seen = set()
                for s in range(MIN_SAVE_CELL, max_cell_allow + 1):
                    c = max(2, int(round(span_x / float(s))))
                    r = max(2, int(round(span_y / float(s))))
                    # enforce at least 3x3 cells due to 1/3 bound
                    if c < 3 or r < 3:
                        continue
                    cw = int(round(span_x / float(c)))
                    rh = int(round(span_y / float(r)))
                    if cw < MIN_SAVE_CELL or rh < MIN_SAVE_CELL:
                        continue
                    if cw > span_x // 3 or rh > span_y // 3:
                        continue
                    key_pair = (c, r)
                    if key_pair not in seen:
                        seen.add(key_pair)
                        candidates.append(key_pair)
                # Ensure current grid included if it meets constraints
                if (cols, rows) not in seen:
                    candidates.insert(0, (cols, rows))

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            bundle = {
                "source_image": image_path if os.path.isabs(image_path) else os.path.relpath(image_path, project_root),
                "arena_bounds": {
                    "left": rect_bounds[0],
                    "top": rect_bounds[1],
                    "right": rect_bounds[2],
                    "bottom": rect_bounds[3]
                } if rect_bounds else None,
                "transform_json": os.path.relpath(tpath, project_root) if tpath and os.path.isabs(tpath) else tpath,
                "grids": []
            }

            saved_images = []
            for (gc, gr) in candidates:
                overlay_i = render_grid(base, gc, gr, rect_bounds)
                img_path = os.path.join(output_dir, f"{base_name}_grid_{gc}x{gr}.png")
                cv2.imwrite(img_path, overlay_i)
                rel_img = os.path.relpath(img_path, project_root)
                cw = int(round(span_x / float(gc)))
                rh = int(round(span_y / float(gr)))
                bundle["grids"].append({
                    "cols": gc,
                    "rows": gr,
                    "cell_size_px": {"x": cw, "y": rh},
                    "image": rel_img
                })
                saved_images.append(rel_img)

            out_json_path = os.path.join(data_dir, f"{base_name}_grids.json")
            with open(out_json_path, 'w') as f:
                json.dump(bundle, f, indent=2)
            print("Saved:")
            for p in saved_images:
                print(f"  {p}")
            print(f"  {os.path.relpath(out_json_path, project_root)}")

    cv2.destroyAllWindows()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Grid overlay tool")
    parser.add_argument(
        "image_or_transform",
        help="Path to rectified image PNG (preferred) OR transform JSON (will resolve rectified image)",
    )
    args = parser.parse_args()

    # Resolve input: allow passing transform JSON to find rectified image
    image_path = args.image_or_transform
    if image_path.lower().endswith(".json"):
        try:
            project_root, _, _ = ensure_output_dirs()
            with open(image_path, "r") as f:
                data = json.load(f)
            rectified_rel = data.get("rectified_image") or data.get("source_image")
            if not rectified_rel:
                print("Error: transform JSON missing 'rectified_image' or 'source_image'")
                return 1
            # Normalize to absolute path relative to project root
            rectified_rel = rectified_rel.replace("\\", "/")
            image_path = rectified_rel if os.path.isabs(rectified_rel) else os.path.join(project_root, rectified_rel)
        except Exception as e:
            print(f"Error: unable to resolve image from JSON: {e}")
            return 1

    try:
        return run(image_path)
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
