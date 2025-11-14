#!/usr/bin/env python3
"""Tool 6 - compute pixel-to-real-world calibration from Tool 5 outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

# Ensure project root is on the import path so we can access src.calibration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import (  # noqa: E402
    build_calibration_payload,
    bounds_to_corners,
    compute_pixel_spans,
    compute_real_lengths,
    generate_ratio_samples,
)


def to_project_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def normalise_path(base: Path, raw: str | None) -> Path | None:
    if not raw:
        return None
    cleaned = raw.replace("\\", os.sep)
    candidate = Path(cleaned)
    if candidate.is_absolute():
        return candidate

    # Prefer resolution relative to the manifest directory, but fall back to project root.
    base_candidate = (base / cleaned).resolve()
    if base_candidate.exists():
        return base_candidate

    project_candidate = (PROJECT_ROOT / cleaned.lstrip(os.sep)).resolve()
    if project_candidate.exists():
        return project_candidate

    # As a last resort, return the path relative to project root (even if it does not yet exist).
    return project_candidate




def choose_view_image(grid_data: Dict, grid_dir: Path) -> Path | None:
    source = grid_data.get("source_image") or grid_data.get("rectified_image")
    if source:
        return normalise_path(grid_dir, source)
    return None


def derive_output_path(grid_path: Path) -> Path:
    base = grid_path.stem
    # Remove _grid_{cols}x{rows} suffix if present
    import re
    base = re.sub(r"_grid_\d+x\d+$", "", base)
    calibration_name = f"{base}_calibration.json"
    return PROJECT_ROOT / "data" / calibration_name


def load_wall_lengths_from_calibration(grid_path: Path) -> Dict[str, float] | None:
    """Try to load wall lengths from an existing calibration file."""
    # Try to find existing calibration file
    base = grid_path.stem
    import re
    base = re.sub(r"_grid_\d+x\d+$", "", base)
    calibration_path = PROJECT_ROOT / "data" / f"{base}_calibration.json"
    
    if calibration_path.exists():
        try:
            with open(calibration_path, 'r') as f:
                cal_data = json.load(f)
            wall_lengths = cal_data.get("wall_lengths_mm")
            if wall_lengths and all(k in wall_lengths for k in ["top", "right", "bottom", "left"]):
                print(f"✓ Loaded wall lengths from existing calibration: {calibration_path.name}")
                return wall_lengths
        except Exception as e:
            print(f"Warning: Could not load wall lengths from {calibration_path.name}: {e}")
    
    return None


def get_wall_lengths(grid_path: Path) -> Dict[str, float]:
    """Get wall lengths - try to load from existing calibration, otherwise use standard values."""
    # Try to load from existing calibration file
    wall_lengths = load_wall_lengths_from_calibration(grid_path)
    
    if wall_lengths:
        return wall_lengths
    
    # Use standard wall lengths (these are always the same)
    print("Using standard wall lengths:")
    standard_lengths = {
        "top": 3628.0,
        "right": 2408.0,
        "bottom": 3628.0,
        "left": 2408.0
    }
    for key, value in standard_lengths.items():
        print(f"  {key:>6}: {value:.1f} mm")
    return standard_lengths


def run_viewer(
    image_path: Path,
    origin: Tuple[float, float],
    mm_per_pixel_x: float,
    mm_per_pixel_y: float,
) -> int:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: unable to load image for viewer: {image_path}")
        return 0  # Continue without viewer

    window = "Tool 6 - Calibration Viewer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, min(1600, img.shape[1]), min(900, img.shape[0]))

    cursor = [int(round(origin[0])), int(round(origin[1]))]

    def on_mouse(event, mx, my, _flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            # Adjust for header offset (header is 80px tall)
            header_height = 80
            if my >= header_height:
                cursor[0] = mx
                cursor[1] = my - header_height
            # Mouse in header area, don't update cursor position

    cv2.setMouseCallback(window, on_mouse)
    print("Move the cursor to inspect TL->cursor distance (press ENTER/SPACE to continue, 'q' to quit pipeline).")
    
    def draw_header_overlay(display_img: np.ndarray) -> np.ndarray:
        """Draw header above the image (not overlaying it)."""
        h_img, w_img = display_img.shape[:2]
        header_height = 80
        
        # Create a new image with header space above
        canvas = np.zeros((h_img + header_height, w_img, 3), dtype=np.uint8)
        
        # Draw header bar
        cv2.rectangle(canvas, (0, 0), (w_img, header_height), (40, 40, 40), -1)
        cv2.rectangle(canvas, (0, header_height - 1), (w_img, header_height), (80, 80, 80), 1)
        
        # Status text
        status_text = "Real-World Calibration Viewer | Move cursor to see distance in mm"
        instruction_text = "ENTER/SPACE to continue | 'q' to quit pipeline"
        
        # Draw status text in header
        cv2.putText(canvas, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, instruction_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Place original image below header
        canvas[header_height:, :] = display_img
        
        return canvas
    
    while True:
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        display = img.copy()
        origin_point = (int(round(origin[0])), int(round(origin[1])))
        cursor_point = (int(cursor[0]), int(cursor[1]))
        cv2.circle(display, origin_point, 5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(display, cursor_point, 5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(display, origin_point, cursor_point, (0, 255, 0), 1, cv2.LINE_AA)
        # Draw header above the image (not overlaying it)
        display_with_header = draw_header_overlay(display)
        cv2.imshow(window, display_with_header)
        key = cv2.waitKey(16) & 0xFF
        dx_px = cursor_point[0] - origin_point[0]
        dy_px = cursor_point[1] - origin_point[1]
        distance_mm = math.hypot(dx_px * mm_per_pixel_x, dy_px * mm_per_pixel_y)
        print(f"\rTL corner -> cursor: {distance_mm:8.2f} mm", end="", flush=True)
        
        # Check for ENTER (13) or SPACEBAR (32) to proceed
        if key == 13 or key == 32:  # ENTER or SPACEBAR
            print()
            break
        
        # Check for Q to quit entire pipeline
        if key == ord("q"):
            print()
            cv2.destroyWindow(window)
            return 2  # Special exit code to stop pipeline
    
    print()
    cv2.destroyWindow(window)
    return 0  # Normal completion



def main() -> int:
    parser = argparse.ArgumentParser(description="Tool 6 - pixel-to-real-world calibrator")
    parser.add_argument("grid_json", help="Path to grid JSON output from Tool 4 (e.g., data/GPS-Real_grid.json)")
    args = parser.parse_args()

    grid_path = Path(args.grid_json)
    if not grid_path.is_absolute():
        grid_path = (PROJECT_ROOT / grid_path).resolve()
    if not grid_path.exists():
        print(f"Error: grid JSON not found: {grid_path}")
        return 1

    with open(grid_path, 'r') as f:
        grid_data = json.load(f)

    # Extract grid configuration and arena bounds
    current_grid = grid_data.get("current_grid")
    arena_bounds = grid_data.get("arena_bounds")

    if not arena_bounds:
        print("Error: grid JSON missing 'arena_bounds' (expected from Tool 4 output)")
        return 1

    if not current_grid:
        print("Error: grid JSON missing 'current_grid' configuration")
        return 1

    cols = current_grid["cols"]
    rows = current_grid["rows"]
    cell_size_px = current_grid.get("cell_size_px", {"x": 30, "y": 30})

    print(f"Loaded grid configuration: {cols}x{rows} cells, {cell_size_px['x']}x{cell_size_px['y']} px/cell")

    corners = bounds_to_corners(arena_bounds)
    pixel_spans = compute_pixel_spans(corners)

    print("Detected corner pixels:")
    for name, point in corners.items():
        print(f"  {name:>12}: ({point[0]:.2f}, {point[1]:.2f})")

    print("\nPixel spans between corners:")
    for name, length in pixel_spans.items():
        print(f"  {name:>12}: {length:.3f} px")

    print("\n" + "="*60)
    wall_lengths_mm = get_wall_lengths(grid_path)
    print("="*60)
    real_lengths_mm = compute_real_lengths(wall_lengths_mm)
    samples = generate_ratio_samples(pixel_spans, real_lengths_mm)

    width_mm = (wall_lengths_mm["top"] + wall_lengths_mm["bottom"]) / 2.0
    height_mm = (wall_lengths_mm["right"] + wall_lengths_mm["left"]) / 2.0
    width_px = (pixel_spans["top"] + pixel_spans["bottom"]) / 2.0
    height_px = (pixel_spans["right"] + pixel_spans["left"]) / 2.0
    mm_per_pixel_x = width_mm / width_px if width_px else float("nan")
    mm_per_pixel_y = height_mm / height_px if height_px else float("nan")
    axis_mm_per_pixel = {"x": mm_per_pixel_x, "y": mm_per_pixel_y}

    rectified_view = choose_view_image(grid_data, grid_path.parent)
    if rectified_view is None:
        print("Warning: unable to resolve a rectified image path from grid data; viewer disabled")
    else:
        print(f"Resolved rectified image: {rectified_view}")

    output_path = derive_output_path(grid_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assumptions = {
        "diagonals": "Diagonal lengths derived from mean horizontal/vertical wall lengths",
    }
    payload = build_calibration_payload(
        grids_manifest=to_project_relative(grid_path),
        rectified_image=to_project_relative(rectified_view),
        arena_bounds=arena_bounds,
        corner_pixels=corners,
        pixel_spans=pixel_spans,
        samples=samples,
        wall_lengths_mm=wall_lengths_mm,
        real_lengths_mm=real_lengths_mm,
        axis_mm_per_pixel=axis_mm_per_pixel,
        timestamp=datetime.now(timezone.utc),
        assumptions=assumptions,
    )

    print("\nCalibration summary:")
    for sample in samples:
        print(f"  {sample.name:>12}: {sample.cm_per_pixel:.6f} cm/px")
    stats = payload["cm_per_pixel_stats"]
    print(
        f"\n  Mean cm/px: {stats['mean']:.6f}\n"
        f"  Min cm/px : {stats['min']:.6f}\n"
        f"  Max cm/px : {stats['max']:.6f}\n"
        f"  Std dev   : {stats['stddev']:.6f}"
    )

    print(
        f"\n  Horizontal: {mm_per_pixel_x:0.3f} mm/px ({mm_per_pixel_x/10.0:0.3f} cm/px)"
        f"\n  Vertical  : {mm_per_pixel_y:0.3f} mm/px ({mm_per_pixel_y/10.0:0.3f} cm/px)"
    )

    # Auto-save calibration (no viewer, no prompt)
    print("\n✓ Calibration computed successfully")
    
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"✓ Calibration data saved to {output_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
