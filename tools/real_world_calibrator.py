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


def load_grids_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_view_image(manifest: Dict, manifest_dir: Path) -> Path | None:
    grids = manifest.get("grids", [])
    if grids:
        chosen = grids[0]
        return normalise_path(manifest_dir, chosen.get("image"))
    source = manifest.get("source_image") or manifest.get("rectified_image")
    if source:
        return normalise_path(manifest_dir, source)
    return None


def derive_output_path(manifest_path: Path) -> Path:
    base = manifest_path.stem
    if base.endswith("_grids"):
        base = base[: -len("_grids")]
    calibration_name = f"{base}_calibration.json"
    return PROJECT_ROOT / "data" / calibration_name


def prompt_wall_lengths() -> Dict[str, float]:
    print("\nWhich wall am I measuring?")
    print("We'll start at the top-left corner and move clockwise around the arena.")
    print("Type each length in millimetres and press ENTER to move on.\n")

    prompts = [
        ("top", "TL - TR length [mm]"),
        ("right", "TR - BR length [mm]"),
        ("bottom", "BR - BL length [mm]"),
        ("left", "BL - TL length [mm]"),
    ]

    measurements: Dict[str, float] = {}
    for key, label in prompts:
        while True:
            raw = input(f"{label}: ").strip()
            if not raw:
                print("Please enter a value before continuing.")
                continue
            try:
                value = float(raw)
            except ValueError:
                print("Please enter a numeric value (millimetres).")
                continue
            if value <= 0:
                print("Value must be greater than zero.")
                continue
            measurements[key] = value
            break
    return measurements


def run_viewer(
    image_path: Path,
    origin: Tuple[float, float],
    mm_per_pixel_x: float,
    mm_per_pixel_y: float,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Warning: unable to load image for viewer: {image_path}")
        return

    window = "Tool 6 - Calibration Viewer"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, min(1600, img.shape[1]), min(900, img.shape[0]))

    cursor = [int(round(origin[0])), int(round(origin[1]))]

    def on_mouse(event, mx, my, _flags, _param):
        if event == cv2.EVENT_MOUSEMOVE:
            cursor[0] = mx
            cursor[1] = my

    cv2.setMouseCallback(window, on_mouse)
    print("Move the cursor to inspect TL->cursor distance (press 'q' to close).")
    while True:
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        display = img.copy()
        origin_point = (int(round(origin[0])), int(round(origin[1])))
        cursor_point = (int(cursor[0]), int(cursor[1]))
        cv2.circle(display, origin_point, 5, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(display, cursor_point, 5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.line(display, origin_point, cursor_point, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow(window, display)
        key = cv2.waitKey(16) & 0xFF
        dx_px = cursor_point[0] - origin_point[0]
        dy_px = cursor_point[1] - origin_point[1]
        distance_mm = math.hypot(dx_px * mm_per_pixel_x, dy_px * mm_per_pixel_y)
        print(f"\rTL corner -> cursor: {distance_mm:8.2f} mm", end="", flush=True)
        if key == ord("q"):
            break
    print()
    cv2.destroyWindow(window)



def main() -> int:
    parser = argparse.ArgumentParser(description="Tool 6 - pixel-to-real-world calibrator")
    parser.add_argument("grids_manifest", help="Path to Tool 5 *_grids.json output")
    args = parser.parse_args()

    manifest_path = Path(args.grids_manifest)
    if not manifest_path.is_absolute():
        manifest_path = (PROJECT_ROOT / manifest_path).resolve()
    if not manifest_path.exists():
        print(f"Error: manifest not found: {manifest_path}")
        return 1

    manifest = load_grids_manifest(manifest_path)
    arena_bounds = manifest.get("arena_bounds")
    if not arena_bounds:
        print("Error: manifest missing 'arena_bounds' (expected from Tool 5 output)")
        return 1

    corners = bounds_to_corners(arena_bounds)
    pixel_spans = compute_pixel_spans(corners)

    print("Detected corner pixels:")
    for name, point in corners.items():
        print(f"  {name:>12}: ({point[0]:.2f}, {point[1]:.2f})")

    print("\nPixel spans between corners:")
    for name, length in pixel_spans.items():
        print(f"  {name:>12}: {length:.3f} px")

    wall_lengths_mm = prompt_wall_lengths()
    real_lengths_mm = compute_real_lengths(wall_lengths_mm)
    samples = generate_ratio_samples(pixel_spans, real_lengths_mm)

    width_mm = (wall_lengths_mm["top"] + wall_lengths_mm["bottom"]) / 2.0
    height_mm = (wall_lengths_mm["right"] + wall_lengths_mm["left"]) / 2.0
    width_px = (pixel_spans["top"] + pixel_spans["bottom"]) / 2.0
    height_px = (pixel_spans["right"] + pixel_spans["left"]) / 2.0
    mm_per_pixel_x = width_mm / width_px if width_px else float("nan")
    mm_per_pixel_y = height_mm / height_px if height_px else float("nan")
    axis_mm_per_pixel = {"x": mm_per_pixel_x, "y": mm_per_pixel_y}

    rectified_view = choose_view_image(manifest, manifest_path.parent)
    if rectified_view is None:
        print("Warning: unable to resolve a rectified image path from manifest; viewer disabled")
    else:
        print(f"Resolved rectified image: {rectified_view}")

    output_path = derive_output_path(manifest_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    assumptions = {
        "diagonals": "Diagonal lengths derived from mean horizontal/vertical wall lengths",
    }
    payload = build_calibration_payload(
        grids_manifest=to_project_relative(manifest_path),
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

    if rectified_view is not None:
        run_viewer(rectified_view, corners["top_left"], mm_per_pixel_x, mm_per_pixel_y)

    confirm = input("\nPress 's' to save the calibration file, or any other key to cancel: ").strip().lower()
    if confirm != "s":
        print("Calibration not saved.")
        return 0

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Calibration data saved to {output_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
