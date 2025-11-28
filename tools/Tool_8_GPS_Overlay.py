#!/usr/bin/env python3
"""
Tool 8 - GPS Overlay API Creator

Consolidates existing calibration files into a single gps_overlay.json for the overlay API.
Outputs to data/gps_overlay.json by default and reminds how to export the API.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _to_abs(path: str, project_root: Path) -> Path:
    norm = path.replace("\\", os.sep).replace("/", os.sep)
    p = Path(norm)
    return p if p.is_absolute() else (project_root / p)


def load_json_file(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def find_calibration_file(project_root: Path) -> Optional[Path]:
    calib_path = project_root / "data" / "GPS-Real_corrected_rectified_oriented_calibration.json"
    return calib_path if calib_path.exists() else None


def create_gps_overlay_json(
    fisheye_json: Path,
    transform_json: Path,
    grids_json: Path,
    calibration_json: Optional[Path],
    output_path: Path,
) -> Path:
    fisheye_data = load_json_file(fisheye_json)["fisheye_calibration"]
    transform_data = load_json_file(transform_json)
    grid_data = load_json_file(grids_json)

    current_grid = grid_data.get("current_grid")
    if not current_grid:
        raise ValueError("Grid JSON missing 'current_grid'. Run Tool 4 to create the grid first.")

    real_world_data: Dict[str, Any] = {}
    if calibration_json and calibration_json.exists():
        calib_data = load_json_file(calibration_json)
        if "axis_mm_per_pixel" in calib_data:
            real_world_data = {
                "mm_per_pixel_x": calib_data["axis_mm_per_pixel"]["x"],
                "mm_per_pixel_y": calib_data["axis_mm_per_pixel"]["y"],
                "origin_mm": {"x": 0, "y": 0},
            }

    margin_pixels = fisheye_data.get("margin_pixels", 200)
    corrected_size = fisheye_data.get("corrected_size")
    if corrected_size is None:
        img_w, img_h = fisheye_data["image_size"]
        corrected_size = [img_w + 2 * margin_pixels, img_h + 2 * margin_pixels]

    overlay_data = {
        "gps_overlay": {
            "camera_matrix": fisheye_data["camera_matrix"],
            "distortion_coeffs": fisheye_data["distortion_coeffs"],
            "calibration_size": fisheye_data["image_size"],
            "server_size": [2048, 1536],
            "margin_pixels": margin_pixels,
            "corrected_size": corrected_size,
            "scale_factor": 0.8,
            "homography": transform_data["homography_image_to_world_canvas"],
            "arena_bounds": grid_data["arena_bounds"],
            "grid": current_grid,
        }
    }

    if real_world_data:
        overlay_data["gps_overlay"]["real_world"] = real_world_data

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(overlay_data, f, indent=2)

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool 8 - GPS Overlay API Creator")
    parser.add_argument("--fisheye-json", default="data/GPS-Real_fisheye_calibration.json")
    parser.add_argument("--transform-json", default="data/GPS-Real_corrected_transform.json")
    parser.add_argument("--grids-json", default="data/GPS-Real_grid.json")
    parser.add_argument("--calibration-json", default=None)
    parser.add_argument("--output", default="data/gps_overlay.json")
    args = parser.parse_args()

    project_root = _project_root()
    fisheye_path = _to_abs(args.fisheye_json, project_root)
    transform_path = _to_abs(args.transform_json, project_root)
    grids_path = _to_abs(args.grids_json, project_root)
    calib_path = _to_abs(args.calibration_json, project_root) if args.calibration_json else find_calibration_file(project_root)

    try:
        output_path = create_gps_overlay_json(fisheye_path, transform_path, grids_path, calib_path, _to_abs(args.output, project_root))
        created_data = load_json_file(output_path)
        grid_info = created_data["gps_overlay"]["grid"]
        real_world_info = created_data["gps_overlay"].get("real_world")

        print("[SUCCESS] GPSOverlay API data created successfully!")
        print(f"  Output: {output_path}")
        print(f"  Grid: {grid_info['cols']}x{grid_info['rows']} cells ({grid_info['cell_size_px']['x']}x{grid_info['cell_size_px']['y']} px/cell)")
        if real_world_info:
            print(f"  Real-world: Enabled ({real_world_info['mm_per_pixel_x']:.3f} mm/px)")
        else:
            print("  Real-world: Not available (run Tool 6 first)")

        print("\n[EXPORT] To use the API elsewhere, copy:")
        print("  - modules/overlay-api.py")
        print("  - data/gps_overlay.json")
        print("Optional: include tests/test_overlay.py for quick verification.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Failed to create gps_overlay.json: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
