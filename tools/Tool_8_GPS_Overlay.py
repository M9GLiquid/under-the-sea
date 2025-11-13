#!/usr/bin/env python3
"""
Tool 8 - GPS Overlay API Creator

Consolidates existing calibration files into a single standalone JSON file
for the GPSOverlay API.

This tool:
1. Loads fisheye calibration data
2. Loads rectification transform data
3. Extracts 45x29 grid configuration
4. Includes real-world calibration if available
5. Creates a single gps_overlay.json file for standalone API usage

Usage:
    python tools/Tool_8_GPS_Overlay.py --output data/gps_overlay.json
"""

import json
import os
import argparse
from typing import Dict, Any
from pathlib import Path


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _to_abs(path: str, project_root: str) -> str:
    """Convert relative path to absolute"""
    norm = path.replace("\\", os.sep).replace("/", os.sep)
    if os.path.isabs(norm):
        return norm
    return os.path.join(project_root, norm)


def load_json_file(path: str) -> Dict[str, Any]:
    """Load and return JSON data"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_calibration_file(project_root: str) -> str | None:
    """Find calibration file if it exists"""
    # Look for calibration file with pattern: GPS-Real_corrected_rectified_oriented_calibration.json
    calib_path = os.path.join(project_root, "data", "GPS-Real_corrected_rectified_oriented_calibration.json")
    if os.path.exists(calib_path):
        return calib_path

    # Also check src/data/data directory
    src_calib_path = os.path.join(project_root, "src", "data", "data", "GPS-Real_calibration.json")
    if os.path.exists(src_calib_path):
        return src_calib_path

    return None


def create_gps_overlay_json(
    fisheye_json: str,
    transform_json: str,
    grids_json: str,
    calibration_json: str | None = None,
    output_path: str = "data/gps_overlay.json",
    project_root: str = None
) -> str:
    """Create consolidated GPSOverlay JSON from existing files"""

    # Load all source data
    fisheye_data = load_json_file(fisheye_json)["fisheye_calibration"]
    transform_data = load_json_file(transform_json)
    grid_data = load_json_file(grids_json)

    # Extract grid configuration from single grid file
    current_grid = grid_data.get("current_grid")
    if not current_grid:
        raise ValueError("Grid JSON missing 'current_grid' configuration. Make sure Tool 4 has been run to create the grid.")

    print(f"Grid configuration: {current_grid['cols']}x{current_grid['rows']} cells, {current_grid['cell_size_px']['x']}x{current_grid['cell_size_px']['y']} px/cell")

    # Load real-world calibration if available
    real_world_data = {}
    if calibration_json and os.path.exists(calibration_json):
        calib_data = load_json_file(calibration_json)
        if "axis_mm_per_pixel" in calib_data:
            real_world_data = {
                "mm_per_pixel_x": calib_data["axis_mm_per_pixel"]["x"],
                "mm_per_pixel_y": calib_data["axis_mm_per_pixel"]["y"],
                "origin_mm": {"x": 0, "y": 0}
            }
            print(f"[OK] Loaded real-world calibration: {calib_data['axis_mm_per_pixel']['x']:.3f} mm/px")

    # Create consolidated overlay data
    overlay_data = {
        "gps_overlay": {
            "camera_matrix": fisheye_data["camera_matrix"],
            "distortion_coeffs": fisheye_data["distortion_coeffs"],
            "calibration_size": fisheye_data["image_size"],
            "server_size": [2048, 1536],  # GPS server resolution
            "homography": transform_data["homography_image_to_world_canvas"],
            "arena_bounds": grid_data["arena_bounds"],
            "grid": current_grid
        }
    }

    # Add real-world calibration if available
    if real_world_data:
        overlay_data["gps_overlay"]["real_world"] = real_world_data

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save consolidated JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(overlay_data, f, indent=2)

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Tool 8 - GPS Overlay API Creator")
    parser.add_argument("--fisheye-json", default="data/GPS-Real_fisheye_calibration.json",
                       help="Path to fisheye calibration JSON")
    parser.add_argument("--transform-json", default="data/GPS-Real_corrected_transform.json",
                       help="Path to rectification transform JSON")
    parser.add_argument("--grids-json", default="data/GPS-Real_grid.json",
                       help="Path to grid configuration JSON (from Tool 4)")
    parser.add_argument("--calibration-json", default=None,
                       help="Path to real-world calibration JSON (optional)")
    parser.add_argument("--output", default="data/gps_overlay.json",
                       help="Output path for consolidated JSON")
    args = parser.parse_args()

    try:
        project_root = _project_root()

        # Convert paths to absolute
        fisheye_path = _to_abs(args.fisheye_json, project_root)
        transform_path = _to_abs(args.transform_json, project_root)
        grids_path = _to_abs(args.grids_json, project_root)

        # Find calibration file if not specified
        calib_path = None
        if not args.calibration_json:
            calib_path = find_calibration_file(project_root)
            if calib_path:
                print(f"Found calibration file: {os.path.relpath(calib_path, project_root)}")
        else:
            calib_path = _to_abs(args.calibration_json, project_root)

        # Create consolidated JSON
        output_path = create_gps_overlay_json(
            fisheye_path, transform_path, grids_path, calib_path, args.output, project_root
        )

        # Reload to get the grid info for display
        with open(output_path, 'r') as f:
            created_data = json.load(f)
        grid_info = created_data["gps_overlay"]["grid"]
        real_world_info = created_data["gps_overlay"].get("real_world")

        print("[SUCCESS] GPSOverlay API data created successfully!")
        print(f"  Output: {output_path}")
        print(f"  Grid: {grid_info['cols']}x{grid_info['rows']} cells ({grid_info['cell_size_px']['x']}x{grid_info['cell_size_px']['y']} px/cell)")

        if real_world_info:
            print(f"  Real-world: Enabled ({real_world_info['mm_per_pixel_x']:.3f} mm/px)")
        else:
            print("  Real-world: Not available (run Tool 6 first)")

        print("\nUsage: Copy gps_overlay.py and gps_overlay.json to your project")
        print("       from gps_overlay import GPSOverlay")
        print("       overlay = GPSOverlay('gps_overlay.json')")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
