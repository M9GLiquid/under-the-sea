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

        # Copy files to api/ folder for easy export
        api_dir = os.path.join(project_root, "api")
        os.makedirs(api_dir, exist_ok=True)
        
        import shutil
        
        # Copy gps_overlay.json
        api_json_path = os.path.join(api_dir, "gps_overlay.json")
        shutil.copy2(output_path, api_json_path)
        print(f"\n[API] Copied JSON to: {os.path.relpath(api_json_path, project_root)}")
        
        # Verify overlay.py exists in api/ folder (it's already there, no need to copy)
        api_py_path = os.path.join(api_dir, "overlay.py")
        
        if not os.path.exists(api_py_path):
            print(f"[ERROR] API file not found: {api_py_path}")
            print(f"[ERROR] Please ensure api/overlay.py exists. It should be created when you first set up the API.")
            return 1
        
        print(f"[API] API code verified: {os.path.relpath(api_py_path, project_root)}")
        
        # Copy test file
        test_file_path = os.path.join(api_dir, "test_overlay.py")
        test_content = """#!/usr/bin/env python3
\"\"\"
Test script for GPSOverlay API

This script demonstrates how to use the overlay API and tests basic functionality.
\"\"\"

from overlay import GPSOverlay


def main():
    try:
        # Load the API
        overlay = GPSOverlay()

        print("GPSOverlay API - Standalone Test")
        print("=" * 40)

        # Test with sample GPS coordinates (like from your server)
        test_coords = [
            (50, 50),     # Top-left area
            (1024, 768),  # Center area
            (2000, 1500)  # Bottom-right area
        ]

        for gps_x, gps_y in test_coords:
            print(f"\\nGPS Server ({gps_x}, {gps_y}):")

            # Get rectified coordinates
            x_rect, y_rect = overlay.map_coords(gps_x, gps_y)
            print(f"  -> Rectified: ({x_rect:.1f}, {y_rect:.1f})")

            # Get grid cell
            cell = overlay.get_grid_cell(gps_x, gps_y)
            print(f"  -> Grid Cell: {cell['col']}, {cell['row']} (in bounds: {cell['in_bounds']})")

            # Get real-world coordinates (if available)
            if overlay.real_world_available:
                real_pos = overlay.get_real_coords(gps_x, gps_y)
                print(f"  -> Real World: {real_pos['x_mm']:.1f}mm, {real_pos['y_mm']:.1f}mm")
            else:
                print("  -> Real World: Not available (run Tool 6 first)")
        
        # Show grid info
        print("\\nGrid Configuration:")
        print(f"  Size: {overlay.grid_cols}x{overlay.grid_rows} cells")
        print(f"  Cell Size: {overlay.cell_size_px['x']}x{overlay.cell_size_px['y']} pixels")
        print(f"  Arena: {overlay.arena_bounds}")

        if overlay.real_world_available:
            print(f"  Real-World: {overlay.mm_per_pixel_x:.3f} mm/pixel")
        else:
            print("  Real-World: Not calibrated")

        print("\\n[SUCCESS] GPSOverlay API ready for use!")
        print("Copy overlay.py and gps_overlay.json to your project.")

    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Run Tool 8 first: python tools/Tool_8_GPS_Overlay.py")
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Make sure gps_overlay.json contains valid calibration data.")


if __name__ == "__main__":
    main()
"""
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write(test_content)
        print(f"[API] Created test script: {os.path.relpath(test_file_path, project_root)}")
        
        # Create README in api/ folder
        readme_path = os.path.join(api_dir, "README.md")
        readme_content = """# GPSOverlay API - Standalone Package

This folder contains everything you need to use the GPSOverlay API in your project.

## Files

- `overlay.py` - The API code (no external dependencies)
- `gps_overlay.json` - Calibration data (created by Tool 8)
- `test_overlay.py` - Test script demonstrating API usage

## Quick Start

1. Copy both files to your project directory
2. Import and use:

```python
from overlay import GPSOverlay

# Initialize (auto-detects gps_overlay.json in same directory)
overlay = GPSOverlay()

# Or specify custom path
overlay = GPSOverlay("path/to/gps_overlay.json")
```

## Usage Examples

### Transform GPS Coordinates to Rectified Space

```python
# Transform GPS server coordinates (2048x1536) to rectified canvas coordinates
x_rect, y_rect = overlay.map_coords(1024, 768)
print(f"Rectified position: ({x_rect:.1f}, {y_rect:.1f})")
```

### Get Grid Cell Position

```python
# Get which grid cell a GPS coordinate belongs to
cell = overlay.get_grid_cell(1024, 768)

if cell["in_bounds"]:
    print(f"Robot is in cell ({cell['col']}, {cell['row']})")
    print(f"Cell center: ({cell['center_x']:.1f}, {cell['center_y']:.1f})")
else:
    print("Point is outside arena bounds")
```

### Get Real-World Coordinates (if calibrated)

```python
# Convert GPS coordinates to real-world millimeters
try:
    pos = overlay.get_real_coords(1024, 768)
    print(f"Real position: {pos['x_mm']:.1f}mm, {pos['y_mm']:.1f}mm")
    print(f"Distance from origin: {pos['distance_from_origin_mm']:.1f}mm")
except ValueError:
    print("Real-world calibration not available (run Tool 6 first)")
```

### Get Complete Grid Map

```python
# Get all grid cells as a 2D array
grid_map = overlay.get_grid_map()

# Access specific cell (row, col)
cell = grid_map[5][3]  # Row 5, Column 3
print(f"Cell center: ({cell['center_x']:.1f}, {cell['center_y']:.1f})")

# Iterate through all cells
for row in grid_map:
    for cell in row:
        if cell['x_mm'] > 0:  # Only if real-world calibrated
            print(f"Cell ({cell['col']}, {cell['row']}): {cell['x_mm']:.1f}mm, {cell['y_mm']:.1f}mm")
```

## API Reference

### GPSOverlay Class

#### `__init__(json_path=None)`
Initialize the API by loading calibration data.

- `json_path`: Path to `gps_overlay.json`. If `None`, looks for it in the same directory.

#### `map_coords(x, y) -> (float, float)`
Transform GPS server coordinates to rectified canvas coordinates.

- `x`: GPS server X coordinate (typically 0-2048)
- `y`: GPS server Y coordinate (typically 0-1536)
- Returns: `(x_rect, y_rect)` tuple in rectified space

#### `get_grid_cell(x, y) -> dict`
Get grid cell information for GPS coordinates.

- `x`: GPS server X coordinate
- `y`: GPS server Y coordinate
- Returns: Dictionary with `col`, `row`, `in_bounds`, `center_x`, `center_y`

#### `get_real_coords(x, y) -> dict`
Get real-world coordinates in millimeters (requires Tool 6 calibration).

- `x`: GPS server X coordinate
- `y`: GPS server Y coordinate
- Returns: Dictionary with `x_mm`, `y_mm`, `distance_from_origin_mm`
- Raises: `ValueError` if real-world calibration not available

#### `get_grid_map() -> list`
Get complete grid mapping as 2D array.

- Returns: 2D list (rows x cols) of cell dictionaries

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

## Coordinate Systems

- **GPS Server Space**: Original camera image (typically 2048×1536)
- **Rectified Space**: Top-down view after fisheye correction and perspective transformation
- **Grid Space**: Discrete cell indices (col, row) within arena bounds
- **Real-World Space**: Millimeters from arena top-left corner (if calibrated)

## Notes

- The API automatically handles coordinate transformations
- Grid size is dynamically loaded from calibration data
- Real-world coordinates require Tool 6 calibration (optional)
- All coordinates use (0,0) as top-left origin

## Support

For issues or questions, refer to the main project README or calibration tools.
"""
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"[API] Created README: {os.path.relpath(readme_path, project_root)}")
        
        print("\n" + "="*60)
        print("API Package Ready!")
        print("="*60)
        print(f"Copy the entire 'api/' folder to your project:")
        print(f"  - api/overlay.py")
        print(f"  - api/gps_overlay.json")
        print(f"  - api/README.md")
        print(f"  - api/test_overlay.py (optional test script)")
        print("\nThen use:")
        print("  from overlay import GPSOverlay")
        print("  overlay = GPSOverlay()")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
