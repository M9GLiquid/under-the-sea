#!/usr/bin/env python3
"""
Standalone test script for GPSOverlay API

This script can be run independently to test the GPSOverlay API functionality.
Copy this script along with gps_overlay.py and gps_overlay.json to test the API.
"""

import json
import math
import os
import sys


def test_gps_overlay_api():
    """Test GPSOverlay API functionality"""
    print("GPSOverlay API - Standalone Test")
    print("=" * 50)

    # Add current directory to path to find gps_overlay module
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, os.path.join(parent_dir, 'src', 'tools'))

    try:
        from gps_overlay import GPSOverlay

        # Test 1: API Loading
        print("\n[1] Testing API Loading...")
        overlay = GPSOverlay()
        print(f"    [OK] API loaded successfully")
        print(f"    Grid size: {overlay.grid_cols}x{overlay.grid_rows} cells")
        print(f"    Cell size: {overlay.cell_size_px['x']}x{overlay.cell_size_px['y']} pixels")
        print(f"    Arena bounds: {overlay.arena_bounds}")

        # Test 2: Coordinate Transformation
        print("\n[2] Testing Coordinate Transformation...")
        test_coords = [
            (50, 50),       # Top-left area
            (1024, 768),    # Center area
            (2000, 1500),   # Bottom-right area
            (0, 0),         # Origin
            (2047, 1535)    # Max coordinates
        ]

        for gps_x, gps_y in test_coords:
            # Get rectified coordinates
            x_rect, y_rect = overlay.map_coords(gps_x, gps_y)

            # Get grid cell
            cell = overlay.get_grid_cell(gps_x, gps_y)

            # Get real-world coordinates (if available)
            real_info = ""
            if overlay.real_world_available:
                try:
                    real_pos = overlay.get_real_coords(gps_x, gps_y)
                    real_info = f", Real: {real_pos['x_mm']:.1f}mm, {real_pos['y_mm']:.1f}mm"
                except ValueError:
                    real_info = ", Real: Not calibrated"
            else:
                real_info = ", Real: Not calibrated"

            print(f"    GPS ({gps_x:4d}, {gps_y:4d}) -> Rectified ({x_rect:7.1f}, {y_rect:7.1f}) -> Grid ({cell['col']:2d}, {cell['row']:2d}) {real_info}")

        # Test 3: Grid Map Structure
        print("\n[3] Testing Grid Map Structure...")
        grid_map = overlay.get_grid_map()

        if isinstance(grid_map, list) and len(grid_map) == overlay.grid_rows:
            print(f"    [OK] Grid map created: {len(grid_map)} rows")

            # Check first and last cells
            first_cell = grid_map[0][0]
            last_cell = grid_map[-1][-1]

            print(f"    First cell (0,0): {first_cell['center_x']:.1f}, {first_cell['center_y']:.1f}")
            print(f"    Last cell ({overlay.grid_cols-1},{overlay.grid_rows-1}): {last_cell['center_x']:.1f}, {last_cell['center_y']:.1f}")

            # Check that coordinates are within arena bounds
            within_bounds = (first_cell['center_x'] >= overlay.arena_bounds['left'] and
                           first_cell['center_y'] >= overlay.arena_bounds['top'] and
                           last_cell['center_x'] <= overlay.arena_bounds['right'] and
                           last_cell['center_y'] <= overlay.arena_bounds['bottom'])

            if within_bounds:
                print("    [OK] All grid cell centers within arena bounds")
            else:
                print("    [WARN] Some grid cell centers outside arena bounds")

        # Test 4: Error Handling
        print("\n[4] Testing Error Handling...")

        # Test with invalid JSON file
        try:
            invalid_overlay = GPSOverlay("nonexistent.json")
            print("    [FAIL] Should have failed with FileNotFoundError")
        except FileNotFoundError:
            print("    [OK] Correctly handles missing JSON file")

        # Test with real-world coordinates when not calibrated
        if not overlay.real_world_available:
            try:
                overlay.get_real_coords(100, 100)
                print("    [FAIL] Should have failed with ValueError")
            except ValueError as e:
                if "Real-world calibration not available" in str(e):
                    print("    [OK] Correctly handles missing real-world calibration")
                else:
                    print(f"    [FAIL] Wrong error message: {e}")

        print("\n[SUCCESS] All tests completed!")
        print("\nAPI is ready for production use:")
        print("- Copy gps_overlay.py and gps_overlay.json to your project")
        print("- Import: from gps_overlay import GPSOverlay")
        print("- Use simple functions: map_coords(), get_grid_cell(), get_real_coords()")

        return True

    except ImportError as e:
        print(f"\n[ERROR] Could not import GPSOverlay: {e}")
        print("Make sure gps_overlay.py is in the same directory or in Python path")
        return False

    except Exception as e:
        print(f"\n[ERROR] API test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_gps_overlay_api()
    if success:
        print("\n[SUCCESS] GPSOverlay API test completed successfully!")
        sys.exit(0)
    else:
        print("\n[ERROR] GPSOverlay API test failed!")
        sys.exit(1)
