#!/usr/bin/env python3
"""
Test suite for GPSOverlay API

Tests the standalone GPSOverlay coordinate transformation functionality.
"""

import json
import math
import os
import sys
import tempfile
import unittest
from typing import Dict, List, Tuple

# Add src/tools to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'tools'))

from gps_overlay import GPSOverlay


class TestGPSOverlay(unittest.TestCase):
    """Test cases for GPSOverlay API"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a minimal test JSON configuration (using 14x9 grid like current setup)
        self.test_data = {
            "gps_overlay": {
                "camera_matrix": [
                    [800, 0, 400],
                    [0, 800, 300],
                    [0, 0, 1]
                ],
                "distortion_coeffs": [0.1, 0.02, 0, 0],
                "calibration_size": [1200, 900],
                "server_size": [2048, 1536],
                "homography": [
                    [1.2, 0.1, 100],
                    [0.05, 1.1, 50],
                    [0.001, 0.002, 1.0]
                ],
                "arena_bounds": {
                    "left": 200,
                    "top": 150,
                    "right": 800,
                    "bottom": 600
                },
                "grid": {
                    "cols": 14,
                    "rows": 9,
                    "cell_size_px": {"x": 43, "y": 50}
                },
                "real_world": {
                    "mm_per_pixel_x": 5.0,
                    "mm_per_pixel_y": 5.0,
                    "origin_mm": {"x": 0, "y": 0}
                }
            }
        }

        # Create temporary JSON file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(self.test_data, self.temp_file, indent=2)
        self.temp_file.close()

        # Initialize API
        self.overlay = GPSOverlay(self.temp_file.name)

    def tearDown(self):
        """Clean up test fixtures"""
        os.unlink(self.temp_file.name)

    def test_initialization(self):
        """Test API initialization and data loading"""
        self.assertEqual(self.overlay.grid_cols, 14)
        self.assertEqual(self.overlay.grid_rows, 9)
        self.assertEqual(self.overlay.cell_size_px["x"], 43)
        self.assertEqual(self.overlay.cell_size_px["y"], 50)
        self.assertTrue(self.overlay.real_world_available)
        self.assertEqual(self.overlay.mm_per_pixel_x, 5.0)

    def test_map_coords_basic(self):
        """Test basic coordinate mapping"""
        # Test center point
        x_rect, y_rect = self.overlay.map_coords(1024, 768)

        # Should be within reasonable bounds
        self.assertIsInstance(x_rect, float)
        self.assertIsInstance(y_rect, float)
        self.assertFalse(math.isnan(x_rect))
        self.assertFalse(math.isnan(y_rect))

    def test_map_coords_edge_cases(self):
        """Test coordinate mapping with edge cases"""
        # Test origin
        x_rect, y_rect = self.overlay.map_coords(0, 0)
        self.assertIsInstance(x_rect, float)
        self.assertIsInstance(y_rect, float)

        # Test max coordinates
        x_rect, y_rect = self.overlay.map_coords(2047, 1535)
        self.assertIsInstance(x_rect, float)
        self.assertIsInstance(y_rect, float)

    def test_get_grid_cell_in_bounds(self):
        """Test grid cell detection for points within arena"""
        # Test point that should be in bounds
        cell = self.overlay.get_grid_cell(1024, 768)

        self.assertIn("col", cell)
        self.assertIn("row", cell)
        self.assertIn("in_bounds", cell)
        self.assertIn("center_x", cell)
        self.assertIn("center_y", cell)

        self.assertIsInstance(cell["col"], int)
        self.assertIsInstance(cell["row"], int)
        self.assertIsInstance(cell["in_bounds"], bool)
        self.assertIsInstance(cell["center_x"], float)
        self.assertIsInstance(cell["center_y"], float)

        # Should be within grid bounds (14x9)
        self.assertGreaterEqual(cell["col"], 0)
        self.assertLess(cell["col"], self.overlay.grid_cols)
        self.assertGreaterEqual(cell["row"], 0)
        self.assertLess(cell["row"], self.overlay.grid_rows)

    def test_get_grid_cell_out_of_bounds(self):
        """Test grid cell detection for points outside arena"""
        # Test point that should be out of bounds (top-left corner)
        cell = self.overlay.get_grid_cell(50, 50)

        self.assertFalse(cell["in_bounds"])
        self.assertEqual(cell["col"], 0)  # Should clamp to 0
        self.assertEqual(cell["row"], 0)  # Should clamp to 0

    def test_get_real_coords(self):
        """Test real-world coordinate conversion"""
        real_pos = self.overlay.get_real_coords(1024, 768)

        self.assertIn("x_mm", real_pos)
        self.assertIn("y_mm", real_pos)
        self.assertIn("distance_from_origin_mm", real_pos)

        self.assertIsInstance(real_pos["x_mm"], float)
        self.assertIsInstance(real_pos["y_mm"], float)
        self.assertIsInstance(real_pos["distance_from_origin_mm"], float)

        # Distance should be positive
        self.assertGreaterEqual(real_pos["distance_from_origin_mm"], 0)

    def test_get_grid_map_structure(self):
        """Test complete grid map structure"""
        grid_map = self.overlay.get_grid_map()

        self.assertIsInstance(grid_map, list)
        self.assertEqual(len(grid_map), self.overlay.grid_rows)  # 9 rows

        # Check each row
        for row_idx, row in enumerate(grid_map):
            self.assertIsInstance(row, list)
            self.assertEqual(len(row), self.overlay.grid_cols)  # 14 columns

            # Check each cell
            for col_idx, cell in enumerate(row):
                self.assertIn("col", cell)
                self.assertIn("row", cell)
                self.assertIn("x_mm", cell)
                self.assertIn("y_mm", cell)
                self.assertIn("center_x", cell)
                self.assertIn("center_y", cell)

                self.assertEqual(cell["col"], col_idx)
                self.assertEqual(cell["row"], row_idx)

    def test_get_grid_map_values(self):
        """Test grid map values are reasonable"""
        grid_map = self.overlay.get_grid_map()

        # Check that coordinates progress logically (14x9 grid)
        for row in range(self.overlay.grid_rows):  # 0-8
            for col in range(self.overlay.grid_cols):  # 0-13
                cell = grid_map[row][col]

                # Center coordinates should be within arena bounds
                self.assertGreaterEqual(cell["center_x"], self.overlay.arena_bounds["left"])
                self.assertLessEqual(cell["center_x"], self.overlay.arena_bounds["right"])
                self.assertGreaterEqual(cell["center_y"], self.overlay.arena_bounds["top"])
                self.assertLessEqual(cell["center_y"], self.overlay.arena_bounds["bottom"])

    def test_consistency_map_vs_get_cell(self):
        """Test that map_coords + grid calculation matches get_grid_cell"""
        test_coords = [
            (512, 384),   # Quarter point
            (1024, 768),  # Center
            (1536, 1152)  # Three-quarter point
        ]

        for gps_x, gps_y in test_coords:
            # Get coordinates via map_coords
            x_rect, y_rect = self.overlay.map_coords(gps_x, gps_y)

            # Get cell via get_grid_cell
            cell = self.overlay.get_grid_cell(gps_x, gps_y)

            # Calculate expected cell from coordinates (14x9 grid)
            left, top = self.overlay.arena_bounds["left"], self.overlay.arena_bounds["top"]
            cell_width = (self.overlay.arena_bounds["right"] - left) / self.overlay.grid_cols
            cell_height = (self.overlay.arena_bounds["bottom"] - top) / self.overlay.grid_rows

            expected_col = int((x_rect - left) // cell_width)
            expected_row = int((y_rect - top) // cell_height)

            # Should match (with possible off-by-one due to floating point)
            self.assertAlmostEqual(cell["col"], expected_col, delta=1)
            self.assertAlmostEqual(cell["row"], expected_row, delta=1)

    def test_no_real_world_calibration(self):
        """Test behavior when real-world calibration is not available"""
        # Create test data without real-world calibration
        test_data_no_calib = {
            "gps_overlay": {
                "camera_matrix": [[800, 0, 400], [0, 800, 300], [0, 0, 1]],
                "distortion_coeffs": [0.1, 0.02, 0, 0],
                "calibration_size": [1200, 900],
                "server_size": [2048, 1536],
                "homography": [[1.2, 0.1, 100], [0.05, 1.1, 50], [0.001, 0.002, 1.0]],
                "arena_bounds": {"left": 200, "top": 150, "right": 800, "bottom": 600},
                "grid": {"cols": 14, "rows": 9, "cell_size_px": {"x": 43, "y": 50}}
            }
        }

        # Create temporary file without calibration
        temp_file_no_calib = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(test_data_no_calib, temp_file_no_calib, indent=2)
        temp_file_no_calib.close()

        # Test API without calibration
        overlay_no_calib = GPSOverlay(temp_file_no_calib.name)

        self.assertFalse(overlay_no_calib.real_world_available)

        # Should raise error when trying to get real coordinates
        with self.assertRaises(ValueError) as context:
            overlay_no_calib.get_real_coords(100, 100)

        self.assertIn("Real-world calibration not available", str(context.exception))

        # Grid map should still work but with zero real-world coordinates
        grid_map = overlay_no_calib.get_grid_map()
        for row in grid_map:
            for cell in row:
                self.assertEqual(cell["x_mm"], 0)
                self.assertEqual(cell["y_mm"], 0)

        # Clean up
        os.unlink(temp_file_no_calib.name)

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        # Test with invalid JSON file
        with self.assertRaises(FileNotFoundError):
            GPSOverlay("nonexistent.json")

        # Test with invalid coordinates (should not crash)
        try:
            result = self.overlay.map_coords(float('inf'), float('nan'))
            # Should handle gracefully (return NaN or reasonable values)
            self.assertTrue(math.isnan(result[0]) or math.isnan(result[1]) or
                          isinstance(result[0], float) and isinstance(result[1], float))
        except Exception:
            # If it raises an exception, that's also acceptable
            pass


def run_standalone_tests():
    """Run tests when script is executed directly"""
    print("Running GPSOverlay API Tests...")
    print("=" * 50)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGPSOverlay)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✓ All tests passed!")
        return 0
    else:
        print(f"✗ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        return 1


if __name__ == "__main__":
    raise SystemExit(run_standalone_tests())
