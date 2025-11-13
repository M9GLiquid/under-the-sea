#!/usr/bin/env python3
"""
GPSOverlay - Standalone coordinate transformation API

Transforms GPS server coordinates (2048x1536) to:
- Rectified canvas coordinates
- Grid cell positions (dynamically configured)
- Real-world coordinates (mm)

This is a standalone API that can be exported and used independently.

Usage:
    from gps_overlay import GPSOverlay
    overlay = GPSOverlay("gps_overlay.json")

    # Transform GPS coordinates
    cell = overlay.get_grid_cell(50, 50)
    real_pos = overlay.get_real_coords(50, 50)
    grid_map = overlay.get_grid_map()
"""

import json
import math
import os
from typing import Dict, List, Tuple


class GPSOverlay:
    """
    Standalone GPS coordinate transformation API

    Loads calibration data from a single JSON file and provides
    simple functions to transform GPS server coordinates.
    """

    def __init__(self, json_path: str = None):
        """Load calibration data from single JSON file"""
        if json_path is None:
            # Look for gps_overlay.json in the same directory as this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "gps_overlay.json")

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)["gps_overlay"]

        # Camera parameters for fisheye correction
        self.camera_matrix = self.data["camera_matrix"]
        self.dist_coeffs = self.data["distortion_coeffs"]
        self.calib_size = tuple(self.data["calibration_size"])
        self.server_size = tuple(self.data["server_size"])

        # Homography transformation (corrected → rectified canvas)
        self.homography = self.data["homography"]

        # Arena bounds in rectified space
        self.arena_bounds = self.data["arena_bounds"]

        # Grid configuration (dynamically loaded)
        self.grid_cols = self.data["grid"]["cols"]
        self.grid_rows = self.data["grid"]["rows"]
        self.cell_size_px = self.data["grid"]["cell_size_px"]

        # Real-world calibration (if available from Tool 6)
        self.real_world_available = "real_world" in self.data
        if self.real_world_available:
            self.mm_per_pixel_x = self.data["real_world"]["mm_per_pixel_x"]
            self.mm_per_pixel_y = self.data["real_world"]["mm_per_pixel_y"]
            self.origin_mm = self.data["real_world"]["origin_mm"]
        else:
            self.mm_per_pixel_x = self.mm_per_pixel_y = None
            self.origin_mm = {"x": 0, "y": 0}

    def map_coords(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform GPS server coordinates to rectified canvas coordinates

        Args:
            x: GPS server X coordinate (0-2048)
            y: GPS server Y coordinate (0-1536)

        Returns:
            Tuple of (x_rect, y_rect) in rectified canvas space
        """
        # Scale from GPS server resolution to calibration resolution
        scale_x = self.calib_size[0] / self.server_size[0]
        scale_y = self.calib_size[1] / self.server_size[1]

        x_cal = float(x) * scale_x
        y_cal = float(y) * scale_y

        # The fisheye correction was already computed for the calibration image size (1206x906)
        # The homography maps from corrected space to rectified space
        # So we use the calibration coordinates directly with the homography
        x_corr, y_corr = x_cal, y_cal

        # Apply homography transformation (corrected → rectified canvas)
        # Homography matrix multiplication: H * [x, y, 1]
        h = self.homography
        x_rect = h[0][0] * x_corr + h[0][1] * y_corr + h[0][2]
        y_rect = h[1][0] * x_corr + h[1][1] * y_corr + h[1][2]
        w = h[2][0] * x_corr + h[2][1] * y_corr + h[2][2]

        if abs(w) < 1e-10:
            return float('nan'), float('nan')

        return x_rect / w, y_rect / w

    def get_grid_cell(self, x: float, y: float) -> Dict:
        """
        Transform GPS server coordinates to grid cell position

        Args:
            x: GPS server X coordinate (0-2048)
            y: GPS server Y coordinate (0-1536)

        Returns:
            Dict with grid cell info:
            {
                "col": int,           # Grid column (0-44)
                "row": int,           # Grid row (0-28)
                "in_bounds": bool,    # Whether point is within arena
                "center_x": float,    # Cell center X in rectified space
                "center_y": float     # Cell center Y in rectified space
            }
        """
        x_rect, y_rect = self.map_coords(x, y)

        if math.isnan(x_rect) or math.isnan(y_rect):
            return {"col": 0, "row": 0, "in_bounds": False, "center_x": 0, "center_y": 0}

        # Calculate grid cell position
        left, top = self.arena_bounds["left"], self.arena_bounds["top"]
        right, bottom = self.arena_bounds["right"], self.arena_bounds["bottom"]

        cell_width = (right - left) / self.grid_cols
        cell_height = (bottom - top) / self.grid_rows

        col = int((x_rect - left) // cell_width)
        row = int((y_rect - top) // cell_height)

        # Check bounds
        in_bounds = (0 <= col < self.grid_cols and
                    0 <= row < self.grid_rows)

        # Calculate cell center
        center_x = left + (col + 0.5) * cell_width
        center_y = top + (row + 0.5) * cell_height

        return {
            "col": col,
            "row": row,
            "in_bounds": in_bounds,
            "center_x": center_x,
            "center_y": center_y
        }

    def get_real_coords(self, x: float, y: float) -> Dict:
        """
        Transform GPS server coordinates to real-world coordinates in millimeters

        Args:
            x: GPS server X coordinate (0-2048)
            y: GPS server Y coordinate (0-1536)

        Returns:
            Dict with real-world position:
            {
                "x_mm": float,              # X position in mm from origin
                "y_mm": float,              # Y position in mm from origin
                "distance_from_origin_mm": float  # Distance from origin in mm
            }

        Raises:
            ValueError: If real-world calibration is not available
        """
        if not self.real_world_available:
            raise ValueError(
                "Real-world calibration not available. "
                "Run Tool 6 (Real-World Calibrator) and recreate gps_overlay.json with Tool 8."
            )

        x_rect, y_rect = self.map_coords(x, y)

        if math.isnan(x_rect) or math.isnan(y_rect):
            return {"x_mm": 0, "y_mm": 0, "distance_from_origin_mm": 0}

        # Convert from rectified space to real-world mm
        # Origin is at arena top-left corner
        left, top = self.arena_bounds["left"], self.arena_bounds["top"]

        x_mm = (x_rect - left) * self.mm_per_pixel_x
        y_mm = (y_rect - top) * self.mm_per_pixel_y

        distance_mm = math.hypot(x_mm, y_mm)

        return {
            "x_mm": x_mm,
            "y_mm": y_mm,
            "distance_from_origin_mm": distance_mm
        }

    def get_grid_map(self) -> List[List[Dict]]:
        """
        Get complete grid mapping as 2D array

        Returns:
            Grid array (cols x rows) where each element is a dict with cell info:
            {
                "col": int,      # Grid column (0 to cols-1)
                "row": int,      # Grid row (0 to rows-1)
                "x_mm": float,   # X position in mm (if calibration available)
                "y_mm": float,   # Y position in mm (if calibration available)
                "center_x": float,  # Cell center X in rectified space
                "center_y": float   # Cell center Y in rectified space
            }
        """
        grid_map = []

        left, top = self.arena_bounds["left"], self.arena_bounds["top"]
        right, bottom = self.arena_bounds["right"], self.arena_bounds["bottom"]

        cell_width = (right - left) / self.grid_cols
        cell_height = (bottom - top) / self.grid_rows

        for row in range(self.grid_rows):
            grid_row = []

            for col in range(self.grid_cols):
                center_x = left + (col + 0.5) * cell_width
                center_y = top + (row + 0.5) * cell_height

                # Convert to real-world coordinates if available
                if self.real_world_available:
                    x_mm = (center_x - left) * self.mm_per_pixel_x
                    y_mm = (center_y - top) * self.mm_per_pixel_y
                else:
                    x_mm = y_mm = 0

                grid_row.append({
                    "col": col,
                    "row": row,
                    "x_mm": x_mm,
                    "y_mm": y_mm,
                    "center_x": center_x,
                    "center_y": center_y
                })

            grid_map.append(grid_row)

        return grid_map


# Example usage and testing
if __name__ == "__main__":
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
            print(f"\nGPS Server ({gps_x}, {gps_y}):")

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
        print("\nGrid Configuration:")
        print(f"  Size: {overlay.grid_cols}x{overlay.grid_rows} cells")
        print(f"  Cell Size: {overlay.cell_size_px['x']}x{overlay.cell_size_px['y']} pixels")
        print(f"  Arena: {overlay.arena_bounds}")

        if overlay.real_world_available:
            print(f"  Real-World: {overlay.mm_per_pixel_x:.3f} mm/pixel")
        else:
            print("  Real-World: Not calibrated")

        print("\n[SUCCESS] GPSOverlay API ready for use!")
        print("Copy gps_overlay.py and gps_overlay.json to your project.")

    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Run Tool 8 first: python tools/Tool_8_GPS_Overlay.py")
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Make sure gps_overlay.json contains valid calibration data.")
