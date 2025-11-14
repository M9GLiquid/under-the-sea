#!/usr/bin/env python3
"""
GPSOverlay - Standalone coordinate transformation API

Transforms GPS server coordinates (2048x1536) to:
- Rectified canvas coordinates
- Grid cell positions (dynamically configured)
- Real-world coordinates (mm)

This is a standalone API that can be exported and used independently.

Usage:
    from overlay import GPSOverlay
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
        """
        Initialize the GPSOverlay API by loading calibration data.
        
        Args:
            json_path: Path to gps_overlay.json file. If None, looks for 
                      gps_overlay.json in the same directory as this script.
        
        Example:
            # Load from default location (same directory)
            overlay = GPSOverlay()
            
            # Load from custom path
            overlay = GPSOverlay("path/to/gps_overlay.json")
        """
        if json_path is None:
            # Auto-detect: look for gps_overlay.json in the same directory as this script
            # This makes it easy to copy both files together
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, "gps_overlay.json")

        # Load the calibration JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)["gps_overlay"]

        # Image sizes:
        # - calib_size: Original calibration image size (e.g., 1206x906)
        # - server_size: GPS server image size (typically 2048x1536)
        self.calib_size = tuple(self.data["calibration_size"])
        self.server_size = tuple(self.data["server_size"])

        # ===== Transformation Matrix =====
        # Homography matrix that transforms from corrected (fisheye-removed) space
        # to rectified (top-down) canvas space
        # This is a 3x3 matrix that performs perspective transformation
        self.homography = self.data["homography"]

        # ===== Arena Bounds =====
        # The rectangular bounds of the arena in rectified canvas coordinates
        # Format: {"left": float, "top": float, "right": float, "bottom": float}
        # These define the area where the grid is placed
        self.arena_bounds = self.data["arena_bounds"]

        # ===== Grid Configuration =====
        # Grid dimensions and cell size (dynamically loaded from Tool 4)
        # - grid_cols: Number of columns in the grid (e.g., 14)
        # - grid_rows: Number of rows in the grid (e.g., 9)
        # - cell_size_px: Size of each cell in pixels {"x": float, "y": float}
        self.grid_cols = self.data["grid"]["cols"]
        self.grid_rows = self.data["grid"]["rows"]
        self.cell_size_px = self.data["grid"]["cell_size_px"]

        # ===== Real-World Calibration =====
        # Optional: Converts pixels to millimeters (only if Tool 6 was run)
        # - mm_per_pixel_x: Millimeters per pixel in X direction
        # - mm_per_pixel_y: Millimeters per pixel in Y direction
        self.real_world_available = "real_world" in self.data
        if self.real_world_available:
            self.mm_per_pixel_x = self.data["real_world"]["mm_per_pixel_x"]
            self.mm_per_pixel_y = self.data["real_world"]["mm_per_pixel_y"]
        else:
            # Real-world calibration not available - set to None
            self.mm_per_pixel_x = self.mm_per_pixel_y = None

    def map_coords(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform GPS server coordinates to rectified canvas coordinates.
        
        This is the core transformation function that converts coordinates from
        the original GPS server image space (with fisheye distortion) to the
        rectified top-down view space.
        
        Args:
            x: GPS server X coordinate (typically 0-2048)
            y: GPS server Y coordinate (typically 0-1536)
        
        Returns:
            Tuple of (x_rect, y_rect) in rectified canvas space.
            Returns (nan, nan) if transformation fails (e.g., point outside valid area).
        
        Example:
            # Transform a GPS coordinate to rectified space
            x_rect, y_rect = overlay.map_coords(1024, 768)
            print(f"Rectified position: ({x_rect:.1f}, {y_rect:.1f})")
        """
        # Step 1: Scale from GPS server resolution to calibration resolution
        # The calibration was done at a different resolution than the server image,
        # so we need to scale coordinates to match the calibration space
        scale_x = self.calib_size[0] / self.server_size[0]
        scale_y = self.calib_size[1] / self.server_size[1]

        x_cal = float(x) * scale_x
        y_cal = float(y) * scale_y

        # Step 2: The fisheye correction was already computed during calibration
        # The homography matrix maps directly from corrected space to rectified space
        # So we use the calibration coordinates directly with the homography
        x_corr, y_corr = x_cal, y_cal

        # Step 3: Apply homography transformation (corrected → rectified canvas)
        # Homography is a 3x3 matrix that performs perspective transformation
        # Formula: [x', y', w'] = H * [x, y, 1]
        # Final coordinates: x_rect = x'/w', y_rect = y'/w'
        h = self.homography
        x_rect = h[0][0] * x_corr + h[0][1] * y_corr + h[0][2]
        y_rect = h[1][0] * x_corr + h[1][1] * y_corr + h[1][2]
        w = h[2][0] * x_corr + h[2][1] * y_corr + h[2][2]

        # Check for division by zero (invalid transformation)
        if abs(w) < 1e-10:
            return float('nan'), float('nan')

        # Return normalized coordinates
        return x_rect / w, y_rect / w

    def get_grid_cell(self, x: float, y: float) -> Dict:
        """
        Transform GPS server coordinates to grid cell position.
        
        This function maps GPS coordinates to a specific grid cell in the arena.
        Useful for navigation systems that need to know which cell a robot is in.
        
        Args:
            x: GPS server X coordinate (typically 0-2048)
            y: GPS server Y coordinate (typically 0-1536)
        
        Returns:
            Dictionary with grid cell information:
            {
                "col": int,           # Grid column index (0 to grid_cols-1)
                "row": int,           # Grid row index (0 to grid_rows-1)
                "in_bounds": bool,    # True if point is within arena bounds
                "center_x": float,    # Cell center X coordinate in rectified space
                "center_y": float     # Cell center Y coordinate in rectified space
            }
        
        Example:
            # Get grid cell for a GPS coordinate
            cell = overlay.get_grid_cell(1024, 768)
            if cell["in_bounds"]:
                print(f"Robot is in cell ({cell['col']}, {cell['row']})")
                print(f"Cell center: ({cell['center_x']:.1f}, {cell['center_y']:.1f})")
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
        Transform GPS server coordinates to real-world coordinates in millimeters.
        
        This function converts pixel coordinates to real-world measurements.
        Requires that Tool 6 (Real-World Calibrator) was run to provide calibration data.
        
        Args:
            x: GPS server X coordinate (typically 0-2048)
            y: GPS server Y coordinate (typically 0-1536)
        
        Returns:
            Dictionary with real-world position:
            {
                "x_mm": float,              # X position in millimeters from origin (top-left)
                "y_mm": float,              # Y position in millimeters from origin (top-left)
                "distance_from_origin_mm": float  # Euclidean distance from origin in mm
            }
        
        Raises:
            ValueError: If real-world calibration is not available.
                       Run Tool 6 and recreate gps_overlay.json with Tool 8.
        
        Example:
            # Get real-world position
            try:
                pos = overlay.get_real_coords(1024, 768)
                print(f"Real position: {pos['x_mm']:.1f}mm, {pos['y_mm']:.1f}mm")
                print(f"Distance from origin: {pos['distance_from_origin_mm']:.1f}mm")
            except ValueError as e:
                print("Real-world calibration not available:", e)
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
        Get complete grid mapping as a 2D array.
        
        This function returns a complete representation of all grid cells,
        useful for path planning, visualization, or creating lookup tables.
        
        Returns:
            2D list (rows x cols) where each element is a dictionary with cell info:
            [
                [cell(0,0), cell(1,0), ..., cell(cols-1,0)],  # Row 0
                [cell(0,1), cell(1,1), ..., cell(cols-1,1)],  # Row 1
                ...
            ]
            
            Each cell dictionary contains:
            {
                "col": int,           # Grid column index (0 to cols-1)
                "row": int,           # Grid row index (0 to rows-1)
                "x_mm": float,        # X position in mm (0 if calibration not available)
                "y_mm": float,        # Y position in mm (0 if calibration not available)
                "center_x": float,    # Cell center X in rectified space
                "center_y": float     # Cell center Y in rectified space
            }
        
        Example:
            # Get complete grid map
            grid_map = overlay.get_grid_map()
            
            # Access a specific cell (row, col)
            cell = grid_map[5][3]  # Row 5, Column 3
            print(f"Cell ({cell['col']}, {cell['row']}) center: ({cell['center_x']:.1f}, {cell['center_y']:.1f})")
            
            # Iterate through all cells
            for row in grid_map:
                for cell in row:
                    if cell['x_mm'] > 0:  # Only if real-world calibrated
                        print(f"Cell ({cell['col']}, {cell['row']}): {cell['x_mm']:.1f}mm, {cell['y_mm']:.1f}mm")
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
