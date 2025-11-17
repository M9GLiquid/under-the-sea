#!/usr/bin/env python3
"""
Test script for GPSOverlay API

This script demonstrates how to use the overlay API and tests basic functionality.

Usage:
    # Run all tests
    python test_overlay.py
    
    # Run specific test
    python test_overlay.py test_coordinates
    python test_overlay.py test_grid_cells
    python test_overlay.py test_real_world
    python test_overlay.py test_image_transform
"""

import sys
import os
from overlay import GPSOverlay


def test_coordinates():
    """Test coordinate transformation from GPS server space to rectified space"""
    print("=" * 50)
    print("Test: Coordinate Transformation")
    print("=" * 50)
    
    try:
        overlay = GPSOverlay()
        
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
        
        print("\n[SUCCESS] Coordinate transformation test passed!")
        return True
        
    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Run Tool 8 first: python tools/Tool_8_GPS_Overlay.py")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_grid_cells():
    """Test grid cell mapping from GPS coordinates"""
    print("=" * 50)
    print("Test: Grid Cell Mapping")
    print("=" * 50)
    
    try:
        overlay = GPSOverlay()
        
        test_coords = [
            (50, 50),     # Top-left area
            (1024, 768),  # Center area
            (2000, 1500)  # Bottom-right area
        ]
        
        for gps_x, gps_y in test_coords:
            print(f"\nGPS Server ({gps_x}, {gps_y}):")
            
            # Get grid cell
            cell = overlay.get_grid_cell(gps_x, gps_y)
            print(f"  -> Grid Cell: ({cell['col']}, {cell['row']})")
            print(f"  -> In bounds: {cell['in_bounds']}")
            print(f"  -> Cell center: ({cell['center_x']:.1f}, {cell['center_y']:.1f})")
        
        # Show grid info
        print("\nGrid Configuration:")
        print(f"  Size: {overlay.grid_cols}x{overlay.grid_rows} cells")
        print(f"  Cell Size: {overlay.cell_size_px['x']}x{overlay.cell_size_px['y']} pixels")
        print(f"  Arena Bounds: {overlay.arena_bounds}")
        
        print("\n[SUCCESS] Grid cell mapping test passed!")
        return True
        
    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Run Tool 8 first: python tools/Tool_8_GPS_Overlay.py")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_real_world():
    """Test real-world coordinate conversion (requires Tool 6 calibration)"""
    print("=" * 50)
    print("Test: Real-World Coordinates")
    print("=" * 50)
    
    try:
        overlay = GPSOverlay()
        
        if not overlay.real_world_available:
            print("[SKIP] Real-world calibration not available")
            print("Run Tool 6 first to enable real-world coordinate conversion")
            return True
        
        test_coords = [
            (50, 50),     # Top-left area
            (1024, 768),  # Center area
            (2000, 1500)  # Bottom-right area
        ]
        
        for gps_x, gps_y in test_coords:
            print(f"\nGPS Server ({gps_x}, {gps_y}):")
            
            # Get real-world coordinates
            real_pos = overlay.get_real_coords(gps_x, gps_y)
            print(f"  -> Real World: {real_pos['x_mm']:.1f}mm, {real_pos['y_mm']:.1f}mm")
            print(f"  -> Distance from origin: {real_pos['distance_from_origin_mm']:.1f}mm")
        
        print("\nReal-World Calibration:")
        print(f"  X: {overlay.mm_per_pixel_x:.3f} mm/pixel")
        print(f"  Y: {overlay.mm_per_pixel_y:.3f} mm/pixel")
        
        print("\n[SUCCESS] Real-world coordinate test passed!")
        return True
        
    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Run Tool 8 first: python tools/Tool_8_GPS_Overlay.py")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_image_transform():
    """Test image transformation from raw GPS image to rectified view"""
    print("=" * 50)
    print("Test: Image Transformation")
    print("=" * 50)
    
    try:
        import cv2
    except ImportError:
        print("[SKIP] Image transformation requires OpenCV and NumPy")
        print("Install with: pip install opencv-python numpy")
        return True
    
    try:
        overlay = GPSOverlay()
        
        # Resolve image path relative to this script's location
        # Script is in api/, image is in images/ at project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up from api/ to project root
        test_image_path = os.path.join(project_root, "images", "GPS-Real.png")
        
        if not os.path.exists(test_image_path):
            print(f"[ERROR] Test image not found: {test_image_path}")
            print("Make sure images/GPS-Real.png exists in the project root")
            return False
        
        print(f"Test image: {test_image_path}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(project_root, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Test with grid overlay
        print("\nTransforming image with grid overlay...")
        rectified, offset = overlay.transform_image(test_image_path, show_grid=True)
        print(f"  -> Rectified image size: {rectified.shape[1]}x{rectified.shape[0]}")
        print(f"  -> Offset: x={offset['offset_x']}, y={offset['offset_y']}")
        
        output_path = os.path.join(output_dir, "test_rectified_with_grid.png")
        cv2.imwrite(output_path, rectified)
        print(f"  -> Saved to: {output_path}")
        
        # Test without grid overlay
        print("\nTransforming image without grid overlay...")
        rectified_no_grid, offset_no_grid = overlay.transform_image(test_image_path, show_grid=False)
        print(f"  -> Rectified image size: {rectified_no_grid.shape[1]}x{rectified_no_grid.shape[0]}")
        print(f"  -> Offset: x={offset_no_grid['offset_x']}, y={offset_no_grid['offset_y']}")
        
        output_path_no_grid = os.path.join(output_dir, "test_rectified_no_grid.png")
        cv2.imwrite(output_path_no_grid, rectified_no_grid)
        print(f"  -> Saved to: {output_path_no_grid}")
        
        # Test grid cell conversion from click coordinates
        print("\nTesting grid cell conversion from click coordinates...")
        # Simulate a click at center of image
        click_x_img = rectified.shape[1] // 2
        click_y_img = rectified.shape[0] // 2
        x_canvas = click_x_img + offset["offset_x"]
        y_canvas = click_y_img + offset["offset_y"]
        cell = overlay.get_grid_cell_from_rectified(x_canvas, y_canvas)
        print(f"  -> Click at image pixel ({click_x_img}, {click_y_img})")
        print(f"  -> Canvas coordinate ({x_canvas:.1f}, {y_canvas:.1f})")
        if cell["in_bounds"]:
            print(f"  -> Grid cell: ({cell['col']}, {cell['row']})")
        else:
            print(f"  -> Click is outside arena bounds")
        
        print("\n[SUCCESS] Image transformation test passed!")
        return True
        
    except FileNotFoundError:
        print("[ERROR] gps_overlay.json not found!")
        print("Run Tool 8 first: python tools/Tool_8_GPS_Overlay.py")
        return False
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def run_all_tests():
    """Run all available tests"""
    print("\n" + "=" * 50)
    print("GPSOverlay API - Running All Tests")
    print("=" * 50 + "\n")
    
    results = []
    
    # Run all tests
    results.append(("Coordinate Transformation", test_coordinates()))
    results.append(("Grid Cell Mapping", test_grid_cells()))
    results.append(("Real-World Coordinates", test_real_world()))
    results.append(("Image Transformation", test_image_transform()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary")
    print("=" * 50)
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results:
        if result is True:
            print(f"  ✓ {test_name}: PASSED")
            passed += 1
        elif result is False:
            print(f"  ✗ {test_name}: FAILED")
            failed += 1
        else:
            print(f"  ⊘ {test_name}: SKIPPED")
            skipped += 1
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n[SUCCESS] All tests passed!")
        return 0
    else:
        print(f"\n[FAILURE] {failed} test(s) failed")
        return 1


def main():
    """Main entry point - run specific test or all tests"""
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        
        if test_name == "test_coordinates" or test_name == "coordinates":
            test_coordinates()
        elif test_name == "test_grid_cells" or test_name == "grid_cells":
            test_grid_cells()
        elif test_name == "test_real_world" or test_name == "real_world":
            test_real_world()
        elif test_name == "test_image_transform" or test_name == "image_transform":
            test_image_transform()
        else:
            print(f"Unknown test: {test_name}")
            print("\nAvailable tests:")
            print("  test_coordinates    - Test coordinate transformation")
            print("  test_grid_cells     - Test grid cell mapping")
            print("  test_real_world     - Test real-world coordinates")
            print("  test_image_transform - Test image transformation")
            print("\nOr run without arguments to run all tests")
            sys.exit(1)
    else:
        # Run all tests
        sys.exit(run_all_tests())


if __name__ == "__main__":
    main()
