#!/usr/bin/env python3
"""
Unified Pipeline - Run all Tools 1-8 sequentially

This script orchestrates the complete processing pipeline automatically:
1. Tool 1: Fisheye Correction (interactive)
2. Tool 2: Arena Corner Detection (interactive)
3. Tool 3: Arena Rectification (automated)
4. Tool 4: Grid Overlay (interactive)
5. Tool 6: Real-World Calibration (interactive)
6. Tool 7: Point Mapper (interactive viewer)
7. Tool 8: GPS Overlay API Creator (automated)

Note: Tool 9 (ROS2 Coordinate Viewer) is standalone and should be run separately:
    python tools/Tool_9_ROS2_Coordinate_Viewer.py

Usage:
    python main.py
    
Uses images/GPS-Real.png as input (fetch from camera separately if needed).
No arguments needed - everything is automatic!
"""

import sys
import subprocess
import time
import argparse
import importlib.util
from pathlib import Path
from typing import Literal, Set, Union


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent


def run_tool(tool_name: str, args: list, description: str) -> Union[bool, Literal["quit_pipeline"]]:
    """
    Run a tool script and return True if successful, False on failure,
    or the literal string "quit_pipeline" when the user requests to stop.
    Waits for the tool to complete before returning.
    
    Args:
        tool_name: Name of the tool script (e.g., "Tool_1_Fix_Fisheye.py")
        args: List of command-line arguments to pass to the tool
        description: Human-readable description for progress output
    
    Returns:
        True if tool exited successfully, False otherwise, or "quit_pipeline"
    """
    project_root = get_project_root()
    tool_path = project_root / "tools" / tool_name
    
    if not tool_path.exists():
        print(f"❌ Error: Tool not found: {tool_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Running: python {tool_path.name} {' '.join(args)}")
    print("\n💡 Instructions:")
    print("   - Complete your work in this tool")
    print("   - Press 's' to SAVE your work")
    print("   - Press 'q' to QUIT and continue to next tool")
    print("   - The next tool will start automatically after you quit")
    print()
    
    try:
        # Run tool and wait for it to complete
        # stdout/stderr are inherited so user sees output in real-time
        result = subprocess.run(
            [sys.executable, str(tool_path)] + args,
            cwd=str(project_root),
            check=False,  # Don't raise exception, just check returncode
            stdout=None,  # Inherit stdout (show output)
            stderr=None,  # Inherit stderr (show errors)
        )
        
        # Wait is implicit in subprocess.run() - it blocks until process completes
        # Give a moment for any windows to fully close
        time.sleep(0.5)
        
        if result.returncode == 0:
            print(f"\n✅ {description} completed successfully")
            return True
        elif result.returncode == 2:
            # Special exit code: user wants to quit entire pipeline
            print(f"\n⚠️  {description} - User requested to quit pipeline")
            return "quit_pipeline"
        else:
            print(f"\n❌ {description} failed with exit code {result.returncode}")
            return False
            
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted by user during {description}")
        return False
    except Exception as e:
        print(f"\n❌ Error running {description}: {e}")
        return False


def derive_output_paths(base_name: str) -> dict:
    """
    Derive all intermediate file paths based on base image name.
    
    Args:
        base_name: Base name without extension (e.g., "GPS-Real")
    
    Returns:
        Dictionary with all expected file paths
    """
    project_root = get_project_root()
    return {
        "original_image": project_root / "images" / f"{base_name}.png",
        "corrected_image": project_root / "output" / f"{base_name}_corrected.png",
        "fisheye_calibration": project_root / "data" / f"{base_name}_fisheye_calibration.json",
        "corners_image": project_root / "output" / f"{base_name}_corners.png",
        "corners_json": project_root / "data" / f"{base_name}_corrected_corners.json",
        "rectified_image": project_root / "output" / f"{base_name}_corrected_rectified_oriented.png",
        "transform_json": project_root / "data" / f"{base_name}_corrected_transform.json",
        "grid_json": project_root / "data" / f"{base_name}_corrected_rectified_oriented_grid.json",
        "calibration_json": project_root / "data" / f"{base_name}_corrected_rectified_oriented_calibration.json",
        "gps_overlay_json": project_root / "data" / "gps_overlay.json",
    }


def check_file_exists(file_path: Path, description: str) -> bool:
    """Check if a file exists and print status."""
    if file_path.exists():
        print(f"  ✓ {description}: {file_path.name}")
        return True
    else:
        print(f"  ✗ {description}: {file_path.name} (missing)")
        return False


def parse_tool_spec(spec: str) -> Set[int]:
    """
    Parse tool specification string into a set of tool numbers.
    
    Supports:
    - Individual tools: "1,2,3"
    - Ranges: "1-4"
    - Combinations: "1,3-5,7"
    - All: "1-8" or "all"
    
    Args:
        spec: Tool specification string
    
    Returns:
        Set of tool numbers (1-8)
    """
    if not spec or spec.lower() == "all":
        return set(range(1, 9))  # Tools 1-8
    
    tools = set()
    parts = spec.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range: "1-4"
            try:
                start, end = part.split('-')
                start = int(start.strip())
                end = int(end.strip())
                tools.update(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid range format: {part}")
        else:
            # Single number
            try:
                tools.add(int(part))
            except ValueError:
                raise ValueError(f"Invalid tool number: {part}")
    
    # Validate tool numbers
    # Tool 9 is standalone but can be called via --tool 9
    valid_tools = set(range(1, 10))  # Tools 1-9 (Tool 9 is standalone but callable)
    invalid = tools - valid_tools
    if invalid:
        raise ValueError(f"Invalid tool numbers: {invalid}. Valid tools are 1-9")
    
    return tools


def main():
    parser = argparse.ArgumentParser(
        description="Unified Pipeline - Run Tools 1-8 sequentially",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run all tools (1-8) with web camera (default)
  python main.py --source web       # Explicitly fetch from web camera
  python main.py --source image     # Use local images/GPS-Real.png file
  python main.py --tool 1,2,3       # Run tools 1, 2, and 3
  python main.py --tool 1-4         # Run tools 1 through 4
  python main.py --tool 1,3-5,7     # Run tools 1, 3, 4, 5, and 7
  python main.py --tool 6           # Run only tool 6
  
  # Tool 9 is standalone - run separately:
  python tools/Tool_9_ROS2_Coordinate_Viewer.py
        """
    )
    parser.add_argument(
        "--tool",
        type=str,
        default="all",
        help="Tools to run: comma-separated list (1,2,3), ranges (1-4), or 'all' (default: all)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="web",
        choices=["web", "image"],
        help="Image source: 'web' (default) to fetch from web camera, 'image' to use local file"
    )
    
    args = parser.parse_args()
    
    # Parse tool specification
    try:
        tools_to_run = parse_tool_spec(args.tool)
        
        # Handle Tool 9 separately (standalone tool)
        if 9 in tools_to_run:
            tools_to_run.remove(9)
            print("\n" + "="*60)
            print("🔧 Tool 9: ROS2 Coordinate Viewer (Standalone)")
            print("="*60)
            tool9_path = get_project_root() / "tools" / "Tool_9_ROS2_Coordinate_Viewer.py"
            if not tool9_path.exists():
                print(f"❌ Error: Tool 9 not found: {tool9_path}")
                return 1
            print(f"Running standalone tool: python {tool9_path.name}")
            print("="*60 + "\n")
            result = subprocess.run(
                [sys.executable, str(tool9_path)],
                cwd=str(get_project_root()),
                check=False
            )
            return result.returncode
    except ValueError as e:
        print(f"❌ Error parsing --tool argument: {e}")
        return 1
    
    project_root = get_project_root()
    
    # Determine image source and fetch/load accordingly
    image_path = project_root / "images" / "GPS-Real.png"
    
    if args.source == "web":
        # Fetch from web camera
        try:
            axis_module_path = project_root / "tools" / "axis_test.py"
            if not axis_module_path.exists():
                raise ImportError(f"axis_test module not found at {axis_module_path}")
            spec = importlib.util.spec_from_file_location("axis_test", axis_module_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load axis_test module at {axis_module_path}")
            axis_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(axis_module)
            fetch_axis_snapshot = getattr(axis_module, "fetch_axis_snapshot")
            
            # Fetch and save to default location
            img = fetch_axis_snapshot(str(image_path))
            img = fetch_axis_snapshot(str(image_path))
            if img is not None:
                print(f"✓ Successfully fetched image from web camera")
                print(f"   Saved to: {image_path.name}")
            else:
                print(f"⚠ Failed to fetch from web camera, checking for local file...")
                if not image_path.exists():
                    print(f"❌ Error: Could not fetch from web and no local image found: {image_path}")
                    print("   Please ensure images/GPS-Real.png exists or check camera connection")
                    return 1
        except ImportError as e:
            print(f"⚠ Warning: Could not import axis_test module: {e}")
            print(f"   Falling back to local file...")
            if not image_path.exists():
                print(f"❌ Error: Input image not found: {image_path}")
                print("   Please ensure images/GPS-Real.png exists")
                return 1
        except Exception as e:
            print(f"⚠ Warning: Error fetching from web camera: {e}")
            print(f"   Falling back to local file...")
            if not image_path.exists():
                print(f"❌ Error: Input image not found: {image_path}")
                print("   Please ensure images/GPS-Real.png exists")
                return 1
    else:  # args.source == "image"
        # Use local file
        if not image_path.exists():
            print(f"❌ Error: Input image not found: {image_path}")
            print("   Please ensure images/GPS-Real.png exists")
            print("   (Or use --source web to fetch from camera)")
            return 1
        print(f"\n📷 Using local image: {image_path.name}")
    
    print(f"🔧 Tools to run: {sorted(tools_to_run)}")
    
    # Extract base name from image path
    base_name = image_path.stem
    if "_corrected" in base_name:
        base_name = base_name.replace("_corrected", "")
    if "_rectified" in base_name:
        base_name = base_name.split("_rectified")[0]
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Unified Pipeline")
    print(f"{'='*60}")
    print(f"Input image: {image_path.name}")
    print(f"Base name: {base_name}")
    print()
    
    # Derive all file paths
    paths = derive_output_paths(base_name)
    
    # Tool 1: Fisheye Correction
    if 1 in tools_to_run:
        result = run_tool(
            "Tool_1_Fix_Fisheye.py",
            [str(image_path)],
            "Tool 1: Fisheye Correction"
        )
        if result == "quit_pipeline":
            print("\n⚠️  Pipeline stopped by user.")
            return 0
        if not result:
            print("\n❌ Tool 1 failed. Cannot continue.")
            return 1
        # Update image path for next tool
        image_path = paths["corrected_image"]
    else:
        print("\n⏭️  Skipping Tool 1 (not in --tool list)")
        if paths["corrected_image"].exists():
            image_path = paths["corrected_image"]
    
    # Tool 2: Arena Corner Detection
    if 2 in tools_to_run:
        result = run_tool(
            "Tool_2_Detect_Arena_Corners.py",
            [str(image_path)],
            "Tool 2: Arena Corner Detection"
        )
        if result == "quit_pipeline":
            print("\n⚠️  Pipeline stopped by user.")
            return 0
        if not result:
            print("\n❌ Tool 2 failed. Cannot continue.")
            return 1
    else:
        print("\n⏭️  Skipping Tool 2 (not in --tool list)")
    
    # Tool 3: Arena Rectification (automated)
    if 3 in tools_to_run:
        corners_input = paths["corners_json"]
        if not run_tool(
            "Tool_3_Rectify_Arena_Square.py",
            [str(corners_input)],
            "Tool 3: Arena Rectification"
        ):
            print("\n❌ Tool 3 failed. Cannot continue.")
            return 1
    else:
        print("\n⏭️  Skipping Tool 3 (not in --tool list)")
    
    # Tool 4: Grid Overlay
    if 4 in tools_to_run:
        rectified_input = paths["rectified_image"]
        result = run_tool(
            "Tool_4_Grid_Overlay.py",
            [str(rectified_input)],
            "Tool 4: Grid Overlay"
        )
        if result == "quit_pipeline":
            print("\n⚠️  Pipeline stopped by user.")
            return 0
        if not result:
            print("\n❌ Tool 4 failed. Cannot continue.")
            return 1
    else:
        print("\n⏭️  Skipping Tool 4 (not in --tool list)")
    
    # Tool 5: Grid Inspector (optional viewer)
    if 5 in tools_to_run:
        # Find grid image - it might have different dimensions in filename
        grid_images = list((project_root / "output").glob(f"{base_name}_corrected_rectified_oriented_grid_*.png"))
        if grid_images:
            grid_image = grid_images[0]  # Use first match
            run_tool(
                "Tool_5_Grid_Inspector.py",
                [str(grid_image)],
                "Tool 5: Grid Inspector (Viewer)"
            )
        else:
            print("\n⚠️  No grid image found for Tool 5. Skipping.")
    else:
        print("\n⏭️  Skipping Tool 5 (not in --tool list)")
    
    # Tool 6: Real-World Calibration
    if 6 in tools_to_run:
        grids_input = paths["grid_json"]
        result = run_tool(
            "Tool_6_Real_World_Calibrator.py",
            [str(grids_input)],
            "Tool 6: Real-World Calibration"
        )
        if result == "quit_pipeline":
            print("\n⚠️  Pipeline stopped by user.")
            return 0
        if not result:
            print("\n⚠️  Tool 6 failed. Continuing without real-world calibration.")
            print("   You can run it manually later if needed.")
    else:
        print("\n⏭️  Skipping Tool 6 (not in --tool list)")
    
    # Tool 7: Point Mapper (interactive viewer)
    if 7 in tools_to_run:
        tool7_args = [
            "--fisheye-json", str(paths["fisheye_calibration"]),
            "--transform-json", str(paths["transform_json"])
        ]
        result = run_tool(
            "Tool_7_Point_Mapper.py",
            tool7_args,
            "Tool 7: Point Mapper (Viewer)"
        )
        if result == "quit_pipeline":
            print("\n⚠️  Pipeline stopped by user.")
            return 0
        if not result:
            print("\n⚠️  Tool 7 failed. Continuing to next tool.")
    else:
        print("\n⏭️  Skipping Tool 7 (not in --tool list)")
    
    # Tool 8: GPS Overlay API Creator (automated)
    if 8 in tools_to_run:
        tool8_args = [
            "--fisheye-json", str(paths["fisheye_calibration"]),
            "--transform-json", str(paths["transform_json"]),
            "--grids-json", str(paths["grid_json"]),
            "--output", str(paths["gps_overlay_json"])
        ]
        
        # Add calibration JSON if it exists
        if paths["calibration_json"].exists():
            tool8_args.extend(["--calibration-json", str(paths["calibration_json"])])
        
        if not run_tool(
            "Tool_8_GPS_Overlay.py",
            tool8_args,
            "Tool 8: GPS Overlay API Creator"
        ):
            print("\n❌ Tool 8 failed.")
            return 1
    else:
        print("\n⏭️  Skipping Tool 8 (not in --tool list)")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ Pipeline Complete!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    check_file_exists(paths["corrected_image"], "Corrected image")
    check_file_exists(paths["fisheye_calibration"], "Fisheye calibration")
    check_file_exists(paths["corners_json"], "Corners JSON")
    check_file_exists(paths["rectified_image"], "Rectified image")
    check_file_exists(paths["transform_json"], "Transform JSON")
    check_file_exists(paths["grid_json"], "Grid JSON")
    if paths["calibration_json"].exists():
        check_file_exists(paths["calibration_json"], "Real-world calibration")
    check_file_exists(paths["gps_overlay_json"], "GPS Overlay JSON (final output)")
    
    # Export guidance
    modules_dir = project_root / "modules"
    overlay_api_path = modules_dir / "overlay-api.py"
    if overlay_api_path.exists() and paths["gps_overlay_json"].exists():
        print(f"\n📦 API Package:")
        print(f"  Copy these to consume the GPSOverlay API:")
        print(f"    - modules/overlay-api.py")
        print(f"    - data/{paths['gps_overlay_json'].name}")
    
    print(f"\n🎉 All done! Final output: {paths['gps_overlay_json'].name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
