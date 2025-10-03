#!/usr/bin/env python3
"""
GPS Grid Overlay System
Main application entry point for the coordinate grid overlay system.
"""

import sys
import argparse
from pathlib import Path

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="GPS Grid Overlay System - Create coordinate grids on overhead images"
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the overhead image file"
    )
    parser.add_argument(
        "--grid-width",
        type=int,
        default=50,
        help="Grid cell width in pixels (default: 50)"
    )
    parser.add_argument(
        "--grid-height", 
        type=int,
        default=50,
        help="Grid cell height in pixels (default: 50)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for exported data (default: ./data)"
    )
    
    args = parser.parse_args()
    
    # Validate image file exists
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file '{args.image_path}' not found.")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"GPS Grid Overlay System")
    print(f"Image: {args.image_path}")
    print(f"Grid size: {args.grid_width}x{args.grid_height} pixels")
    print(f"Output directory: {args.output_dir}")
    
    # TODO: Initialize and run the main application components
    # This will be implemented in subsequent tasks
    print("Application components will be initialized in subsequent implementation tasks.")

if __name__ == "__main__":
    main()