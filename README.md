# Under-The-Sea Arena Processing Pipeline

A comprehensive Python toolkit for processing fisheye images of underwater arenas, correcting distortion, detecting boundaries, rectifying to top-down views, and creating coordinate grids for robot navigation.

## Project Structure

```
├── main.py                 # Unified pipeline - run all tools sequentially
├── tools/                  # Processing tools (Tool 1-8)
│   ├── Tool_1_Fix_Fisheye.py
│   ├── Tool_2_Detect_Arena_Corners.py
│   ├── Tool_3_Rectify_Arena_Square.py
│   ├── Tool_4_Grid_Overlay.py
│   ├── Tool_5_Grid_Inspector.py
│   ├── Tool_6_Real_World_Calibrator.py
│   ├── Tool_7_Point_Mapper.py
│   ├── Tool_8_GPS_Overlay.py
│   └── axis_test.py        # Axis camera snapshot tool
├── src/                    # Programmatic APIs and shared utilities
│   └── tools/
│       ├── gps_api.py      # GPSMapper API
│       └── gps_overlay.py  # Standalone GPSOverlay API
├── data/                   # JSON data files (calibrations, transforms, grids)
├── output/                 # Processed images
├── images/                 # Input images (GPS-Real.png)
├── tests/                  # Test files and validation scripts
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

**Note:** If you're on Ubuntu/XUbuntu with Python 3.12+, you may need to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start: Unified Pipeline

The easiest way to process images is using the unified pipeline that runs all tools sequentially:

```bash
# Run all tools (default)
python main.py

# Run specific tools
python main.py --tool 1,2,3

# Run a range of tools
python main.py --tool 1-4

# Run combination (individual + range)
python main.py --tool 1,3-5,7

# Run single tool
python main.py --tool 6
```

**Workflow:**
1. Place your image at `images/GPS-Real.png` (or fetch from Axis camera using `python tools/axis_test.py`)
2. Run `python main.py` - it will run Tools 1-8 sequentially
3. For each interactive tool: complete your work → press 's' to save → press 'q' to quit
4. The next tool starts automatically after you quit

**Tool Selection:**
- Default: Runs all tools (1-8)
- `--tool` argument supports:
  - Individual tools: `1,2,3`
  - Ranges: `1-4`
  - Combinations: `1,3-5,7`
  - All: `all` or omit the argument

---

## Complete Processing Pipeline

The toolkit consists of 8 sequential tools that transform raw fisheye images into calibrated, grid-based coordinate systems, plus a standalone API for production use:

### Tool 1: Fisheye Correction (`tools/Tool_1_Fix_Fisheye.py`)
**Purpose:** Correct fisheye lens distortion in underwater arena images.

**Usage:**
```bash
python tools/Tool_1_Fix_Fisheye.py images/GPS-Real.png
```

**Process:**
1. Click points along curved wall edges (minimum 8 points)
2. Tool calculates fisheye calibration parameters
3. Applies distortion correction using OpenCV fisheye functions

**Controls:**
- Left click: Mark points on curved edges
- Middle mouse + drag: Pan view
- Mouse wheel: Zoom in/out
- 'c': Clear all points
- 's': Save corrected image and calibration data
- 'q': Quit

**Outputs:**
- `output/GPS-Real_corrected.png` - Distortion-corrected image
- `data/GPS-Real_fisheye_calibration.json` - Calibration parameters

---

### Tool 2: Arena Corner Detection (`tools/Tool_2_Detect_Arena_Corners.py`)
**Purpose:** Detect the four corners of the arena from the corrected image.

**Usage:**
```bash
python tools/Tool_2_Detect_Arena_Corners.py output/GPS-Real_corrected.png
```

**Process:**
1. Mark two points per wall (8 total points)
2. Tool extends lines and finds intersections
3. Calculates arena corner coordinates

**Controls:**
- Left click: Mark wall points (2 per wall)
- Middle mouse + drag: Pan view
- Mouse wheel: Zoom in/out
- 'c': Clear all points
- 's': Save corner data and visualization
- 'q': Quit

**Outputs:**
- `output/GPS-Real_corners.png` - Visualization with detected corners
- `data/GPS-Real_corners.json` - Corner coordinates

---

### Tool 3: Arena Rectification (`tools/Tool_3_Rectify_Arena_Square.py`)
**Purpose:** Transform the arena to a top-down, orientation-aligned rectangle.

**Usage:**
```bash
# Preferred (pass corners JSON from Tool 2)
python tools/Tool_3_Rectify_Arena_Square.py data/GPS-Real_corrected_corners.json

# Or pass the corrected image path; the tool will auto-resolve data/GPS-Real_corrected_corners.json
python tools/Tool_3_Rectify_Arena_Square.py output/GPS-Real_corrected.png
```

**Process:**
1. Loads corner data from JSON file
2. Computes homography transformation
3. Warps image to top-down view
4. Estimates camera orientation (yaw, pitch, roll)

**Features:**
- Automatically determines optimal output dimensions
- Expands canvas to prevent content cropping
- Draws dashed red border around rectified arena
- Optional: `--width W --height H` for custom dimensions
- Optional: `--size N` for square output

**Outputs:**
- `output/GPS-Real_rectified_oriented.png` - Top-down rectified image
- `data/GPS-Real_transform.json` - Transformation matrix and metadata

---

### Tool 4: Grid Overlay (`tools/Tool_4_Grid_Overlay.py`)
**Purpose:** Create coordinate grids within the rectified arena bounds.

**Usage:**
```bash
# Preferred: pass the rectified PNG from Tool 3
python tools/Tool_4_Grid_Overlay.py output/GPS-Real_corrected_rectified_oriented.png

# Alternatively: pass the transform JSON; the tool will resolve the rectified image path
python tools/Tool_4_Grid_Overlay.py data/GPS-Real_corrected_transform.json
```

**Process:**
1. Loads arena bounds from transform JSON
2. Generates "close-enough" grids with uniform cell distribution
3. Allows interactive adjustment of grid density

**Controls:**
- '+': Increase grid density (more cells)
- '-': Decrease grid density (fewer cells)
- 's': Save multiple valid grid configurations
- 'q': Quit

**Features:**
- Grid confined strictly within arena bounds
- Cell size constraints: 8px minimum, max half arena dimension
- Clean naming: saves to `GPS-Real_grid.json` (no grid size in filename)
- Dynamic grid detection: API automatically uses the configured grid size
- Visual feedback with dashed red grid lines

**Outputs:**
- `output/GPS-Real_grid.png` - Current grid overlay image
- `data/GPS-Real_grid.json` - Current grid settings and metadata

---

### Tool 5: Grid Inspector (`tools/Tool_5_Grid_Inspector.py`)
**Purpose:** Inspect saved grid images and view cell coordinates.

**Usage:**
```bash
python tools/Tool_5_Grid_Inspector.py output/GPS-Real_grid.png
```

**Process:**
1. Loads grid image and configuration from JSON file
2. Displays cell coordinates on mouse hover
3. Shows real-time grid cell indices as you move the cursor

**Features:**
- Hover over cells to see coordinates (e.g., "Cell (0, 0)")
- Loads grid configuration from JSON file (created by Tool 4)
- Automatic arena bounds detection
- Clean window close handling

---

### Tool 6: Real-World Calibrator (`tools/Tool_6_Real_World_Calibrator.py`)
**Purpose:** Convert rectified grid pixels into real-world millimetres.

**Usage:**
```bash
python tools/Tool_6_Real_World_Calibrator.py data/GPS-Real_corrected_rectified_oriented_grid.json
```

**Process:**
1. Loads grid settings and derives rectified arena corners
2. Calculates pixel spans for each wall and both diagonals
3. Prompts for two inputs in millimetres: Height (BL↔TL and BR↔TR) and Width (TL↔TR and BL↔BR)
4. Computes mm↔px ratios and prints summary stats
5. Optionally opens the rectified image while streaming Top-Left→cursor distance in millimetres
6. Press `s` to save the calibration JSON

**Outputs:**
- `data/GPS-Real_corrected_rectified_oriented_calibration.json` – Pixel↔mm conversion stats, user-supplied wall lengths, derived diagonals

---

### Tool 7: Point Mapper (Original → Rectified) (`tools/Tool_7_Point_Mapper.py`)
**Purpose:** Click on the original image and see where that point lands on the rectified top‑down image. Also draws how the entire original frame maps onto the rectified canvas. Supports both manual size specification (for testing) and auto-detection from the loaded image (for API usage).

**Usage:**
```bash
# Manual size specification (for testing/development)
python tools/Tool_7_Point_Mapper.py --server-width 2048 --server-height 1536

# Auto-detect from image (for API usage with actual server streams)
python tools/Tool_7_Point_Mapper.py --auto-detect-size
```

**Process:**
1. Loads fisheye calibration (`data/GPS-Real_fisheye_calibration.json`) and rectification transform (`data/GPS-Real_corrected_transform.json`).
2. Undistorts the clicked original point (fisheye → corrected).
3. Applies the rectification homography (corrected → rectified canvas).
4. Renders the mapped point on the rectified image and overlays the transformed original bounds.

**Controls:**
- Left click (Original window): add a point and map it to Rectified
- r: clear points
- s: save a side-by-side snapshot to `output/`
- q: quit

**Notes:**
- Use `--server-width/--server-height` to match your GPS server's original image resolution. Defaults match `images/GPS-Real.png` (2048×1536).

---

### Tool 8: GPS Overlay API Creator (`tools/Tool_8_GPS_Overlay.py`)
**Purpose:** Consolidates all calibration data into a single standalone JSON file for production API usage.

**Usage:**
```bash
# Create standalone API package (auto-detects all required files)
python tools/Tool_8_GPS_Overlay.py

# Custom output location
python tools/Tool_8_GPS_Overlay.py --output path/to/gps_overlay.json

# With real-world calibration (after running Tool 6)
python tools/Tool_8_GPS_Overlay.py --calibration-json data/GPS-Real_corrected_rectified_oriented_calibration.json
```

**Process:**
1. Loads fisheye calibration data from existing JSON files
2. Dynamically detects grid configuration from Tool 4 output
3. Includes real-world calibration if available (from Tool 6)
4. Creates single `gps_overlay.json` file for standalone API usage

**Output:**
- `data/gps_overlay.json` - Complete calibration data in single file
- Ready for export to production systems

**Notes:**
- Run after completing Tools 1-7 (or use unified pipeline)
- For real-world coordinates, run Tool 6 first to generate calibration data
- Output file is completely self-contained for standalone API usage
- Grid size is detected dynamically from the saved configuration

---

## Fetching Images from Axis Camera

To fetch a fresh image from an Axis P1346 camera:

```bash
# Fetch and save to images/GPS-Real.png
python tools/axis_test.py
```

Or use programmatically:
```python
from tools.axis_test import fetch_axis_snapshot

# Fetch and save to default location (images/GPS-Real.png)
img = fetch_axis_snapshot()

# Or specify custom output path
img = fetch_axis_snapshot("images/my_snapshot.png")
```

**Configuration:** Edit the hardcoded values in `tools/axis_test.py`:
- `CAMERA_IP` - Your camera IP address
- `USERNAME` - Camera username
- `PASSWORD` - Camera password

---

## Programmatic API (for other codebases)

Use the minimal `GPSMapper` to map points and compute grid cells without any GUI or CLI.

Location: `src/tools/gps_api.py`

```python
from src.tools.gps_api import GPSMapper

# Method 1: Default server size (2048x1536)
mapper = GPSMapper(
    fisheye_json="data/GPS-Real_fisheye_calibration.json",
    transform_json="data/GPS-Real_corrected_transform.json",
)

# Method 2: Custom server size for different resolutions
mapper = GPSMapper(
    fisheye_json="data/GPS-Real_fisheye_calibration.json",
    transform_json="data/GPS-Real_corrected_transform.json",
    server_size=(1920, 1080),  # Custom resolution
)

# Method 3: Auto-detect from image (recommended for API usage)
mapper = GPSMapper.from_image(
    fisheye_json="data/GPS-Real_fisheye_calibration.json",
    transform_json="data/GPS-Real_corrected_transform.json",
    image_path="path/to/your/server/image.jpg",
)

# Map a GPS server point to rectified canvas pixels
x_rect, y_rect = mapper.map_original_to_rectified(258, 50)

# Get grid cell from original (GPS) coordinates
cell = mapper.grid_cell_from_original(258, 50, cols=11, rows=8)

# Or from rectified pixels
cell2 = mapper.grid_cell_from_rectified(x_rect, y_rect, cols=11, rows=8)
```

Notes:
- Server image size can be specified manually, auto-detected from an image file, or use the default (2048×1536).
- The scale factor (0.8) matches Tool 1's corrected camera matrix; keep them in sync.


## Complete Workflow Examples

### Option 1: Unified Pipeline (Recommended)
```bash
# Fetch fresh image from camera (optional)
python tools/axis_test.py

# Run all tools sequentially
python main.py

# Or run specific tools
python main.py --tool 1-4        # Run tools 1 through 4
python main.py --tool 1,3,5      # Run tools 1, 3, and 5
python main.py --tool 6          # Run only tool 6
```

### Option 2: Manual Step-by-Step Processing
```bash
# 1. Fetch image from camera (optional)
python tools/axis_test.py

# 2. Correct fisheye distortion
python tools/Tool_1_Fix_Fisheye.py images/GPS-Real.png

# 3. Detect arena corners
python tools/Tool_2_Detect_Arena_Corners.py output/GPS-Real_corrected.png

# 4. Rectify to top-down view
python tools/Tool_3_Rectify_Arena_Square.py data/GPS-Real_corrected_corners.json

# 5. Create coordinate grids
python tools/Tool_4_Grid_Overlay.py output/GPS-Real_corrected_rectified_oriented.png

# 6. Inspect grid cells (optional)
python tools/Tool_5_Grid_Inspector.py output/GPS-Real_corrected_rectified_oriented_grid_*.png

# 7. Calibrate real-world measurements
python tools/Tool_6_Real_World_Calibrator.py data/GPS-Real_corrected_rectified_oriented_grid.json

# 8. Point mapper viewer (optional)
python tools/Tool_7_Point_Mapper.py

# 9. Create standalone API package
python tools/Tool_8_GPS_Overlay.py
```

---

## Standalone GPSOverlay API

For production use, Tool 8 creates a standalone API package that can be exported and used independently.

### Export and Usage
**Files to Export:**
- `src/tools/gps_overlay.py` - Standalone API code
- `data/gps_overlay.json` - Calibration data (created by Tool 8)

**Integration:**
```python
# In your application code
from gps_overlay import GPSOverlay

# Initialize API
overlay = GPSOverlay("gps_overlay.json")

# Transform GPS server coordinates
x_rect, y_rect = overlay.map_coords(50, 50)  # GPS → rectified

cell = overlay.get_grid_cell(50, 50)  # GPS → grid cell
# {"col": 12, "row": 8, "in_bounds": True}

real_pos = overlay.get_real_coords(50, 50)  # GPS → real-world mm
# {"x_mm": 1250.5, "y_mm": 890.2, "distance_from_origin_mm": 1523.1}
```

**Features:**
- **Simple functions:** `map_coords()`, `get_grid_cell()`, `get_real_coords()`, `get_grid_map()`
- **No dependencies:** Uses only built-in Python libraries
- **Real-world support:** Millimeter coordinates when calibrated
- **Grid mapping:** Complete grid array for navigation (size from Tool 4)

## Key Features

- **Interactive Processing:** All tools support zoom/pan for precise point placement
- **Modular Design:** Each tool focuses on one processing step
- **Automatic File Detection:** Tools auto-detect related JSON files
- **Comprehensive Output:** Images to `output/`, data to `data/`
- **Real-world Ready:** Complete pipeline from raw image to calibrated coordinates
- **Quality Control:** Visual feedback and statistical analysis throughout

## Technical Details

- **Fisheye Correction:** Uses OpenCV fisheye calibration with k1-k4 distortion coefficients
- **Homography:** Perspective transformation for arena rectification
- **Grid Generation:** "Close-enough" tiling with distributed rounding errors
- **Coordinate Systems:** Careful mapping between screen, image, and world coordinates
- **File Formats:** PNG images, JSON metadata, standardized naming conventions