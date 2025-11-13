# Under-The-Sea Arena Processing Pipeline

A comprehensive Python toolkit for processing fisheye images of underwater arenas, correcting distortion, detecting boundaries, rectifying to top-down views, and creating coordinate grids for robot navigation.

## Project Structure

```
├── tools/                  # Processing tools (Tool 1-8)
├── src/                    # Programmatic APIs and shared utilities
├── data/                   # JSON data files (calibrations, transforms, grids)
├── output/                 # Processed images
├── images/                 # Input images
├── tests/                  # Test files and validation scripts
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

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
**Purpose:** Convert rectified grid pixels into real-world centimetres using Tool 5 corner data.

**Usage:**
```bash
python tools/Tool_6_Real_World_Calibrator.py data/GPS-Real_corrected_rectified_oriented_grid_{cols}x{rows}.json
```

**Process:**
1. Loads individual grid settings and derives rectified arena corners
2. Calculates pixel spans for each wall and both diagonals
3. Guides you through four corner-to-corner wall measurements (TL→TR, TR→BR, BR→BL, BL→TL) in millimetres
4. Computes cm↔px ratios, prints summary stats, and (optionally) opens the rectified image while streaming the Top-Left→cursor distance in millimetres to the terminal
5. After inspection, prompts you to press `s` to write the calibration JSON (anything else cancels)

**Outputs:**
- `data/GPS-Real_corrected_rectified_oriented_calibration.json` – Pixel↔cm conversion stats, user-supplied wall lengths, derived diagonals, and assumptions

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

### Tool 8: GPSOverlay API Creator (`tools/Tool_8_GPS_Overlay.py`)
**Purpose:** Consolidates all calibration data into a single standalone JSON file for production API usage.

**Usage:**
```bash
# Create standalone API package
python tools/Tool_8_GPS_Overlay.py

# Custom output location
python tools/Tool_8_GPS_Overlay.py --output path/to/gps_overlay.json

# With real-world calibration (after running Tool 6)
python tools/Tool_8_GPS_Overlay.py --calibration-json data/GPS-Real_corrected_rectified_oriented_calibration.json
```

**Process:**
1. Loads fisheye calibration data from existing JSON files
2. Dynamically detects grid configuration (uses first/selected grid)
3. Includes real-world calibration if available (from Tool 6)
4. Creates single `gps_overlay.json` file for standalone API usage

**Output:**
- `data/gps_overlay.json` - Complete calibration data in single file
- Ready for export to production systems

**Notes:**
- Run after completing Tools 1-7
- For real-world coordinates, run Tool 6 first to generate calibration data
- Output file is completely self-contained for standalone API usage
- Tool 4 now uses simple naming: `GPS-Real_grid.json` (no grid size in filename)
- Grid size is detected dynamically from the saved configuration
- No legacy filename parsing - all configuration stored in JSON files

---

## Standalone GPSOverlay API

For production use, export the standalone API package created by Tool 8.

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
- **Grid mapping:** Complete 45×29 grid array for navigation

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


## Complete Workflow Example

### Manual Step-by-Step Processing
```bash
# 1. Correct fisheye distortion
python3 tools/Tool_1_Fix_Fisheye.py /home/thomas/Dev/Python/Mermaid/images/GPS-Real.png

# 2. Detect arena corners
python3 tools/Tool_2_Detect_Arena_Corners.py /home/thomas/Dev/Python/Mermaid/output/GPS-Real_corrected.png

# 3. Rectify to top-down view (prefer JSON)
python3 tools/Tool_3_Rectify_Arena_Square.py /home/thomas/Dev/Python/Mermaid/data/GPS-Real_corrected_corners.json

# 4. Create coordinate grids
python tools/Tool_4_Grid_Overlay.py output/GPS-Real_corrected_rectified_oriented.png

# 5. Inspect grid cells
python tools/Tool_5_Grid_Inspector.py output/GPS-Real_grid.png

# 6. Calibrate real-world measurements
python tools/Tool_6_Real_World_Calibrator.py data/GPS-Real_grid.json

# 7. Create standalone API package
python tools/Tool_8_GPS_Overlay.py --output data/gps_overlay.json
```

## Standalone GPSOverlay API

For production use, Tool 8 creates a standalone API package that can be exported and used independently of the full toolkit.

### Tool 8: GPSOverlay API Creator (`tools/Tool_8_GPS_Overlay.py`)
**Purpose:** Consolidates all calibration data into a single JSON file for standalone API usage.

**Usage:**
```bash
# Create standalone API package
python tools/Tool_8_GPS_Overlay.py

# Custom output location
python tools/Tool_8_GPS_Overlay.py --output path/to/gps_overlay.json
```

**Process:**
1. Loads fisheye calibration data from existing JSON files
2. Extracts 45×29 grid configuration
3. Includes real-world calibration if available (from Tool 6)
4. Creates single `gps_overlay.json` file for API usage

**Output:**
- `gps_overlay.json` - Complete calibration data in single file

### GPSOverlay API (`src/tools/gps_overlay.py`)
**Purpose:** Standalone Python API for GPS coordinate transformations. Can be exported and used independently.

**Features:**
- **Simple functions:** `map_coords()`, `get_grid_cell()`, `get_real_coords()`, `get_grid_map()`
- **No dependencies:** Uses only built-in Python libraries
- **Single JSON file:** All calibration data in one file
- **Dynamic grid detection:** Automatically uses grid size from Tool 4
- **Real-world support:** Millimeter coordinates when calibrated

**Usage:**
```python
from gps_overlay import GPSOverlay

# Load calibration data
overlay = GPSOverlay("gps_overlay.json")

# Transform GPS server coordinates (2048×1536)
x_rect, y_rect = overlay.map_coords(50, 50)  # → rectified coordinates

cell = overlay.get_grid_cell(50, 50)  # → grid cell info
# {"col": 4, "row": 2, "in_bounds": True, "center_x": 450.5, "center_y": 320.2}

real_pos = overlay.get_real_coords(50, 50)  # → real-world mm (if calibrated)
# {"x_mm": 1250.5, "y_mm": 890.2, "distance_from_origin_mm": 1523.1}

# Get complete grid mapping (size from Tool 4)
grid_map = overlay.get_grid_map()  # Array of all grid cells (e.g., 14×9)
```

**Export for Standalone Use:**
1. Copy `src/tools/gps_overlay.py` and `data/gps_overlay.json` to your project
2. Import and use: `from gps_overlay import GPSOverlay`
3. No additional dependencies required

**Testing:**
```bash
# Run standalone tests (requires test files)
python tests/test_gps_overlay_standalone.py

# Or run the built-in API test
python src/tools/gps_overlay.py
```

**Integration:**
```python
# In your application code
from gps_overlay import GPSOverlay

# Initialize once
overlay = GPSOverlay()

# Transform coordinates as needed
for gps_point in gps_stream:
    cell = overlay.get_grid_cell(gps_point.x, gps_point.y)
    if cell["in_bounds"]:
        handle_grid_cell(cell["col"], cell["row"])
```

### Automated Pipeline (Planned)
A unified pipeline in `main.py` will run all tools sequentially, allowing you to process an image from start to finish with a single command:

```bash
# Process entire pipeline automatically
python main.py images/GPS-Real.png

# With optional parameters
python main.py images/GPS-Real.png --output-dir ./results
```

This automated pipeline will:
- Run Tools 1-8 in sequence
- Handle intermediate file naming automatically
- Provide progress feedback
- Allow customization of grid parameters
- Generate a complete processing report including standalone API package

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