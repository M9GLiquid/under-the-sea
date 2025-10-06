# Under-The-Sea Arena Processing Pipeline

A comprehensive Python toolkit for processing fisheye images of underwater arenas, correcting distortion, detecting boundaries, rectifying to top-down views, and creating coordinate grids for robot navigation.

## Project Structure

```
├── tools/                  # Processing tools (Tool 1-6)
├── data/                   # JSON data files (calibrations, transforms, grids)
├── output/                 # Processed images
├── images/                 # Input images
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Complete Processing Pipeline

The toolkit consists of 6 sequential tools that transform raw fisheye images into calibrated, grid-based coordinate systems:

### Tool 1: Fisheye Correction (`tools/fix_fisheye.py`)
**Purpose:** Correct fisheye lens distortion in underwater arena images.

**Usage:**
```bash
python tools/fix_fisheye.py images/YOUR_IMAGE.png
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

### Tool 2: Arena Corner Detection (`tools/detect_arena_corners.py`)
**Purpose:** Detect the four corners of the arena from the corrected image.

**Usage:**
```bash
python tools/detect_arena_corners.py output/GPS-Real_corrected.png
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

### Tool 3: Arena Rectification (`tools/rectify_arena_square.py`)
**Purpose:** Transform the arena to a top-down, orientation-aligned rectangle.

**Usage:**
```bash
python tools/rectify_arena_square.py output/GPS-Real_corrected.png
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

### Tool 4: Grid Overlay (`tools/grid_overlay.py`)
**Purpose:** Create coordinate grids within the rectified arena bounds.

**Usage:**
```bash
python tools/grid_overlay.py output/GPS-Real_rectified_oriented.png
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
- Batch saving: saves all grids meeting criteria (≥30px cells, ≤1/3 arena span)
- Visual feedback with dashed red grid lines

**Outputs:**
- `output/GPS-Real_grid_{cols}x{rows}.png` - Grid overlay images
- `data/GPS-Real_grids.json` - Grid manifest with all configurations

---

### Tool 5: Grid Inspector (`tools/grid_inspector.py`)
**Purpose:** Inspect saved grid images and view cell coordinates.

**Usage:**
```bash
python tools/grid_inspector.py output/GPS-Real_grid_8x6.png
```

**Process:**
1. Loads grid image and arena bounds
2. Parses grid dimensions from filename
3. Displays cell coordinates on mouse hover

**Features:**
- Hover over cells to see coordinates (e.g., "Cell (0, 0)")
- Automatic arena bounds detection from grids JSON
- Clean window close handling

---

### Tool 6: Real-World Calibrator (`tools/real_world_calibrator.py`)
**Purpose:** Convert rectified grid pixels into real-world centimetres using Tool 5 corner data.

**Usage:**
```bash
python tools/real_world_calibrator.py data/GPS-Real_corrected_rectified_oriented_grids.json
```

**Process:**
1. Loads Tool 5 `*_grids.json` manifest and derives rectified arena corners
2. Calculates pixel spans for each wall and both diagonals
3. Guides you through four corner-to-corner wall measurements (TL→TR, TR→BR, BR→BL, BL→TL) in millimetres
4. Computes cm↔px ratios, prints summary stats, and (optionally) opens the rectified image while streaming the Top-Left→cursor distance in millimetres to the terminal
5. After inspection, prompts you to press `s` to write the calibration JSON (anything else cancels)

**Outputs:**
- `data/GPS-Real_corrected_rectified_oriented_calibration.json` – Pixel↔cm conversion stats, user-supplied wall lengths, derived diagonals, and assumptions

---

## Complete Workflow Example

### Manual Step-by-Step Processing
```bash
# 1. Correct fisheye distortion
python tools/fix_fisheye.py images/GPS-Real.png

# 2. Detect arena corners
python tools/detect_arena_corners.py output/GPS-Real_corrected.png

# 3. Rectify to top-down view
python tools/rectify_arena_square.py output/GPS-Real_corrected.png

# 4. Create coordinate grids
python tools/grid_overlay.py output/GPS-Real_rectified_oriented.png

# 5. Inspect grid cells
python tools/grid_inspector.py output/GPS-Real_grid_8x6.png

# 6. Calibrate real-world measurements
python tools/real_world_calibrator.py data/GPS-Real_corrected_rectified_oriented_grids.json
```

### Automated Pipeline (Planned)
When Tool 6 is implemented, a unified pipeline will be created in `main.py` that runs all 6 tools sequentially, allowing you to process an image from start to finish with a single command:

```bash
# Process entire pipeline automatically
python main.py images/GPS-Real.png

# With optional parameters
python main.py images/GPS-Real.png --grid-size 8x6 --output-dir ./results
```

This automated pipeline will:
- Run Tools 1-6 in sequence
- Handle intermediate file naming automatically
- Provide progress feedback
- Allow customization of grid parameters
- Generate a complete processing report

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