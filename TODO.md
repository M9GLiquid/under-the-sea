# TODO List

## Remaining Tasks

### Tool 6: Real-World Calibrator
- [ ] Implement `tools/real_world_calibrator.py`
  - [ ] Load grid image and arena bounds from grids JSON
  - [ ] Click-to-measure pixel distances between two points
  - [ ] Input real-world measurements in centimeters
  - [ ] Calculate pixel-to-cm conversion ratios
  - [ ] Save calibration data to `data/NAME_calibration.json`
  - [ ] Multiple measurements for improved accuracy
  - [ ] Statistical analysis (average, min/max, standard deviation)

### Automated Pipeline
- [ ] Create unified pipeline in `main.py`
  - [ ] Run all 6 tools sequentially
  - [ ] Handle intermediate file naming automatically
  - [ ] Provide progress feedback
  - [ ] Allow customization of grid parameters
  - [ ] Generate complete processing report

### Optional Improvements
- [ ] Refactor code to eliminate duplicates
  - [ ] Extract common functions (file path guessing, arena bounds loading, etc.)
  - [ ] Create shared utility modules
  - [ ] Standardize error handling across tools
  - [ ] Consolidate similar mouse callback patterns

## Completed Tasks ✅
- [x] Tool 1: Fisheye Correction with zoom/pan
- [x] Tool 2: Arena Corner Detection with zoom/pan
- [x] Tool 3: Arena Rectification with orientation alignment
- [x] Tool 4: Grid Overlay with interactive controls
- [x] Tool 5: Grid Inspector with cell coordinate display
- [x] Updated README with complete pipeline documentation
