# TODO List

## Remaining Tasks

### Automated Pipeline
- [ ] Create unified pipeline in `main.py`
  - [ ] Run all 6 tools sequentially
  - [ ] Handle intermediate file naming automatically
  - [ ] Provide progress feedback
  - [ ] Allow customization of grid parameters
  - [ ] Generate complete processing report

### Shared Commons Refactor
- [ ] Establish shared modules for all tools
  - [ ] Implement `src/tools/io.py` for arena bundle load/save helpers
  - [ ] Implement `src/tools/cli.py` for common argument parsing and entrypoints
  - [ ] Implement `src/tools/images.py` for OpenCV window and image utilities
  - [ ] Implement `src/tools/geometry.py` for reusable math helpers

### Tool 1 Refactor
- [ ] Replace ad-hoc JSON handling with shared IO helpers
- [ ] Use shared CLI scaffolding
- [ ] Persist fisheye outputs into unified arena bundle structure

### Tool 2 Refactor
- [ ] Load/save via shared arena bundle
- [ ] Reuse common image viewer utilities
- [ ] Store detected bounds in standardized format

### Tool 3 Refactor
- [ ] Consume shared arena bundle inputs (bounds, fisheye data)
- [ ] Leverage geometry helpers for homography math
- [ ] Record rectified image metadata back into bundle

### Tool 4 Refactor
- [ ] Pull rectified data using shared IO modules
- [ ] Use common grid/geometry helpers for mesh generation
- [ ] Append generated grids to bundle manifest

### Tool 5 Refactor
- [ ] Inspect grids directly from bundle manifest
- [ ] Standardize viewer with shared image utilities
- [ ] Persist selected grid metadata in bundle

### Tool 6 Refactor
- [ ] Switch to shared IO/CLI modules completely
- [ ] Reuse geometry helpers for span calculations
- [ ] Store calibration metrics inside bundle and remove legacy files

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
- [x] Tool 6: [px] -> [mm] convertion.
- [x] Updated README with complete pipeline documentation
