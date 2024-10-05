# Numerical Programming Project

### Overview
This project extracts curve-related data from hand-drawn images, interpolates them using 3 different methods, and plots the graphs. All core implementations, including edge detection, interpolation, and clustering, are custom-written from scratch.

### Project Structure
- **Images/**: Contains 4 different curve images to work on.
- **Scripts/**:
  - `interpolations.py`: Implements interpolation algorithms using the Strategy Design Pattern.
  - `edge_detection.py`: Custom Sobel edge detection.
  - `clustering.py`: K-Means clustering and helper functions.
  - `image_processing.py`: Image handling functions, including Sobel matrix application.
  - `path_fixer.py`: Fixes path intersections.
  - `main.py`: Runs the project. Switch interpolation algorithms and images easily.

### Features
- Custom edge detection for extracting unordered points from images.
- K-Means clustering for point reduction.
- Path fixing for handling intersections using dot product direction checks.
- Curve segmentation and three interpolation methods:
  - Piecewise Linear Approximation
  - Least Square Approximation
  - Cubic Splines Approximation

### How to Run
1. Place your image in the `Images/` folder.
2. Modify `main.py` to choose an interpolation method and image path.
3. Run `main.py` to process and plot the graph.

### Notes
- Interpolations are segmented for better visualization.
- Works on both closed and non-closed graphs.
