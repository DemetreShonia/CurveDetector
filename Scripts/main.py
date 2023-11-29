from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from image_processing import ImageProcessor
from edge_detection import EdgeDetection
from clustering import kmeans, find_closest_point, create_closed_loop, segment_points

# Load the image and perform edge detection
img_processor = ImageProcessor('curve3.jpg')
pixels = img_processor.get_image_matrix()
width, height = img_processor.get_image_size()

sobel_x, sobel_y = EdgeDetection.sobel_operator()
edges_x = img_processor.apply_kernel(sobel_x)
edges_y = img_processor.apply_kernel(sobel_y)

combined_edges = EdgeDetection.combine_edges(edges_x, edges_y, width, height)

# Detect points (high-intensity areas)
threshold = 200
points = []
for y in range(height):
    for x in range(width):
        if combined_edges.getpixel((x, y)) > threshold:
            points.append((x, y))

# Convert points to a NumPy array
points_array = np.array(points)

# Apply custom KMeans clustering
num_clusters = 100
cluster_centers = kmeans(points_array, num_clusters)

# Find the ordered points forming a closed loop
ordered_points = create_closed_loop(cluster_centers)

# Plotting the points forming the closed loop
plt.figure(figsize=(8, 6))
segmented_arrays = segment_points(np.array(ordered_points))
for idx, segment in enumerate(segmented_arrays, start=1):
    plt.scatter(segment[:, 0], segment[:, 1], label=f'Segment {idx}')


    for j, point in enumerate(segment):
        plt.text(point[0], point[1], str(j + 1), ha='center', va='center', fontsize=8)
        
from interpolations import Interpolator, LinearPieceWiseInterpolation, CubicSplineInterpolation, LeastSquaresInterpolation

# Example usage
linear_strategy = LinearPieceWiseInterpolation()
least_squares_strategy = LeastSquaresInterpolation()
cubic_spline_strategy = CubicSplineInterpolation()

# Initialize the Interpolator with a default strategy
interpolator = Interpolator(cubic_spline_strategy)



for segment in segmented_arrays:
    segment = segment[np.argsort(segment[:, 0])]
    x = segment[:, 0]
    y = segment[:, 1]


    interpolator.interpolate(x, y) # returns values too


plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Segments based on X-coordinate')
plt.show()

