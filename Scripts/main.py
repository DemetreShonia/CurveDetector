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
        


def piecewise_linear_approximation(x, y):
    for i in range(len(x) - 1):
        x_start, x_end = x[i], x[i + 1]
        y_start, y_end = y[i], y[i + 1]

        slope = (y_end - y_start) / (x_end - x_start)
        intercept = y_start - slope * x_start

        x_interpolated = np.linspace(x_start, x_end, 100)
        y_interpolated = slope * x_interpolated + intercept

        plt.plot(x_interpolated, y_interpolated, color='red')
    
    plt.scatter(x, y)


def least_squares_approximation(x, y, degree=3):
    A = np.vander(x, degree + 1, increasing=True)

    # Check matrix rank
    if np.linalg.matrix_rank(A) == A.shape[1]:  # Check if matrix is full rank
        # Solve the normal equations to obtain coefficients
        ATA = np.dot(A.T, A)
        ATy = np.dot(A.T, y)
        coeffs = np.linalg.solve(ATA, ATy)

        # Generate points for smooth curve display
        x_smooth = np.linspace(np.min(x), np.max(x), 100)
        A_smooth = np.vander(x_smooth, degree + 1, increasing=True)
        y_smooth = np.dot(A_smooth, coeffs)

        # Plot each segment's points and the fitted polynomial curve
        plt.scatter(x, y)
        plt.plot(x_smooth, y_smooth, color='red')
    else:
        print("Matrix is not full rank, skipping segment fitting.")

def cubic_spline_interpolation(x, y):
    n = len(x)
    if n != len(y):
        print("Interpolation value is outside the range of provided data")


    h = np.diff(x)
    delta = np.diff(y) / h

    # Construct the tridiagonal matrix
    matrix = np.zeros((n, n))
    matrix[0, 0] = 1
    matrix[n - 1, n - 1] = 1

    for i in range(1, n - 1):
        matrix[i, i - 1] = h[i - 1]
        matrix[i, i] = 2 * (h[i - 1] + h[i])
        matrix[i, i + 1] = h[i]

    rhs = np.zeros(n)
    rhs[1:-1] = 3 * (delta[1:] - delta[:-1])

    # Solve the linear system to find the coefficients
    c = np.linalg.solve(matrix, rhs)

    # Calculate the remaining coefficients
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    for i in range(n - 1):
        b[i] = delta[i] - h[i] * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    coefficients = []
    for i in range(n - 1):
        coefficients.append((y[i], b[i], c[i], d[i]))

    return coefficients

def cubic_spline_interpolate(x, coefficients, x_interpolate):
    for i in range(len(coefficients)):
        if x_interpolate >= x[i] and x_interpolate <= x[i + 1]:
            a, b, c, d = coefficients[i]
            t = (x_interpolate - x[i]) / (x[i + 1] - x[i])
            interpolated_value = a + b * (x_interpolate - x[i]) + c * (x_interpolate - x[i]) ** 2 + d * (x_interpolate - x[i]) ** 3
            return interpolated_value

    print("Interpolation value is outside the range of provided data")

for segment in segmented_arrays:
    segment = segment[np.argsort(segment[:, 0])]
    x = segment[:, 0]
    y = segment[:, 1]
   

    coefficients = cubic_spline_interpolation(x, y)
    
    # Interpolate values within the current segment
    min_x = np.min(x)
    max_x = np.max(x)
    
    num_interpolation_points = 100  # Increase the number of points for smoother plots
    interpolated_xs = np.linspace(min_x, max_x, num_interpolation_points)
    
    interpolated_values = []
    for x_interpolate in interpolated_xs:
        interpolated_value = cubic_spline_interpolate(x, coefficients, x_interpolate)
        interpolated_values.append(interpolated_value)
    
    # Plot the original segment and the interpolated values
    plt.plot(x, y, 'o', label='Original Segment')
    plt.plot(interpolated_xs, interpolated_values, label='Interpolated Values')

    # piecewise_linear_approximation(x,y)



plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Segments based on X-coordinate')
plt.show()

# ... (additional parts of the script)
