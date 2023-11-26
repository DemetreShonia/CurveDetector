from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np

# Load the image
img = Image.open('curve1.jpg').convert('L')  # Convert image to grayscale

# Convert the image to a matrix
pixels = img.load()
width, height = img.size

# Sobel operator kernels
sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

# Function to convolve the image with a kernel
def apply_kernel(image, kernel):
    new_image = Image.new('L', (width, height))
    new_pixels = new_image.load()
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            pixel = (
                kernel[0][0] * image.getpixel((x - 1, y - 1)) +
                kernel[0][1] * image.getpixel((x, y - 1)) +
                kernel[0][2] * image.getpixel((x + 1, y - 1)) +
                kernel[1][0] * image.getpixel((x - 1, y)) +
                kernel[1][1] * image.getpixel((x, y)) +
                kernel[1][2] * image.getpixel((x + 1, y)) +
                kernel[2][0] * image.getpixel((x - 1, y + 1)) +
                kernel[2][1] * image.getpixel((x, y + 1)) +
                kernel[2][2] * image.getpixel((x + 1, y + 1))
            )
            new_pixels[x, y] = int(pixel)
    return new_image

# Apply Sobel operator in both x and y directions
edges_x = apply_kernel(img, sobel_x)
edges_y = apply_kernel(img, sobel_y)

# Combine x and y edge images
combined_edges = Image.new('L', (width, height))
combined_pixels = combined_edges.load()
for y in range(height):
    for x in range(width):
        val = int(((edges_x.getpixel((x, y)) ** 2) + (edges_y.getpixel((x, y)) ** 2)) ** 0.5)
        combined_pixels[x, y] = val if val <= 255 else 255

# Detect points (high-intensity areas)
threshold = 200  # Adjust threshold as needed
points = []
for y in range(height):
    for x in range(width):
        if combined_edges.getpixel((x, y)) > threshold:
            points.append((x, y))

def kmeans(points, k, max_iterations=10):
    points_list = list(points)  # Convert points to a list explicitly
    centroids = random.sample(points_list, k)
    for _ in range(max_iterations):
        clusters = {i: [] for i in range(k)}

        # Assign each point to the nearest centroid
        for point in points_list:
            distances = [np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

        # Update centroids
        new_centroids = [np.mean(clusters[i], axis=0) for i in range(k)]

        # If centroids have not changed, stop
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids

# Convert points to a NumPy array
points_array = np.array(points)

# Apply custom KMeans clustering
num_clusters = 100  # Adjust the number of clusters as needed
cluster_centers = kmeans(points_array, num_clusters)

# Display the resulting image with detected points




def find_closest_point(current_point, remaining_points):
    min_distance = float('inf')
    closest_point = None
    closest_index = -1

    for i, point in enumerate(remaining_points):
        dist = np.linalg.norm(current_point - point)
        if dist < min_distance:
            min_distance = dist
            closest_point = point
            closest_index = i

    return closest_point, closest_index

def create_closed_loop(points):
    remaining_points = points.copy()
    ordered_points = [remaining_points.pop(0)]

    while remaining_points:
        current_point = ordered_points[-1]
        closest_point, closest_index = find_closest_point(current_point, remaining_points)

        if closest_point is not None:
            ordered_points.append(closest_point)
            remaining_points.pop(closest_index)
        else:
            break

    # Check if the loop is closed and add the starting point if needed
    if not np.array_equal(ordered_points[0], ordered_points[-1]):
        ordered_points.append(ordered_points[0])

    return ordered_points



ordered_points = create_closed_loop(cluster_centers)

# 'ordered_points' now contains the sequence of points forming the closed loop
ordered_points = np.array(ordered_points)
points = np.array(points)
# Plotting the points forming the closed loop
plt.figure(figsize=(8, 6))

def segment_points(points):
    segmented_arrays = []
    start_index = 0
    increasing = None

    for i in range(1, len(points)):
        if increasing is None:
            if points[i, 0] > points[i - 1, 0]:
                increasing = True
            elif points[i, 0] < points[i - 1, 0]:
                increasing = False

        if increasing is not None:
            if (increasing and points[i, 0] < points[i - 1, 0]) or (not increasing and points[i, 0] > points[i - 1, 0]):
                segmented_arrays.append(points[start_index:i])
                start_index = i
                increasing = not increasing

    # Add the last segment
    segmented_arrays.append(points[start_index:])

    return segmented_arrays

# Segmented arrays based on changes in x-coordinate
segmented_arrays = segment_points(ordered_points)

# Plotting each sub-array with a different color
for idx, segment in enumerate(segmented_arrays, start=1):
    plt.scatter(segment[:, 0], segment[:, 1], label=f'Segment {idx}')

plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Segments based on X-coordinate')
# plt.legend()
plt.show()




# gradient = np.linspace(0, 1, len(ordered_points))

# # Create a colormap using the gradient (from green to blue)
# colors = plt.cm.Blues(gradient)

# # Scatter plot with gradient colors
# scatter = plt.scatter(
#     [point[0] for point in ordered_points],
#     [point[1] for point in ordered_points],
#     color='white',  # Set color to white
#     edgecolor='white',  # Add black edges for better visibility
#     label='Ordered Points',
#     s = 10
# )


# for i, point in enumerate(ordered_points):
#     plt.text(point[0], point[1], str(i + 1), ha='center', va='center', fontsize=8)

# # plt.plot(ordered_points[:, 0], ordered_points[:, 1], color='red', linestyle='-', label='Closed Loop')
# plt.title('Closed Loop of Ordered Points')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.grid(True)
# plt.show()


# Plot the cluster centers as points on the graph
# plt.scatter([center[0] for center in cluster_centers], [center[1] for center in cluster_centers], color='red', s=2)  # Adjust the size 's' for point markers

print(cluster_centers)
# plt.show()
