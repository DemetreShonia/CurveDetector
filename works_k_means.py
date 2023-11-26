from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np

# Load the image
img = Image.open('handwritten_curve.jpg').convert('L')  # Convert image to grayscale

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

def kmeans(points, k, max_iterations=100):
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
num_clusters = 200  # Adjust the number of clusters as needed
cluster_centers = kmeans(points_array, num_clusters)

# Display the resulting image with detected points
plt.imshow(combined_edges, cmap='gray')
plt.title('Detected Edges with Reduced Points using Custom KMeans')
plt.axis('off')  # Hide axes

# Plot the cluster centers as points on the graph
plt.scatter([center[0] for center in cluster_centers], [center[1] for center in cluster_centers], color='red', s=2)  # Adjust the size 's' for point markers

plt.show()
