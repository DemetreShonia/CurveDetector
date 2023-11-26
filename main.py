from PIL import Image
import matplotlib.pyplot as plt
import random

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

random_points = random.sample(points, 200)

# Display the resulting image with detected points
plt.imshow(combined_edges, cmap='gray')
plt.title('Detected Edges with Fewer Points')
plt.axis('off')  # Hide axes

# Plot the randomly selected 100 points on the graph
for point in random_points:
    plt.scatter(point[0], point[1], color='red', s=2)  # Adjust the size 's' as needed for point markers

plt.show()