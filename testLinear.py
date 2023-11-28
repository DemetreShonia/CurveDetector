import matplotlib.pyplot as plt
import numpy as np

# Given points array
# Assuming points is a list of (x, y) coordinates
# Define a longer points array
# Define a longer points array with an aggressive and curvy pattern
# Define a longer points array with an aggressive and curvy pattern
points = [
    (1, 2), (2, 9), (3, 4), (4, 16), (5, 7),
    (6, 25), (7, 10), (8, 36), (9, 13), (10, 49),
    (11, 16), (12, 64), (13, 19), (14, 81), (15, 22),
    (16, 100), (17, 25), (18, 121), (19, 28), (20, 144)
    # Add more points as needed...
]




# Interpolation function using Lagrange polynomial
def lagrange_interpolation(points):
    def L(k, x):
        result = 1
        for j, p in enumerate(points):
            if j != k:
                result *= (x - points[j][0]) / (points[k][0] - points[j][0])
        return result

    def P(x):
        return sum(points[k][1] * L(k, x) for k in range(len(points)))

    return P

# Generate interpolated points
interpolated_points = []
if len(points) > 1:
    min_x = min(points, key=lambda p: p[0])[0]
    max_x = max(points, key=lambda p: p[0])[0]
    f = lagrange_interpolation(points)
    x_values = np.linspace(min_x, max_x, 100)
    interpolated_points = [(x, f(x)) for x in x_values]

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(*zip(*points), color='red', label='Original Points')
if interpolated_points:
    plt.plot(*zip(*interpolated_points), label='Interpolated Curve')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Curve Approximation by Interpolation')
plt.legend()
plt.grid(True)
plt.show()
