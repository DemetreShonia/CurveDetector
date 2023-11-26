import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Function to generate points on a circle using parametric equations
def generate_circle_points(radius, num_points):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.column_stack((x, y))

# Function to fit a circle equation to the given points
def circle_equation(x, a, b, r):
    return (x[0] - a)**2 + (x[1] - b)**2 - r**2

# Generate points forming a circle
radius = 5
num_points = 200
circle_points = generate_circle_points(radius, num_points)

# Adding noise to the generated points (optional, for more realistic scenarios)
noise_magnitude = 0.1
cpoints = circle_points 

# Fit the circle equation to the points using curve fitting
popt, pcov = curve_fit(circle_equation, (cpoints[:, 0], cpoints[:, 1]), np.zeros(len(cpoints)))

# Extract the parameters of the fitted circle
a_fit, b_fit, r_fit = popt

print("Fitted circle equation parameters:")
print(f"a = {a_fit}, b = {b_fit}, r = {r_fit}")

# Divide the circle into two halves by angle
half_circle_points_1 = circle_points[:num_points // 2]
half_circle_points_2 = circle_points[num_points // 2:]

# Plotting the two halves of the circle separately
plt.figure(figsize=(8, 8))
plt.scatter(cpoints[:, 0], cpoints[:, 1], label='Noisy Circle Points')
plt.plot(half_circle_points_1[:, 0], half_circle_points_1[:, 1], color='red', label='Half 1', linewidth=2)
plt.plot(half_circle_points_2[:, 0], half_circle_points_2[:, 1], color='blue', label='Half 2', linewidth=2)
plt.axis('equal')
plt.title('Fitted Circle Approximation (Divided into Two Halves)')
plt.legend()
plt.show()
