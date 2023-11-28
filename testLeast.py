import matplotlib.pyplot as plt
import numpy as np

# Given points array
points = [
    (1, 2), (2, 9), (3, 4), (4, 16), (5, 7),
    (6, 25), (7, 10), (8, 36), (9, 13), (10, 49),
    (11, 16), (12, 64), (13, 19), (14, 81), (15, 22),
    (16, 100), (17, 25), (18, 121), (19, 28), (20, 144)
    # Add more points as needed...
]

# Extract x and y coordinates from points
x_values = [p[0] for p in points]
y_values = [p[1] for p in points]

# Set a fixed seed for reproducibility
np.random.seed(42)  # Change seed value for different results

# Perform polynomial fitting using least squares method
degree_of_polynomial = 3  # Choose the degree of polynomial (adjust as needed)
coefficients = np.polyfit(x_values, y_values, degree_of_polynomial)
poly_function = np.poly1d(coefficients)

# Generate points for the fitted polynomial curve
x_fit = np.linspace(min(x_values), max(x_values), 100)
y_fit = poly_function(x_fit)

# Plotting
plt.figure(figsize=(8, 6))
plt.scatter(x_values, y_values, color='red', label='Original Points')
plt.plot(x_fit, y_fit, label=f'Fitted Curve (Degree {degree_of_polynomial})', color='blue')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Curve Approximation by Polynomial Fitting (Least Squares)')
plt.legend()
plt.grid(True)
plt.show()
