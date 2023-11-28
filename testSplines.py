import matplotlib.pyplot as plt

# Given points array
# Assuming points is a list of (x, y) coordinates
# Define a longer points array with an irregular pattern
# Define a longer points array with an aggressive and curvy pattern
points = [
    (236, 632),
    (222, 636),
    (203, 614),
    (185, 597),
    (174, 578),
    (167, 560),
    (166, 542)
    # Add more points as needed...
]



# Implementing spline interpolation using piecewise polynomials (cubic splines)
def cubic_spline(points):
    n = len(points)
    h = [points[i + 1][0] - points[i][0] for i in range(n - 1)]
    alpha = [((3 / h[i]) * (points[i + 1][1] - points[i][1])) - ((3 / h[i - 1]) * (points[i][1] - points[i - 1][1])) for i in range(1, n - 1)]

    l = [1] * n
    mu = [0] * n
    z = [0] * n

    for i in range(1, n - 1):
        l[i] = 2 * (points[i + 1][0] - points[i - 1][0]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i - 1] - h[i - 1] * z[i - 1]) / l[i]

    b = [0] * n
    c = [0] * n
    d = [0] * n

    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (points[j + 1][1] - points[j][1]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    def spline(x):
        for i in range(n - 1):
            if points[i][0] <= x <= points[i + 1][0]:
                dx = x - points[i][0]
                return points[i][1] + b[i] * dx + c[i] * dx ** 2 + d[i] * dx ** 3

    return spline

spline_function = cubic_spline(points)
x_values = list(range(min(points, key=lambda p: p[0])[0], max(points, key=lambda p: p[0])[0] + 1))
spline_interpolated_points = [(x, spline_function(x)) for x in x_values]

# Plotting the spline interpolated curve
plt.figure(figsize=(8, 6))
plt.scatter(*zip(*points), color='red', label='Original Points')
if spline_interpolated_points:
    plt.plot([p[0] for p in spline_interpolated_points], [p[1] for p in spline_interpolated_points], label='Spline Interpolated Curve')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Curve Approximation by Cubic Splines')
plt.legend()
plt.grid(True)
plt.show()
