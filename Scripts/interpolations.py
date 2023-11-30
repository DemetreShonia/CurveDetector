import matplotlib.pyplot as plt
import numpy as np


class InterpolationStrategy:
    def interpolate(self, x, y):
        pass

class LinearPieceWiseInterpolation(InterpolationStrategy):
    def interpolate(self, x, y):
        for i in range(len(x) - 1):
            x_start, x_end = x[i], x[i + 1]
            y_start, y_end = y[i], y[i + 1]

            slope = (y_end - y_start) / (x_end - x_start)
            intercept = y_start - slope * x_start

            x_interpolated = np.linspace(x_start, x_end, 100)
            y_interpolated = slope * x_interpolated + intercept

            plt.plot(x_interpolated, y_interpolated, color='red')
    
        plt.scatter(x, y)

class LeastSquaresInterpolation(InterpolationStrategy):
    def interpolate(self, x, y):
        degree = 3
        A = np.vander(x, degree + 1, increasing=True)

        if np.linalg.matrix_rank(A) == A.shape[1]: 
            ATA = np.dot(A.T, A)
            ATy = np.dot(A.T, y)
            coeffs = np.linalg.solve(ATA, ATy)

            x_smooth = np.linspace(np.min(x), np.max(x), 100)
            A_smooth = np.vander(x_smooth, degree + 1, increasing=True)
            y_smooth = np.dot(A_smooth, coeffs)

            plt.scatter(x, y)
            plt.plot(x_smooth, y_smooth, color='red')
        else:
            print("Matrix is not full rank, skipping segment fitting.")

class CubicSplineInterpolation(InterpolationStrategy):
    def interpolate(self, x, y):
        def cubic_spline_interpolation(x, y):
            n = len(x)
            if n != len(y):
                print("Interpolation value is outside the range of provided data")


            h = np.diff(x)
            delta = np.diff(y) / h

            matrix = np.zeros((n, n))
            matrix[0, 0] = 1
            matrix[n - 1, n - 1] = 1

            for i in range(1, n - 1):
                matrix[i, i - 1] = h[i - 1]
                matrix[i, i] = 2 * (h[i - 1] + h[i])
                matrix[i, i + 1] = h[i]

            rhs = np.zeros(n)
            rhs[1:-1] = 3 * (delta[1:] - delta[:-1])

            c = np.linalg.solve(matrix, rhs)

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

        
        coefficients = cubic_spline_interpolation(x, y)
    
        min_x = np.min(x)
        max_x = np.max(x)
        
        num_interpolation_points = 200  # Increase the number of points for smoother plots
        interpolated_xs = np.linspace(min_x, max_x, num_interpolation_points)
        
        interpolated_values = []
        for x_interpolate in interpolated_xs:
            interpolated_value = cubic_spline_interpolate(x, coefficients, x_interpolate)
            interpolated_values.append(interpolated_value)
        
        plt.plot(x, y, 'o', label='Original Segment')
        plt.plot(interpolated_xs, interpolated_values, label='Interpolated Values')
        

class Interpolator:
    def __init__(self, strategy):
        self.strategy = strategy

    def set_strategy(self, strategy):
        self.strategy = strategy

    def interpolate(self, x, y):
        return self.strategy.interpolate(x, y)

