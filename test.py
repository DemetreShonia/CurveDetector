import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

def find_closest_point(current_point, remaining_points):
    min_distance = float('inf')
    closest_point = None
    closest_index = -1

    for i, point in enumerate(remaining_points):
        dist = distance.euclidean(current_point, point)
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

# Your 'points' list here...
import numpy as np

points = [
    np.array([362.25714286, 855.62857143]), np.array([288.109375, 801.65625]), np.array([607.65671642, 694.86567164]),
    np.array([501.07407407, 861.14814815]), np.array([540.47619048, 534.9047619]), np.array([492.96969697, 469.57575758]),
    np.array([ 97.2173913 , 460.17391304]), np.array([297.45454545, 226.92727273]), np.array([341.5       , 199.13793103]),
    np.array([532.06666667, 525.11111111]), np.array([524.36666667, 850.63333333]), np.array([ 96.11111111, 448.33333333]),
    np.array([447.64102564, 871.51282051]), np.array([270.53623188, 778.39130435]), np.array([203.83333333, 598.33333333]),
    np.array([162.18055556, 546.75]), np.array([376.9375 , 861.28125]), np.array([205.70707071, 283.16161616]),
    np.array([368.80597015, 186.40298507]), np.array([206.05263158, 605.05263158]), np.array([209.71428571, 614.19047619]),
    np.array([126.61904762, 359.82539683]), np.array([104.8 , 404.55]), np.array([254.87272727, 746.27272727]),
    np.array([183.95454545, 568.52272727]), np.array([305.67924528, 820.33962264]), np.array([201.25, 593.16666667]),
    np.array([226.05263158, 668.55263158]), np.array([579.64285714, 582.5]), np.array([198.41176471, 589.23529412]),
    np.array([469.29508197, 370.78688525]), np.array([428.95121951, 872.2195122]), np.array([505.41304348, 490.04347826]),
    np.array([217.72413793, 636.55172414]), np.array([170.51685393, 303.40449438]), np.array([568.7826087 , 568.91304348]),
    np.array([592.54285714, 605.6]), np.array([101.45454545, 480.59090909]), np.array([146.94805195, 534.50649351]),
    np.array([232.4893617 , 690.63829787]), np.array([471.48148148, 412.18518519]), np.array([556.38461538, 549.53846154]),
    np.array([392.94736842, 865.92105263]), np.array([475.28125   , 247.08333333]), np.array([499.35714286, 478.54761905]),
    np.array([241.92592593, 715.7962963]), np.array([145.48275862, 331.18390805]), np.array([585.625 , 591.5625]),
    np.array([ 99.31818182, 470.59090909]), np.array([110.78431373, 500.11764706]), np.array([174.89655172, 559.05172414]),
    np.array([132.23611111, 522.27777778]), np.array([565.65217391, 563.43478261]), np.array([490.        , 864.58333333]),
    np.array([469.        , 395.76744186]), np.array([221.5       , 651.46666667]), np.array([104.63888889, 490.72222222]),
    np.array([478.35789474, 296.17894737]), np.array([606.77192982, 665.61403509]), np.array([549.31428571, 542.4]),
    np.array([213.86363636, 625.]), np.array([474.03846154, 424.15384615]), np.array([478.10714286, 867.71428571]),
    np.array([512.5625, 856.5625]), np.array([195.36842105, 584.21052632]), np.array([561.91666667, 556.41666667]),
    np.array([319.05882353, 211.39215686]), np.array([473.7804878 , 337.80487805]), np.array([482.13333333, 445.93333333]),
    np.array([336.22222222, 842.77777778]), np.array([573.08, 574.64]), np.array([243.27777778, 265.27777778]),
    np.array([580.09302326, 803.74418605]), np.array([322.05128205, 833.53846154]), np.array([113.86792453, 383.60377358]),
    np.array([602.30232558, 641.88372093]), np.array([546.51724138, 837.03448276]), np.array([568.        , 819.16216216]),
    np.array([484.6 , 453.15]), np.array([ 99.90625, 421.34375]), np.array([523.25454545, 513.89090909]),
    np.array([601.23728814, 754.50847458]), np.array([591.53191489, 781.93617021]), np.array([120.09836066, 510.36065574]),
    np.array([ 96.17857143, 435.78571429]), np.array([191.0625, 577.5]), np.array([476.66666667, 433.0952381]),
    np.array([606.02898551, 724.7826087]), np.array([513.43396226, 502.32075472]), np.array([478.76190476, 439.9047619]),
    np.array([454.08064516, 202.24193548]), np.array([557.25, 828.875]), np.array([464.11764706, 869.35294118]),
    np.array([274.375, 245.578125]), np.array([598.66666667, 622.58974359]), np.array([487.5862069 , 460.65517241]),
    np.array([349.72, 850.2]), np.array([535.78571429, 844.17857143]), np.array([410.87804878, 870.2195122]),
    np.array([407.96825397, 180.38095238])
]


ordered_points = create_closed_loop(points)

# 'ordered_points' now contains the sequence of points forming the closed loop
ordered_points = np.array(ordered_points)
points = np.array(points)
# Plotting the points forming the closed loop
plt.figure(figsize=(8, 6))

gradient = np.linspace(0, 1, len(ordered_points))

# Create a colormap using the gradient (from green to blue)
colors = plt.cm.Blues(gradient)

# Scatter plot with gradient colors
plt.scatter(
    [point[0] for point in ordered_points],
    [point[1] for point in ordered_points],
    color=colors,
    label='Ordered Points'
)

# plt.plot(ordered_points[:, 0], ordered_points[:, 1], color='red', linestyle='-', label='Closed Loop')
plt.title('Closed Loop of Ordered Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()