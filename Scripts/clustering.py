import random
import numpy as np

def kmeans(points, k, max_iterations=10):
    points_list = list(points) 
    centroids = random.sample(points_list, k)
    for _ in range(max_iterations):
        clusters = {i: [] for i in range(k)}

        for point in points_list:
            distances = [np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(point)

        new_centroids = [np.mean(clusters[i], axis=0) for i in range(k)]

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids

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

    if not np.array_equal(ordered_points[0], ordered_points[-1]):
        ordered_points.append(ordered_points[0])

    return ordered_points



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

    segmented_arrays.append(points[start_index:])

    return segmented_arrays

