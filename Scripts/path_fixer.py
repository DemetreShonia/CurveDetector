import numpy as np

class PathFixer:
    @staticmethod
    def normalize_vector(vector):
        magnitude = np.linalg.norm(vector)
        return vector / magnitude if magnitude != 0 else vector

    @staticmethod
    def get_direction_vector(point1, point2):
        vec = np.array(point1) - np.array(point2)
        return PathFixer.normalize_vector(vec)

    @staticmethod
    def get_points_in_radius(point, radius, ordered_points):
        points_in_radius = []
        for i, p in enumerate(ordered_points):
            if np.linalg.norm(point - p) < radius:
                points_in_radius.append((p, i))
        return points_in_radius

    def fix_ordering(self, ordered_points, radius):
        op = ordered_points.copy()
        n = len(ordered_points)
        path = [ordered_points[0], ordered_points[1]]

        for i in range(2, n):
            current_point = ordered_points[i]
            prev_point = path[-1]
            prev_dir_vector = self.get_direction_vector(prev_point, path[-2])
            
            points_in_radius = self.get_points_in_radius(current_point, radius, ordered_points)
            
            max_dot_product = -np.inf
            best_direction = None
            best_point_id = None
            
            for point, point_id in points_in_radius:
                if point_id <= i - 2 or point_id >= i:
                    continue
                
                direction_vector = self.get_direction_vector(point, current_point)
                if direction_vector is None:
                    continue
                
                dot_product = np.dot(prev_dir_vector, direction_vector)
                if dot_product > max_dot_product:
                    max_dot_product = dot_product
                    best_direction = direction_vector
                    best_point_id = point_id
            
            if best_direction is not None and best_point_id is not None:
                ordered_points[i], ordered_points[best_point_id] = ordered_points[best_point_id], ordered_points[i]
                path.append(ordered_points[i])
        
        return op
