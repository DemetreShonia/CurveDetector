from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from image_processing import ImageProcessor
from edge_detection import EdgeDetection
from clustering import kmeans, create_closed_loop, segment_points
from path_fixer import PathFixer
from interpolations import Interpolator

class Main:
    def __init__(self, image_path, threshold=200, num_clusters=100, radius=10):
        self.image_path = image_path
        self.threshold = threshold
        self.num_clusters = num_clusters
        self.radius = radius

    def process_image(self):
        img_processor = ImageProcessor(self.image_path)
        pixels = img_processor.get_image_matrix()
        width, height = img_processor.get_image_size()

        sobel_x, sobel_y = EdgeDetection.sobel_operator()
        edges_x = img_processor.apply_kernel(sobel_x)
        edges_y = img_processor.apply_kernel(sobel_y)

        combined_edges = EdgeDetection.combine_edges(edges_x, edges_y, width, height)

        points = []
        for y in range(height):
            for x in range(width):
                if combined_edges.getpixel((x, y)) > self.threshold:
                    points.append((x, y))

        points_array = np.array(points)

        cluster_centers = kmeans(points_array, self.num_clusters)

        ordered_points = create_closed_loop(cluster_centers)

        fixer = PathFixer()
        ordered_points = fixer.fix_ordering(ordered_points, self.radius)

        return ordered_points

    def plot_segments(self, ordered_points, show_ordering=True):
        plt.figure(figsize=(8, 6))
        segmented_arrays = segment_points(np.array(ordered_points))
        for idx, segment in enumerate(segmented_arrays, start=1):
            plt.scatter(segment[:, 0], segment[:, 1], label=f'Segment {idx}')


        if show_ordering:
            for j, point in enumerate(ordered_points):
                plt.text(point[0], point[1], str(j + 1), ha='center', va='center', fontsize=8)
        

    def perform_interpolation(self, ordered_points, interpolation_strategy):
        interpolator = Interpolator(interpolation_strategy)

        for segment in segment_points(np.array(ordered_points)):
            segment = segment[np.argsort(segment[:, 0])]
            x = segment[:, 0]
            y = segment[:, 1]
            interpolator.interpolate(x, y)
        
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Segments Based On Function Definition')
        plt.show()

from interpolations import LinearPieceWiseInterpolation,  LeastSquaresInterpolation, CubicSplineInterpolation


if __name__ == "__main__":
    main = Main('curve4.jpg')
    ordered_points = main.process_image()
    main.plot_segments(ordered_points, False)
    main.perform_interpolation(ordered_points, LinearPieceWiseInterpolation())
