from PIL import Image

class EdgeDetection:
    @staticmethod
    def sobel_operator():
        sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        return sobel_x, sobel_y

    @staticmethod
    def combine_edges(edges_x, edges_y, width, height):
        combined_edges = Image.new('L', (width, height))
        combined_pixels = combined_edges.load()
        for y in range(height):
            for x in range(width):
                val = int(((edges_x.getpixel((x, y)) ** 2) + (edges_y.getpixel((x, y)) ** 2)) ** 0.5)
                combined_pixels[x, y] = val if val <= 255 else 255
        return combined_edges
