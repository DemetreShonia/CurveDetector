from PIL import Image

class ImageProcessor:
    def __init__(self, image_path):
        self.image = Image.open(image_path).convert('L')
        self.width, self.height = self.image.size

    def get_image_matrix(self):
        return self.image.load()

    def apply_kernel(self, kernel):
        new_image = Image.new('L', (self.width, self.height))
        new_pixels = new_image.load()
        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                pixel = (
                    kernel[0][0] * self.image.getpixel((x - 1, y - 1)) +
                    kernel[0][1] * self.image.getpixel((x, y - 1)) +
                    kernel[0][2] * self.image.getpixel((x + 1, y - 1)) +
                    kernel[1][0] * self.image.getpixel((x - 1, y)) +
                    kernel[1][1] * self.image.getpixel((x, y)) +
                    kernel[1][2] * self.image.getpixel((x + 1, y)) +
                    kernel[2][0] * self.image.getpixel((x - 1, y + 1)) +
                    kernel[2][1] * self.image.getpixel((x, y + 1)) +
                    kernel[2][2] * self.image.getpixel((x + 1, y + 1))
                )
                new_pixels[x, y] = int(pixel)
        return new_image

    def get_image_size(self):
        return self.width, self.height
