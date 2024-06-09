import numpy as np
from scipy import ndimage


class Domain:
    candidates = [[direction, angle] for direction in [1, -1] for angle in [0, 90, 180, 270]]

    def __init__(self, x, y, data, code):
        self.w = None
        self.x = x
        self.y = y
        self.code = code
        self.data = np.array(data)
        self.transformation_data = None

    # масштабирование
    @staticmethod
    def reduce(data, factor):
        result = np.zeros((data.shape[0] // factor, data.shape[1] // factor))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.mean(data[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor])
        return result

    @staticmethod
    # поворот на заданный угол
    def rotate(data, angle):
        return ndimage.rotate(data, angle, reshape=False)

    @staticmethod
    # отражение
    def flip(data, direction):
        return data[::direction, :]

    def set_transformation(self, number_w):
        self.w = number_w
        self.apply_transformation()

    @staticmethod
    def find_contrast_and_brightness2(region_data, domain_data):
        # Fit the contrast and the brightness
        A = np.concatenate((np.ones((domain_data.size, 1)), np.reshape(domain_data, (domain_data.size, 1))), axis=1)
        b = np.reshape(region_data, (np.array(region_data).size,))
        x, _, _, _ = np.linalg.lstsq(A, b)
        contrast = x[1]
        brightness = x[0]
        brightness = np.clip(brightness, 0, 255)
        return contrast, brightness

    def apply_transformation(self):
        direction, angle = self.candidates[self.w]
        self.transformation_data = self.rotate(self.flip(self.data, direction), angle)

    def set_data(self, new_x, new_y, new_data, factor):
        self.x = new_x
        self.y = new_y
        self.data = new_data
        self.data = self.reduce(self.data, factor)
        self.apply_transformation()
