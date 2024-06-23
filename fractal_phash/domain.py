from block import *


class Domain(Block):
    candidates = [[direction, angle] for direction in [1, -1] for angle in [0, 90, 180, 270]]

    def __init__(self, x, y, data, code, w):
        transform_data = self.apply_transformation(w, data)
        super().__init__(x, y, transform_data, code)
        self.w = w
        self.transformation_data = transform_data

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

    def apply_transformation(self, w, data):
        direction, angle = self.candidates[w]
        return self.rotate(self.flip(data, direction), angle)

