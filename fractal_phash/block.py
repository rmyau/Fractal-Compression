import numpy as np
from scipy import ndimage
import cv2

class Block:
    def __init__(self, x, y, data, code):
        self.w = None
        self.x = x
        self.y = y
        self.code = code
        self.data = np.array(data)
        self.phash = self.perceptual_hash(data)
    @staticmethod
    def perceptual_hash(img, hash_size=8):
        # Изменение размера изображения
        img = cv2.resize(img, (hash_size, hash_size))

        # Создание бинарного хэша
        mean = np.mean(img)
        binary_hash = (img > mean).astype(np.uint8)

        # Преобразование бинарного хэша в шестнадцатеричное представление
        hex_hash = ''.join(['{:02x}'.format(int(''.join(map(str, row)), 2)) for row in binary_hash])

        return hex_hash


