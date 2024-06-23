import cv2
import numpy as np


# def perceptual_hash(image_path, hash_size=8):
#     # Загрузка изображения и преобразование в оттенки серого
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     # Изменение размера изображения
#     img = cv2.resize(img, (hash_size, hash_size))
#
#     # Создание бинарного хэша
#     mean = np.mean(img)
#     binary_hash = (img > mean).astype(np.uint8)
#
#     # Преобразование бинарного хэша в шестнадцатеричное представление
#     hex_hash = ''.join(['{:02x}'.format(int(''.join(map(str, row)), 2)) for row in binary_hash])
#
#     return hex_hash


def hamming_distance(hash1, hash2):
    """
    Вычисляет расстояние Хэмминга между двумя хэшами.
    """
    # Проверка на одинаковую длину хэшей
    if len(hash1) != len(hash2):
        raise ValueError("Хэши должны иметь одинаковую длину")

    # Вычисление расстояния Хэмминга
    distance = sum(bit1 != bit2 for bit1, bit2 in zip(hash1, hash2))
    return distance
# Пример использования
image_path = r".\images\test_512.png"
phash1 = perceptual_hash(image_path)
print("Perceptual hash:", phash1)
phash2 = perceptual_hash(r".\images\test_512.png")
print("Perceptual hash:", phash2)
print(hamming_distance(hash1=phash1, hash2=phash2))
