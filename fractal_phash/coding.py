import numpy as np
import matplotlib.image as mpimg
from region import *
from domain import *
import matplotlib.pyplot as plt
import math
import time
import cv2
import numpy as np


class ImageCoding:
    def __init__(self, img_path, region_size=32, min_region_size=4, max_error=0.2, domain_step=None):
        img = mpimg.imread(img_path)
        img = self.get_greyscale_image(img)
        self.img = img
        self.img_size = self.img.shape[0]

        self.region_size = region_size
        self.min_region_size = min_region_size
        self.domain_size = region_size * 2
        self.step_generation_domain = domain_step if domain_step else region_size // 2
        self.current_level = 0
        self.max_error = max_error


    def get_greyscale_image(self, img):
        return np.mean(img[:, :, :2], 2)

    # генерируем все ранги
    def generate_regions(self):
        region_size = self.region_size
        count_regions = self.img.shape[0] // region_size
        regions = []
        for i in range(count_regions):
            for j in range(count_regions):
                region_data = self.img[i * region_size:(i + 1) * region_size, j * region_size:(j + 1) * region_size]
                new_region = Region(y=i * region_size, x=j * region_size, data=region_data, size=region_size)
                regions.append(new_region)
        return regions

    def get_img_data_by_coordinates(self, x, y):
        data = self.img[y: y + self.domain_size, x: x + self.domain_size]
        return data

    # генерация всех доменных блоков с заданным шагом
    def generate_domain_blocks(self):
        domain_blocks = []
        image = self.img
        domain_size = self.domain_size
        step = self.step_generation_domain
        code = 0
        for i in range((image.shape[0] - domain_size) // self.step_generation_domain + 1):
            for j in range((image.shape[0] - domain_size) // self.step_generation_domain + 1):
                data = image[i * step:i * step + domain_size, j * step:j * step + domain_size]
                for w in range(8):
                    domain_blocks.append(
                        Domain(y=i, x=j, data=data, code=code, w=w))
                code += 1
        return domain_blocks

    @staticmethod
    def encode_transform(domain_block, region):  # массив вывода в файл для ранга
        return [region.x, region.y, region.size, domain_block.x, domain_block.y, domain_block.w]

    @staticmethod
    def decompose_image(regions_old):
        regions = []
        for region in regions_old:
            small_regions = region.decompose_region()
            regions.extend(small_regions)
        return regions

    @staticmethod
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

    def algorithm(self):
        # множество рангов
        regions = self.generate_regions()
        domain_blocks = self.generate_domain_blocks()
        count_encoded_regions = 0  # количество закодированных рангов

        result_coding = []

        while self.region_size >= self.min_region_size:
            # print('Size', self.region_size)
            # print('COUNT REGIONS', len(regions))
            for region in regions:
                # если последний уровень, то ищем наилучший результат

                if self.region_size == self.min_region_size:
                    best_hamming = self.hamming_distance(region.phash, domain_blocks[0].phash)
                    best_domain = domain_blocks[0]
                for domain_block in domain_blocks:
                    hamming = self.hamming_distance(region.phash, domain_block.phash)
                    if hamming <= 5:
                        D = domain_block.reduce(domain_block.data, self.domain_size//region.size)
                        contrast, brightness = region.find_contrast_and_brightness2(D)
                        D = contrast * D + brightness
                        error = euclidian_distance(region.data, D)

                        if error <= self.max_error:
                            result_coding.append(region.output_data(domain_block, contrast, brightness))
                            #print('find domain')
                            break
                    elif self.region_size == self.min_region_size and hamming < best_hamming:
                        best_hamming = hamming
                        best_domain = domain_block
                if self.region_size == self.min_region_size:
                    region.best_domain = best_domain

            # для всех рангов, для которых не нашлось соответствие
            regions = [region for region in regions if not region.find_domain]

            if self.region_size != self.min_region_size:
                self.region_size //= 2
                regions = self.decompose_image(regions)
            else:
                break

        # для всех оставшихся рангов добавляем наилучший домен
        for region in regions:
            D = region.best_domain.reduce(region.best_domain.data, self.domain_size//region.size)
            contrast, brightness = region.find_contrast_and_brightness2(D)
            result_coding.append(region.output_data(region.best_domain, contrast, brightness))

        return result_coding

    @staticmethod
    def find_coordinates_for_domain(img_size, code, domain_size, step_domain):
        domain_in_row = (img_size - domain_size) // step_domain + 1
        x = code % domain_in_row
        y = code // domain_in_row
        return int(x), int(y)

    @staticmethod
    def decompress(file_name, nb_iter=8):
        img_size, domain_size, transformations, step_domain = ImageCoding.read_file(file_name)
        height = width = img_size
        iterations = [np.random.randint(0, 256, (height, width))]
        cur_img = np.zeros((height, width))
        for i_iter in range(nb_iter):
            print(i_iter)
            for i in range(len(transformations)):
                xr, yr, region_size, domain_code, w = map(int, transformations[i][:5])
                contrast, brightness = map(float, transformations[i][-2:])
                xd, yd = ImageCoding.find_coordinates_for_domain(img_size, domain_code, domain_size, step_domain)
                domain_data = Domain.reduce(iterations[-1][yd: yd + domain_size, xd: xd + domain_size],
                                            domain_size // region_size)
                direction, angle = Domain.candidates[w]
                domain_data = contrast * Domain.rotate(Domain.flip(np.array(domain_data), direction),
                                                       angle) + brightness
                cur_img[yr: yr + region_size, xr: xr + region_size] = np.clip(domain_data, 0, 255)
            iterations.append(cur_img)
            cur_img = np.zeros((height, width))
        return iterations

    @staticmethod
    def save_file(file_name, transformations, domain_size, step_domain, img_size):
        output = open(f'{file_name[:-4]}.fbr', 'wb')
        output.write(f'{img_size} {domain_size} {step_domain}\n'.encode())
        for transformation in transformations:
            xr, yr, rsize, domain_code, w, contrast, brightness = transformation
            data = f'{xr} {yr} {rsize} {domain_code} {w} {contrast} {brightness}\n'
            output.write(data.encode())
        output.close()

    @staticmethod
    def read_file(file_name):
        file = open(file_name, 'rb')
        decode_base = [int(x) for x in file.readline().split()]
        img_size, domain_size, step_domain = decode_base
        transformations = [[y for y in x.split()] for x in file.readlines()]

        return img_size, domain_size, transformations, step_domain

def rmse(d, r):
    # print(np.max(np.array(d)/255) >= 1, np.max(np.array(r)/255) >= 1)
    return np.sqrt(np.mean(np.square(np.array(d)/255 - np.array(r)/255)))


def euclidian_distance(x, w):
    result = np.sqrt(np.sum(np.square(np.array(x)/255 - np.array(w)/255)))
    return result
def plot_iterations(iterations, target=None):
    # Configure plot
    plt.figure()
    nb_row = math.ceil(np.sqrt(len(iterations)))
    nb_cols = nb_row
    # Plot
    for i, img in enumerate(iterations):
        plt.subplot(nb_row, nb_cols, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
        plt.title(str(i))
        if target is None:
            plt.title(str(i))
        else:
            plt.title(str(i) + ' iteration\nd = ' + '{0:.2f}'.format(euclidian_distance(target, img))+'\nRMSE = ' + '{0:.2f}'.format(rmse(target, img)))

        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()

image_path = 'images/capibara_1024.gif'
coding = ImageCoding(image_path, region_size=32, domain_step=16, max_error=0.3)
# coding_r = ImageCoding('images/promzone.gif', region_size=8, domain_step=8, max_error=0.3)
start_time = time.time()
result_coding1 = coding.algorithm()
ImageCoding.save_file(image_path, result_coding1, coding.domain_size, coding.step_generation_domain,
                      coding.img_size)
result1 = time.time() - start_time

iterations = ImageCoding.decompress('images/capibara_1024.fbr')
plot_iterations(iterations, coding.img)
plt.show()

print('---- %s seconds for 1 ---- ' % result1)
#
# coding2 = ImageCoding('images/promzone.gif', region_size=16, domain_step=16, max_error=0.1)
# start_time = time.time()
# result_coding2 = coding2.algorithm()
# ImageCoding.save_file(image_path, result_coding2, coding2.domain_size, coding2.step_generation_domain,
#                       coding2.img_size)
# result2 = time.time() - start_time
#
#
# iterations = ImageCoding.decompress('images/promzone.fbr')
# plot_iterations(iterations, coding2.img)
# plt.show()
#
# print('---- %s seconds for 2 ---- ' % result2)
