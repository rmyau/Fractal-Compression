import numpy as np
import matplotlib.image as mpimg
from region import *
from genetic_algorithm import *
import matplotlib.pyplot as plt
import math
import time


class ImageCoding:

    def __init__(self, img_path, region_size=32, min_region_size=4, num_generations=10, mutation_rate=0.3,
                 max_error=0.2, step_generation_domain=32):
        img = mpimg.imread(img_path)
        img = self.get_greyscale_image(img)
        self.img = img
        self.img_size = img.shape[0]

        self.region_size = region_size
        self.min_region_size = min_region_size
        self.domain_size = region_size * 2
        self.step_generation_domain = step_generation_domain
        self.num_generations = num_generations  # максимальное количество поколений
        self.mutation_rate = mutation_rate
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
                domain_blocks.append(Domain(y=i, x=j, data=data, code=code))
                code += 1
        return domain_blocks

    @staticmethod
    def scale_blocks(domain_blocks, factor):
        scaled_blocks = []
        for domain_idx in range(len(domain_blocks)):
            data = Domain.reduce(domain_blocks[domain_idx].data, factor)
            domain_blocks[domain_idx].data = data
            scaled_blocks.append(domain_blocks[domain_idx])
        return scaled_blocks

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

    def genetic_algorithm(self):
        # множество рангов
        regions = self.generate_regions()
        count_encoded_regions = 0  # количество закодированных рангов

        # множество доменных блоков
        domain_blocks = self.generate_domain_blocks()

        result_coding = []
        factor = self.domain_size // self.region_size
        # Main algorithm
        while regions:
            # Scale domain blocks
            scaled_domain_blocks = self.scale_blocks(domain_blocks, factor)

            # Generate random population
            population = generate_population(scaled_domain_blocks, len(regions) - count_encoded_regions)

            current_generation = 0  # текущее поколение

            while regions and current_generation < self.num_generations:
                current_generation += 1
                print(f'CURRENT_GENERATION = {current_generation}')
                print(f'LEN REGIONS = {len(regions)}')
                next_population = []

                index_region_using = 0  # индекс рассматриваемого на данный момент ранга
                while index_region_using != len(regions):  # вычисляем пригодность доменов для всех рангов
                    # Compute fitness for all domains of population
                    fitness_scores = [fitness(domain_block, regions[index_region_using]) for domain_block in population]

                    # Find optimal domain block
                    optimal_idx = np.argmin(fitness_scores)
                    if fitness_scores[optimal_idx] < self.max_error:  # если найденный доменный блок оптимальнее
                        cur_region = regions[index_region_using]
                        result_coding.append(
                            cur_region.output_data(population[optimal_idx]))  # сохраняем оптимальный блок
                        regions = [region for region in regions if region is not cur_region]
                        index_region_using -= 1
                    elif current_generation != self.num_generations:
                        # Perform crossover and mutation operations
                        parent1, parent2 = select_parents(population, fitness_scores)
                        child = crossover(parent1, parent2, self)

                        child = mutation(child, self.mutation_rate, self)
                        next_population.append(child)

                    index_region_using += 1
                # mutation_for_population(next_population, self.mutation_rate, self)
                population = next_population
            factor = 2
            if len(regions) > self.min_region_size:
                self.region_size //= 2
                regions = self.decompose_image(regions)
            else:
                self.max_error = 0.3
                factor = 1
        return result_coding

    def genetic_algorithm2(self):
        # множество рангов
        regions = self.generate_regions()
        count_encoded_regions = 0  # количество закодированных рангов

        # множество доменных блоков
        domain_blocks = self.generate_domain_blocks()

        result_coding = []
        factor = self.domain_size // self.region_size
        # Main algorithm
        while regions:
            # Scale domain blocks
            scaled_domain_blocks = self.scale_blocks(domain_blocks, factor)

            # Generate random population
            population = generate_population(scaled_domain_blocks, len(regions) - count_encoded_regions)

            current_generation = 0  # текущее поколение

            while regions and current_generation < self.num_generations:
                current_generation += 1
                # print(f'CURRENT_GENERATION = {current_generation}')
                # print(f'LEN REGIONS = {len(regions)}')
                next_population = []

                index_region_using = 0  # индекс рассматриваемого на данный момент ранга
                while index_region_using != len(regions):  # вычисляем пригодность доменов для всех рангов
                    # Compute fitness for all domains of population
                    fitness_scores = [fitness(domain_block, regions[index_region_using]) for domain_block in population]

                    # Find optimal domain block
                    optimal_idx = np.argmin(fitness_scores)
                    if fitness_scores[optimal_idx] < self.max_error:  # если найденный доменный блок оптимальнее
                        cur_region = regions[index_region_using]
                        result_coding.append(
                            cur_region.output_data(population[optimal_idx]))  # сохраняем оптимальный блок
                        regions = [region for region in regions if region is not cur_region]
                        index_region_using -= 1
                    elif current_generation != self.num_generations:
                        # Perform crossover and mutation operations
                        parent1, parent2 = select_parents(population, fitness_scores)
                        child = crossover(parent1, parent2, self)

                        # child = mutation(child, self.mutation_rate, self)
                        next_population.append(child)

                    index_region_using += 1
                mutation_for_population(next_population, self.mutation_rate, self)
                population = next_population
            factor = 2
            if len(regions) > self.min_region_size:
                self.region_size //= 2
                regions = self.decompose_image(regions)
            else:
                self.max_error = 0.3
                factor = 1
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


# coding = ImageCoding('images/monkey.gif', region_size=16, step_generation_domain=8, max_error=0.2)
# start_time = time.time()
# result_coding1 = coding.genetic_algorithm()
# result1 = time.time() - start_time
# ImageCoding.save_file('images/monkey.gif', result_coding1, coding.domain_size, coding.step_generation_domain, coding.img_size)
#
# iterations = ImageCoding.decompress('images/monkey.fbr')
# plot_iterations(iterations, coding.img)
# plt.show()
# print('---- %s seconds for 1 ---- ' % result1)

coding2 = ImageCoding('images/capibara_512.gif', region_size=32, step_generation_domain=16, max_error=0.2)
start_time = time.time()
result_coding2 = coding2.genetic_algorithm()
result2 = time.time() - start_time
ImageCoding.save_file('images/capibara_512.gif', result_coding2, coding2.domain_size, coding2.step_generation_domain, coding2.img_size)


iterations = ImageCoding.decompress('images/capibara_512.fbr')
plot_iterations(iterations, coding2.img)
plt.show()
print('---- %s seconds for 2 ---- ' % result2)
