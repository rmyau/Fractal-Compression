import random
import numpy as np
from domain import *
import copy


def euclidian_distance(x, w):
    result = np.sqrt(np.sum(np.square(np.array(x)/255 - np.array(w)/255)))
    return result


def fitness(domain, region):
    '''Функция фитнеса'''
    contrast, brightness = Domain.find_contrast_and_brightness2(np.array(region.data),
                                                                np.array(domain.transformation_data))
    domain.contrast, domain.brightness = contrast, brightness
    rang_array = np.array(region.data).astype(np.float32)
    domain_array = np.array(domain.transformation_data).astype(np.float32)

    domain_array = domain_array * contrast + brightness
    error = euclidian_distance(rang_array, domain_array)
    return error


def generate_population(blocks, population_size):
    '''Генерация случайно популяции'''
    population = []
    w_list = [x for x in range(0, 8)]
    for i in range(population_size):
        chromosome = random.choice(blocks)  #:chromosome: Domain
        w = random.choice(w_list)
        chromosome.set_transformation(w)
        population.append(chromosome)
    return population


def select_parents(population, fitness_scores, tournament_size=2):
    """Функция для выбора родителей с помощью турнирной селекции"""
    population_with_scores = list(zip(population, fitness_scores))
    parents = []
    while len(parents) < 2:
        tournament = random.sample(population_with_scores, min(tournament_size, len(population_with_scores)))
        tournament.sort(key=lambda x: x[1])  # Сортировка по оценкам приспособленности в обратном порядке
        winner = tournament[0][0]  # Выбираем победителя турнира
        if winner not in parents:  # Проверяем, что победитель не был уже выбран
            parents.append(winner)

    return parents[0], parents[1]


def crossover(parent1, parent2, ImageCoding):
    """Функция для скрещивания двух родителей"""
    parent1 = copy.deepcopy(parent1)
    parent2 = copy.deepcopy(parent2)
    random_column = random.randint(1, 3)
    if random_column == 1:
        parent1_x = parent1.x
        data_parent = ImageCoding.get_img_data_by_coordinates(parent2.x, parent1.y)
        parent1.set_data(parent2.x, parent1.y, data_parent, ImageCoding.domain_size // ImageCoding.region_size)
        data_parent = ImageCoding.get_img_data_by_coordinates(parent1_x, parent2.y)
        parent2.set_data(parent1_x, parent2.y, data_parent, ImageCoding.domain_size // ImageCoding.region_size)
    elif random_column == 2:
        new_y = parent1.y
        data_parent = ImageCoding.get_img_data_by_coordinates(parent1.x, parent2.y)
        parent1.set_data(parent1.x, parent2.y, data_parent, ImageCoding.domain_size // ImageCoding.region_size)
        data_parent = ImageCoding.get_img_data_by_coordinates(parent2.x, new_y)
        parent2.set_data(parent2.x, new_y, data_parent, ImageCoding.domain_size // ImageCoding.region_size)
    else:
        new_w = parent1.w
        parent1.set_transformation(parent2.w)
        parent2.set_transformation(new_w)
    return random.choice([parent2, parent1])


def mutation(individual, rate, ImageCoding):
    """Функция для мутации гена в индивидууме"""
    if random.random() < rate:
        random_column = random.randint(1, 3)
        if random_column == 3:
            new_w = random.randint(1, 7)
            individual.set_transformation(new_w)
        elif random_column == 1:
            new_x = random.randint(0, ImageCoding.img_size - ImageCoding.domain_size)
            new_data = ImageCoding.get_img_data_by_coordinates(new_x, individual.y)
            individual.set_data(new_x, individual.y, new_data, ImageCoding.domain_size // ImageCoding.region_size)
        else:
            new_y = random.randint(0, ImageCoding.img_size - ImageCoding.domain_size)
            new_data = ImageCoding.get_img_data_by_coordinates(individual.x, new_y)
            individual.set_data(individual.x, new_y, new_data, ImageCoding.domain_size // ImageCoding.region_size)
    return individual


def mutation_for_population(population, rate, ImageCoding):
    """Функция мутации для популяции"""
    if random.random() < rate:
        random_column = random.randint(1, 3)
        new_population = []
        for individual in population:
            if random_column == 3:
                new_w = random.randint(1, 7)
                individual.set_transformation(new_w)
            elif random_column == 1:
                new_x = random.randint(0, ImageCoding.img_size - ImageCoding.domain_size)
                new_data = ImageCoding.get_img_data_by_coordinates(new_x, individual.y)
                individual.set_data(new_x, individual.y, new_data, ImageCoding.domain_size // ImageCoding.region_size)
            else:
                new_y = random.randint(0, ImageCoding.img_size - ImageCoding.domain_size)
                new_data = ImageCoding.get_img_data_by_coordinates(individual.x, new_y)
                individual.set_data(individual.x, new_y, new_data, ImageCoding.domain_size // ImageCoding.region_size)
            new_population.append(individual)
        return new_population
    return population
