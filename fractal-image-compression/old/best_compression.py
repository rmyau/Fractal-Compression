import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
import time
from property import *


class domain:
    def __init__(self, data, x, y):
        self.data = data
        self.property_weight = get_property_vector(data)
        self.x = x
        self.y = y


# Parameters
directions = [1, -1]
angles = [0, 90, 180, 270]
candidates = [[direction, angle] for direction in directions for angle in angles]


def reduce(img, factor):
    result = np.zeros((img.shape[0] // factor, img.shape[1] // factor))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.mean(img[i * factor:(i + 1) * factor, j * factor:(j + 1) * factor])
    return result


def rotate(img, angle):
    return ndimage.rotate(img, angle, reshape=False)


def flip(img, direction):
    return img[::direction, :]


def apply_transformation(img, direction, angle, contrast=1.0, brightness=0.0):
    return contrast * rotate(flip(img, direction), angle) + brightness


def get_property_vector(data):
    property_weight = []
    property_list = [standart_deviation, skewness, neighbor_contrast, beta, maximum_gradient]
    for i in range(len(property_list)):
        property_weight.append(property_list[i](data))
    return property_weight


def get_greyscale_image(img):
    return np.mean(img[:, :, :2], 2)


def find_contrast_and_brightness2(D, S):

    # Fit the contrast and the brightness
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    contrast = x[1]
    brightness = x[0]
    brightness = np.clip(brightness, 0, 255)
    return contrast, brightness


def get_extremum_list(blocks):
    extremum_list = [[float('-inf'), float('inf')] for i in range(5)]
    for block in blocks:
        for i in range(5):
            data = block.data
            if block.property_weight[i] > extremum_list[i][0]:
                extremum_list[i][0] = block.property_weight[i]
            if block.property_weight[i] < extremum_list[i][1]:
                extremum_list[i][1] = block.property_weight[i]
    return extremum_list


def normalization_blocks(blocks, extremum_list):
    for i in range(len(blocks)):
        for j in range(5):
            fmin = extremum_list[j][1]
            fmax = extremum_list[j][0]
            try:
                blocks[i].property_weight[j] = (blocks[i].property_weight[j] - fmin) / (fmax - fmin)
            except AttributeError:
                blocks[i][j] = (blocks[i][j] - fmin) / (fmax - fmin)


def generate_all_source_blocks(img, source_size, step, source_blocks):
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):
            source_blocks.append(
                domain(data=img[k * step:k * step + source_size, l * step:l * step + source_size], x=l, y=k))


def classification_source_blocks(img, source_size, step):
    source_blocks = []
    generate_all_source_blocks(img, source_size, step, source_blocks)
    extremum_list = get_extremum_list(source_blocks)
    normalization_blocks(source_blocks, extremum_list)
    return source_blocks


def read_neurons_weight():
    file = open('models/weights.txt', 'r')
    info = file.readlines()
    info = np.reshape(info, (8, 8))
    weight = np.zeros((8, 8, 5))
    for i in range(8):
        for j in range(8):
            # print([float(x) for x in info[i][j].split()])
            weight[i][j] = [float(x) for x in info[i][j].split()]
    return weight


def euclidian_distance(x, w):
    result = math.sqrt(np.sum(np.square(np.array(x) - np.array(w))))
    return result

def euclidian_distance_block(x, w):
    result = math.sqrt(np.sum(np.square(np.array(x)/255 - np.array(w)/255)))
    return result

def generate_all_transformed_blocks(source_size, destination_size, domain_):
    transformed_blocks = []
    factor = source_size // destination_size
    # Extract the source block and reduce it to the shape of a destination block
    S = reduce(domain_.data, factor)
    # Generate all possible transformed blocks
    for direction, angle in candidates:
        transformed_blocks.append(
            (domain_.y, domain_.x, direction, angle, apply_transformation(S, direction, angle)))
    return transformed_blocks


def competition(weight_neurons, x):  # возвращает позицию победившего нейрона
    d = []
    for i in range(8):
        for j in range(8):
            d.append(euclidian_distance(weight_neurons[i][j], x))
    return d.index(min(d)) // 8, d.index(min(d)) - d.index(min(d)) // 8 * 8


def get_domain_classes(img, weight_neurons, source_size, step):
    classification = [[[] for i in range(8)] for j in range(8)]
    source_blocks = []
    generate_all_source_blocks(img, source_size, step, source_blocks)
    extremum_list = get_extremum_list(source_blocks)
    normalization_blocks(source_blocks, extremum_list)
    for domain_ in source_blocks:
        y, x = competition(weight_neurons, domain_.property_weight)
        classification[y][x].append(domain_)
    return classification, extremum_list


def compress(img, source_size, destination_size, step, eps_weight):
    weight_neurons = read_neurons_weight()
    classification, extremum_list = get_domain_classes(img, weight_neurons, source_size, step)
    transformations = []
    i_count = img.shape[0] // destination_size
    j_count = img.shape[1] // destination_size
    for i in range(i_count):
        transformations.append([])
        # print("{}/{}".format(i, i_count))
        for j in range(j_count):
            transformations[i].append(None)
            min_d = float('inf')
            min_distance_weight = float('inf')
            best_transformation = 0
            # Extract the destination block
            D = img[i * destination_size:(i + 1) * destination_size, j * destination_size:(j + 1) * destination_size]
            d_property = get_property_vector(D)
            normalization_blocks([d_property], extremum_list)
            y_neuron, x_neuron = competition(weight_neurons, d_property)

            for m in range(-1, 2):
                for n in range(-1, 2):
                    if 0 <= y_neuron + m < 8 and 0 <= x_neuron + n < 8:
                        for domain_ in classification[y_neuron + m][x_neuron + n]:
                            distance = euclidian_distance(d_property, domain_.property_weight)
                            # print(f'distance {distance}')
                            if distance < min_distance_weight:
                                min_distance_weight = distance
                                best_domain = domain_
                                if min_distance_weight < eps_weight:
                                    transformed_blocks = generate_all_transformed_blocks(source_size, destination_size,
                                                                                         domain_)
                                    for k, l, direction, angle, S in transformed_blocks:
                                        contrast, brightness = find_contrast_and_brightness2(D, S)
                                        S = contrast * S + brightness
                                        d = euclidian_distance_block(D, S)
                                        if d < min_d:
                                            min_d = d
                                            transformations[i][j] = (k, l, direction, angle, contrast, brightness)
                                            # print("{}/{} ; {}/{}".format(i, i_count, j, j_count))
            if transformations[i][j] is None:
                transformed_blocks = generate_all_transformed_blocks(source_size, destination_size, best_domain)
                for k, l, direction, angle, S in transformed_blocks:
                    contrast, brightness = find_contrast_and_brightness2(D, S)
                    S = contrast * S + brightness
                    d = (np.sum(np.square(D - S))) / (destination_size * destination_size * 256 * 256)
                    if d < min_d:
                        min_d = d
                        transformations[i][j] = (k, l, direction, angle, contrast, brightness)
    return transformations

def rmse(d, r):
    # print(np.max(np.array(d)/255) >= 1, np.max(np.array(r)/255) >= 1)
    return np.sqrt(np.mean(np.square(np.array(d)/255 - np.array(r)/255)))
def decompress(transformations, source_size, destination_size, step, nb_iter=6):
    factor = source_size // destination_size
    height = len(transformations) * destination_size
    width = len(transformations[0]) * destination_size
    iterations = [np.random.randint(0, 256, (height, width))]
    cur_img = np.zeros((height, width))
    for i_iter in range(nb_iter):
        # print(i_iter)
        for i in range(len(transformations)):
            for j in range(len(transformations[i])):
                # Apply transform
                k, l, flip, angle, contrast, brightness = transformations[i][j]
                k, l, flip, angle = [int(x) for x in [k, l, flip, angle]]
                S = reduce(iterations[-1][k * step:k * step + source_size, l * step:l * step + source_size], factor)
                D = apply_transformation(S, flip, angle, contrast, brightness)
                cur_img[i * destination_size:(i + 1) * destination_size,
                j * destination_size:(j + 1) * destination_size] = np.clip(D, 0, 255)
        iterations.append(cur_img)
        cur_img = np.zeros((height, width))
    return iterations


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
            plt.title(str(i) + ' iteration\nd = ' + '{0:.2f}'.format(euclidian_distance_block(target, img))+'\nRMSE = ' + '{0:.2f}'.format(rmse(target, img)))

        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()


#

def test_compress(img_name, source_size, destination_size, step, eps_weight):
    img = mpimg.imread(img_name)
    img = get_greyscale_image(img)
    # plt.figure()
    # plt.imshow(img, cmap='gray', interpolation='none')
    start_time = time.time()
    transformations = compress(img, source_size, destination_size, step, eps_weight)
    save_file(img_name, transformations, source_size, destination_size, step)
    print('---- %s seconds ---- ' % (time.time() - start_time))


def test_decompress(img_name):
    img = mpimg.imread(img_name)
    img = get_greyscale_image(img)
    source_size, destination_size, step, transformations = read_file(f'{img_name[:-4]}.fbr')
    iterations = decompress(transformations, source_size, destination_size, step, nb_iter=8)
    plot_iterations(iterations, img)
    plt.figure()
    plt.imshow(iterations[len(iterations)-1],cmap='gray', interpolation='none')
    plt.show()


def save_file(file_name, transformations, source_size, destination_size, step):
    output = open(f'{file_name[:-4]}.fbr', 'wb')
    height = len(transformations) * destination_size
    width = len(transformations[0]) * destination_size
    output.write(f'{height} {width}\n'.encode())
    output.write(f'{source_size} {destination_size} {step}\n'.encode())
    for i in range(len(transformations)):
        for j in range(len(transformations[i])):
            k, l, flip, angle, contrast, brightness = transformations[i][j]
            info = f'{k} {l} {flip} {angle} {contrast} {brightness}\n'
            output.write(info.encode())
    output.close()


def read_file(file_name):
    file = open(file_name, 'rb')
    size_img = [int(x) for x in file.readline().split()]
    height, width = size_img[0], size_img[1]
    size_blocks = [int(x) for x in file.readline().split()]
    source_size, destination_size, step = size_blocks[0], size_blocks[1], size_blocks[2]
    transformations = [[float(y) for y in x.split()] for x in file.readlines()]
    transformations = np.reshape(transformations, (height // destination_size, width // destination_size, 6))
    return source_size, destination_size, step, transformations


test_compress('images/promzone.gif', 64, 32, 32, 0.2)
# read_file('images/monkey_256_grey.fbr')
test_decompress('images/promzone.gif')
# img = mpimg.imread('images/raccoon_512.gif')
# img = get_greyscale_image(img)
# plt.figure()
# plt.imshow(img, cmap='gray', interpolation='none')
# plt.show()

test_compress('images/nature_512.gif', 32, 16, 8, 0.1)
test_decompress('images/nature_512.gif')

test_compress('images/nature_512.gif', 32, 16, 16, 0.2)
test_decompress('images/nature_512.gif')

test_compress('images/nature_512.gif', 64, 32, 16, 0.2)
test_decompress('images/nature_512.gif')

test_compress('images/nature_512.gif', 64, 32, 32, 0.2)
test_decompress('images/nature_512.gif')

