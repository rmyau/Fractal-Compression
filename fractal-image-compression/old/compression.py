import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import numpy as np
import math
import time
from property import *

class domain:
    def __init__(self, data, x, y):
        self.data = data
        self.property_weight = self.get_property_vector(data)
        self.x = x
        self.y = y

    @staticmethod
    def get_property_vector(data):
        property_weight = []
        property_list = [standart_deviation, skewness, neighbor_contrast, beta, maximum_gradient]
        for i in range(len(property_list)):
            property_weight.append(property_list[i](data))
        return property_weight


# Manipulate channels

def get_greyscale_image(img):
    return np.mean(img[:, :, :2], 2)

# Transformations

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


# Contrast and brightness

def find_contrast_and_brightness1(D, S):
    # Fix the contrast and only fit the brightness
    contrast = 0.75
    brightness = (np.sum(D - contrast * S)) / D.size
    return contrast, brightness


def find_contrast_and_brightness2(D, S):
    # Fit the contrast and the brightness
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    # x = optimize.lsq_linear(A, b, [(-np.inf, -2.0), (np.inf, 2.0)]).x
    return x[1], x[0]


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
            blocks[i].property_weight[j] = (blocks[i].property_weight[j] - fmin) / (fmax - fmin)


# Compression for greyscale images
def generate_all_source_blocks(img, source_size, step, source_blocks):
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):
            source_blocks.append(
                domain(data=img[k * step:k * step + source_size, l * step:l * step + source_size], x=l, y=k))


def generate_all_transformed_blocks(img, source_size, destination_size, step, transformed_blocks):
    factor = source_size // destination_size
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):
            # Extract the source block and reduce it to the shape of a destination block
            S = reduce(img[k * step:k * step + source_size, l * step:l * step + source_size], factor)
            # Generate all possible transformed blocks
            for direction, angle in candidates:
                transformed_blocks.append((k, l, direction, angle, apply_transformation(S, direction, angle)))
    return transformed_blocks


def compress2(img, source_size, destination_size, step):
    source_blocks = []
    generate_all_source_blocks(img, source_size, step, source_blocks)
    extremum_list = get_extremum_list(source_blocks)
    normalization_blocks(source_blocks, extremum_list)


def compress(img, source_size, destination_size, step):
    transformations = []
    transformed_blocks = generate_all_transformed_blocks(img, source_size, destination_size, step, [])
    # transformed_blocks = generate_all_transformed_blocks(img, source_size*2, destination_size, step, transformed_blocks)
    i_count = img.shape[0] // destination_size
    j_count = img.shape[1] // destination_size
    for i in range(i_count):
        transformations.append([])
        for j in range(j_count):
            print("{}/{} ; {}/{}".format(i, i_count, j, j_count))
            transformations[i].append(None)
            min_d = float('inf')
            # Extract the destination block
            D = img[i * destination_size:(i + 1) * destination_size, j * destination_size:(j + 1) * destination_size]
            # Test all possible transformations and take the best one
            for k, l, direction, angle, S in transformed_blocks:
                contrast, brightness = find_contrast_and_brightness2(D, S)
                S = contrast * S + brightness
                d = (np.sum(np.square(D - S))) / (destination_size * destination_size * 256 * 256)
                if d < min_d:
                    min_d = d
                    transformations[i][j] = (k, l, direction, angle, contrast, brightness)
    return transformations


def decompress(transformations, source_size, destination_size, step, nb_iter=8):
    factor = source_size // destination_size
    height = len(transformations) * destination_size
    width = len(transformations[0]) * destination_size
    iterations = [np.random.randint(0, 256, (height, width))]
    cur_img = np.zeros((height, width))
    for i_iter in range(nb_iter):
        print(i_iter)
        for i in range(len(transformations)):
            for j in range(len(transformations[i])):
                # Apply transform
                k, l, flip, angle, contrast, brightness = transformations[i][j]
                S = reduce(iterations[-1][k * step:k * step + source_size, l * step:l * step + source_size], factor)
                D = apply_transformation(S, flip, angle, contrast, brightness)
                cur_img[i * destination_size:(i + 1) * destination_size,
                j * destination_size:(j + 1) * destination_size] = D
        iterations.append(cur_img)
        cur_img = np.zeros((height, width))
    return iterations


# def save_file(width,height):
#     output = open(f'out.fbr', 'wb')
#     #rsize, dsize,
#     output.write(f'{width} {height}\n'.encode())
#     for level in levelRange:
#         for rang in level:
#             if not rang.haveNextLevel:
#                 output.write(get_info_rang(rang).encode())
#     output.close()

# Plot

def plot_iterations(iterations, target=None):
    # Configure plot
    plt.figure()
    nb_row = math.ceil(np.sqrt(len(iterations)))
    nb_cols = nb_row
    # Plot
    for i, img in enumerate(iterations):
        plt.subplot(nb_row, nb_cols, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
        if target is None:
            plt.title(str(i))
        else:
            # Display the RMSE
            plt.title(str(i) + ' (' + '{0:.2f}'.format(np.sqrt(np.mean(np.square(target - img)))) + ')')
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()


# Parameters

directions = [1, -1]
angles = [0, 90, 180, 270]
candidates = [[direction, angle] for direction in directions for angle in angles]


# Tests
def test():
    img = mpimg.imread('monkey_256_grey.gif')
    img = get_greyscale_image(img)
    compress2(img, 8, 4, 8)


def test_greyscale():
    img = mpimg.imread('images/monkey.gif')
    img = get_greyscale_image(img)
    # img = reduce(img, 4)
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='none')
    start_time = time.time()
    transformations = compress(img, 8, 4, 8)
    print('---- %s seconds ---- ' % (time.time() - start_time))
    iterations = decompress(transformations, 8, 4, 8)
    plot_iterations(iterations, img)
    plt.show()


if __name__ == '__main__':
    test_greyscale()
    # test_rgb()
