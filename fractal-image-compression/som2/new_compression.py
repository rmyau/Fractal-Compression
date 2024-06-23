import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
import time
from property import *
from som2 import SOMNetwork


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


def normalize_vector(vector, extremum_list):
    for j in range(5):
        fmin = extremum_list[j][1]
        fmax = extremum_list[j][0]
        if fmax != fmin:
            vector[j] = (vector[j] - fmin) / (fmax - fmin)
        else:
            vector[j] = 0.0
    return vector


def get_property_vector(data):
    property_list = [standart_deviation, skewness, neighbor_contrast, beta, maximum_gradient]
    return [prop(data) for prop in property_list]


def get_property_vector_range(data, som_model, extremum_list):
    property_list = [standart_deviation, skewness, neighbor_contrast, beta, maximum_gradient]
    property_vector = normalize_vector([prop(data) for prop in property_list], extremum_list)
    win_index_2d = som_model(property_vector)
    return property_vector, [win_index_2d[0], win_index_2d[1]]


def get_greyscale_image(img):
    return np.mean(img[:, :, :2], 2)


def find_contrast_and_brightness2(D, S):
    # Fit the contrast and the brightness
    A = np.concatenate((np.ones((S.size, 1)), np.reshape(S, (S.size, 1))), axis=1)
    b = np.reshape(D, (D.size,))
    x, _, _, _ = np.linalg.lstsq(A, b)
    contrast = x[1]
    brightness = x[0]

    # Ограничиваем яркость в пределах 0-255
    brightness = np.clip(brightness, 0, 255)

    return contrast, brightness


def get_extremum_list(blocks):
    extremum_list = [[float('-inf'), float('inf')] for i in range(5)]
    for block in blocks:
        for i in range(5):
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
            if fmax != fmin:
                blocks[i].property_weight[j] = (blocks[i].property_weight[j] - fmin) / (fmax - fmin)
            else:
                blocks[i].property_weight[j] = 0.0

def generate_all_source_blocks(img, source_size, step, source_blocks):
    for k in range((img.shape[0] - source_size) // step + 1):
        for l in range((img.shape[1] - source_size) // step + 1):
            source_blocks.append(
                domain(data=img[k * step:k * step + source_size, l * step:l * step + source_size], x=l, y=k))


def normalization_source_blocks(img, source_size, step):
    source_blocks = []
    generate_all_source_blocks(img, source_size, step, source_blocks)
    extremum_list = get_extremum_list(source_blocks)
    normalization_blocks(source_blocks, extremum_list)
    return source_blocks, extremum_list


def load_weights(model, filepath):
    with open(filepath, 'rb') as f:
        weights = np.load(f, allow_pickle=True)
    if weights.shape != model.w.shape:
        weights = np.reshape(weights, model.w.shape)
    model.w.assign(weights)


def euclidian_distance(x, w):
    # result = math.sqrt(np.sum(np.square(np.array(x) - np.array(w))))
    result = np.linalg.norm(np.array(x) - np.array(w))
    return result


def rmse(d, s):
    return np.sqrt(np.mean(np.square(np.array(d)/255 - np.array(s)/255)))


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


def get_domain_classes(img, som_model, source_size, step):
    classification = [[[] for _ in range(som_model.dim)] for _ in range(som_model.dim)]
    source_blocks, extremum_list = normalization_source_blocks(img, source_size, step)
    for domain_ in source_blocks:
        win_index_2d = som_model(domain_.property_weight)
        classification[win_index_2d[0]][win_index_2d[1]].append(domain_)
    return classification, extremum_list


def compress(img, som_model, source_size, destination_size, step, eps_weight):
    classification, extremum_list = get_domain_classes(img, som_model, source_size, step)
    transformations = []
    i_count = img.shape[0] // destination_size
    j_count = img.shape[1] // destination_size
    for i in range(i_count):
        transformations.append([])
        for j in range(j_count):
            transformations[i].append(None)
            min_d = float('inf')
            min_distance_weight = float('inf')
            # Extract the destination block
            D = img[i * destination_size:(i + 1) * destination_size, j * destination_size:(j + 1) * destination_size]
            d_property, class_coordinates = get_property_vector_range(D, som_model, extremum_list)
            for domain_ in classification[class_coordinates[0]][class_coordinates[1]]:
                distance = euclidian_distance(d_property, domain_.property_weight)
                if distance < min_distance_weight:
                    min_distance_weight = distance
                    best_domain = domain_

                    transformed_blocks = generate_all_transformed_blocks(source_size, destination_size, domain_)
                    for k, l, direction, angle, S in transformed_blocks:
                        contrast, brightness = find_contrast_and_brightness2(D, S)
                        S = contrast * S + brightness
                        d = rmse(D, S)
                        if d < eps_weight and d < min_d:
                            min_d = d
                            transformations[i][j] = (k, l, direction, angle, contrast, brightness)

            if transformations[i][j] is None:
                transformed_blocks = generate_all_transformed_blocks(source_size, destination_size, best_domain)
                for k, l, direction, angle, S in transformed_blocks:
                    contrast, brightness = find_contrast_and_brightness2(D, S)
                    S = contrast * S + brightness
                    d = rmse(D, S)
                    if d < min_d:
                        min_d = d
                        transformations[i][j] = (k, l, direction, angle, contrast, brightness)
    return transformations


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
                j * destination_size:(j + 1) * destination_size] = D
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
        print(i)
        plt.subplot(nb_row, nb_cols, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
        plt.title(str(i))
        if target is None:
            plt.title(str(i))
        else:
            # Display the RMSE
            plt.title(str(i) + ' iteration\nRMSE ' + '{0:.2f}'.format(rmse(target, img)))
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
    plt.tight_layout()


def test_compress(img_name, source_size, destination_size, step, eps_weight):
    img = mpimg.imread(img_name)
    img = get_greyscale_image(img)
    # plt.figure()
    # plt.imshow(img, cmap='gray', interpolation='none')
    start_time = time.time()
    input_dim = 5
    som = SOMNetwork(input_dim=input_dim, dim=8)
    load_weights(som, 'models/som_weights.npy')
    transformations = compress(img, som, source_size, destination_size, step, eps_weight)
    save_file(img_name, transformations, source_size, destination_size, step)
    print('---- %s seconds ---- ' % (time.time() - start_time))


def test_decompress(img_name):
    img = mpimg.imread(img_name)
    img = get_greyscale_image(img)
    source_size, destination_size, step, transformations = read_file(f'{img_name[:-4]}.fbr')
    iterations = decompress(transformations, source_size, destination_size, step, nb_iter=8)
    plot_iterations(iterations, img)
    plt.figure()
    plt.imshow(iterations[len(iterations) - 1], cmap='gray', interpolation='none')
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


test_compress('../images/nature_512.gif', 64, 32, 16, 0.2)
# # read_file('images/monkey_256_grey.fbr')
test_decompress('../images/nature_512.gif')
# img = mpimg.imread('images/raccoon_512.gif')
# img = get_greyscale_image(img)
# plt.figure()
# plt.imshow(img, cmap='gray', interpolation='none')
# plt.show()
