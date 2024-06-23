import os
from new_compression import get_greyscale_image, normalization_source_blocks
import matplotlib.image as mpimg
from som2 import SOMNetwork, tf, np
import time

# Определение размера карты и других параметров SOM
som_dim = 8
input_dim = 5
sigma = 3
learning_rate = 0.1
tay2 = 1000

iter = 0

# Создание экземпляра SOMNetwork
som = SOMNetwork(input_dim=input_dim, dim=som_dim, sigma=sigma, learning_rate=learning_rate, tay2=tay2,
                 dtype=tf.float64)


# Функция сохранения результатов обучения
def save_weights(model, filepath):
    weights = model.get_weights()
    with open(filepath, 'wb') as f:
        np.save(f, weights)


# Функция обучения SOM
def learn():
    global iter
    for imagename in os.listdir('..\images\learn'):
        img = mpimg.imread(os.path.join('..\images\learn', imagename))
        print(os.path.join('..\images\learn', imagename))
        img = get_greyscale_image(img)
        source_blocks = normalization_source_blocks(img, 8, 4)

        start = time.time()
        for block in source_blocks:
            iter += 1
            if iter % 1000 == 0:
                print('iter:', iter)
            # Выполнение операции обучения
            som.train_step(tf.expand_dims(block.property_weight, axis=0), iter)
        end = time.time()
        print(end - start)
    # Получение результатов и сохранение
    save_weights(som, 'models/som_weights.npy')


learn()
