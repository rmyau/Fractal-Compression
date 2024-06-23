from old.som_test import *
from old.best_compression import *
import os

som_dim = 8
som = SOMNetwork(input_dim=5, dim=som_dim, dtype=tf.float64, sigma=3)
iter = 0
training_op, lr_summary, sigma_summary = som.training_op()

def save_result(result):
    file = open('models/weights.txt', 'w')
    for i in range(len(result)):
        for j in range(len(result[0])):
            weight = [str(x) for x in result[i][j].tolist()]
            file.write(f'{" ".join(weight)}\n');
    file.close()

def learn():
    global iter
    for imagename in os.listdir('../images/learn'):
        img = mpimg.imread(os.path.join('../images/learn', imagename))
        print(os.path.join('../images/learn', imagename))
        img = get_greyscale_image(img)
        source_blocks = classification_source_blocks(img, 8, 4)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            init.run()
            start = time.time()
            for block in source_blocks:
                iter += 1
                if iter % 1000 == 0:
                    print('iter:', iter)
                sess.run(training_op, feed_dict={som.x: block.property_weight, som.n: iter})
            end = time.time()
            print(end - start)
            result = tf.reshape(som.w, [som_dim, som_dim, -1]).eval()
            # print(result)
    save_result(result)
