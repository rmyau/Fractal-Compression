import numpy as np
import tensorflow as tf
class SOMNetwork(tf.keras.Model):
    def __init__(self, input_dim, dim=10, sigma=None, learning_rate=0.1, tay2=1000,
                 dtype=tf.float64):  # изменено на tf.float64
        super(SOMNetwork, self).__init__()
        if not sigma:
            sigma = dim / 2
        self.dim = dim
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.tay1 = 1000 / np.log(sigma)
        self.minsigma = sigma * np.exp(-1000 / (1000 / np.log(sigma)))
        self.tay2 = tay2
        self.optimizer = tf.keras.optimizers.Adam()

        self.w = tf.Variable(tf.random.uniform([dim * dim, input_dim], minval=-1, maxval=1, dtype=dtype))

        # Define positions
        self.positions = tf.constant(np.array([[i, j] for i in range(dim) for j in range(dim)]), dtype=tf.int64)

    def call(self, inputs):
        win_index = self.competition(inputs)
        win_index_2d = tf.convert_to_tensor([win_index // self.dim, win_index - win_index // self.dim * self.dim],
                                            dtype=tf.int64)
        return win_index_2d

    def competition(self, inputs):
        distance = tf.sqrt(tf.reduce_sum(tf.square(inputs - self.w), axis=1))
        return tf.argmin(distance, axis=0)

    def train_step(self, inputs, iterations):
        with tf.GradientTape() as tape:
            win_index = self.competition(inputs)
            coop_dist = tf.sqrt(tf.reduce_sum(tf.square(tf.cast(self.positions -
                                                                [win_index // self.dim,
                                                                 win_index - win_index // self.dim * self.dim],
                                                                dtype=self.dtype)), axis=1))
            sigma = tf.cond(iterations > 1000, lambda: self.minsigma,
                            lambda: tf.cast(self.sigma, dtype=tf.float64) * tf.exp(
                                -tf.cast(iterations, dtype=tf.float64) / tf.cast(self.tay1, dtype=tf.float64)))

            coop_dist = tf.cast(coop_dist, dtype=tf.float64)
            tnh = tf.exp(-tf.square(coop_dist) / (2 * tf.square(sigma)))  # topological neighbourhood
            lr = self.learning_rate * tf.exp(-iterations / self.tay2)
            minlr = 0.01
            lr = tf.cast(tf.cond(lr <= minlr, lambda: minlr, lambda: lr),tf.float64)
            delta = tf.cast(tf.transpose(lr * tnh * tf.transpose(inputs - self.w)), tf.float64)
            loss = tf.reduce_sum(delta ** 2)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}


