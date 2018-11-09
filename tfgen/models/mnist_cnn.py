import random
import tensorflow as tf
import numpy as np
from tfgen.generate_model import save_checkpoint, save_graph, freeze_graph, here


class MNIST_CNN(object):
    def __init__(self):
        self.mnist_loaded = False
        self.batch_index = 0
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def model(self):
        tf.reset_default_graph()

        # x  = tf.placeholder(tf.uint8, shape=(None, 28, 28))
        y = tf.placeholder(tf.uint8, shape=(None,))
        x = tf.placeholder(tf.float32, shape=(None, 28, 28))

        y_one_hot = tf.one_hot(y, 10, dtype=tf.float32)

        x_reshaped = tf.reshape(x, [-1, 28, 28, 1])
        # x_float = tf.dtypes.cast(x_reshaped, tf.float32)
        # x_normed = x_float / 255.0
        x_normed = x_reshaped

        # conv1 = tf.layers.Conv2D(32, [5, 5])(x_normed)
        conv1 = tf.layers.Conv2D(8, [2, 2])(x_normed)
        pool1 = tf.layers.MaxPooling2D([2, 2], [2, 2])(conv1)

        # conv2 = tf.layers.Conv2D(64, [5, 5])(pool1)
        conv2 = tf.layers.Conv2D(16, [2, 2])(pool1)
        pool2 = tf.layers.MaxPooling2D([2, 2], [2, 2])(conv2)

        flat = tf.layers.Flatten()(pool2)

        # fc1 = tf.layers.Dense(1024, activation=tf.nn.relu,
        fc1 = tf.layers.Dense(64, activation=tf.nn.relu,
            kernel_initializer=tf.initializers.random_normal)(flat)
        fc2 = tf.layers.Dense(10, activation=tf.nn.relu,
            kernel_initializer=tf.initializers.random_normal, name='output')(fc1)
        
        out = fc2

        loss = tf.losses.mean_squared_error(y_one_hot, fc2)

        # append :output to the name of every op that is intended to be output
        # this way I can get what I need
        output_names = ['output/Relu']

        train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        return x, y, out, train, loss, output_names

    def batch(self, batch_size):
        if not self.mnist_loaded:
            mnist = tf.keras.datasets.mnist
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
            self.rand = random.Random()
            self.rand.seed(a=42)
        idx = self.rand.sample(range(len(self.x_train)), batch_size)
        self.x_train
        x = np.array([self.x_train[i] for i in idx])
        y = np.array([self.y_train[i] for i in idx])
        # if self.batch_index + batch_size < len(self.x_train):
        #     x = self.x_train[self.batch_index : self.batch_index + batch_size]
        #     y = self.y_train[self.batch_index : self.batch_index + batch_size]
        # else:
        #     remaining = self.batch_index + batch_size - len(self.x_train)
        #     x = self.x_train[self.batch_index:] + self.x_train[:remaining]
        #     y = self.y_train[self.batch_index:] + self.x_train[:remaining]
        return x, y

    def test_data(self):
        # Must have already loaded mnist by calling self.batch(...)
        return self.x_test, self.y_test


def main():
    cnn = MNIST_CNN()
    inp, label, out, train, loss, output_names = cnn.model()

    num_batches = 1000
    batch_size = 25

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(num_batches):
            x_batch, y_batch = cnn.batch(batch_size)
            x_batch = x_batch.astype(np.float32)
            x_batch /= 255.0

            _, loss_batch = sess.run([train, loss], feed_dict={
                inp: x_batch,
                label: y_batch,
            })
            print("\033[K", end='\r')
            print("Loss: ", loss_batch, end='\r')
    
        x_test, y_test = cnn.test_data()
        x_test = x_test.astype(np.float32)
        x_test /= 255.0
        loss_test = sess.run(loss, feed_dict={
            inp: x_test,
            label: y_test,
        })
        print("Final loss: ", loss_test)
        print()

        ckpt, meta_graph = save_checkpoint(sess)
        graph_def = save_graph(sess)
        output_graph = here("frozen.pb")

        freeze_graph(graph_def, ckpt, meta_graph, output_names, output_graph)


if __name__ == '__main__':
    main()