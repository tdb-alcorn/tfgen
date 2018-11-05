#!/usr/bin/env python


import os
import random
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph as tf_freeze_graph


class XOR(object):
    def model():
        tf.reset_default_graph()

        x  = tf.placeholder(tf.float32, shape=(None, 2))
        y = tf.placeholder(tf.float32, shape=(None,))

        hidden = tf.layers.dense(x, 4,
            activation=tf.nn.relu, 
            kernel_initializer=tf.initializers.constant([[1, 1, -1, -1], [1, -1, 1, -1]]))
        out_l = tf.layers.dense(hidden, 1,
            activation=tf.nn.relu,
            kernel_initializer=tf.initializers.constant([[-1, 1, 1, -1]]))

        out = tf.reshape(out_l, (-1,), name='output')

        # append :output to the name of every op that is intended to be output
        # this way I can get what I need
        output_names = ['output']

        loss = tf.losses.mean_squared_error(y, out)

        train = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

        return x, y, out, train, loss, output_names

    def generate_batch(n=10, low=-1, high=1):
        x = list()
        y = list()
        for i in range(n):
            datum = (random.uniform(low, high), random.uniform(low, high))
            xor = (datum[0] > 0) ^ (datum[1] > 0)
            xor_f = 1.0 if xor else 0.0
            x.append(datum)
            y.append(xor_f)
        return x, y


def here(filename):
    return os.path.join(os.curdir, filename)


def save_graph(sess):
    name = 'graph.pbtxt'
    tf.train.write_graph(sess.graph_def, here(''), name, as_text=True)
    return here(name)


def save_checkpoint(sess):
    prefix = here('ckpt')
    saver = tf.train.Saver()
    checkpoint_path = saver.save(sess, prefix)
    meta_path = prefix + '.meta'
    return checkpoint_path, meta_path


def freeze_graph(graph_def, ckpt, meta_graph, output_nodes, output_graph, input_binary=False):
    tf_freeze_graph(
        graph_def,
        "",
        input_binary,
        ckpt,
        ",".join(output_nodes),
        "save/restore_all",
        "save/Const:0",
        output_graph,
        False,
        "",
        "",
        meta_graph)


def main2():
    import sys
    import glob
    ckpt_path = sys.argv[1]
    output_names = ["fco/BiasAdd"]
    # meta_path = glob.glob(ckpt_path + '/*.meta')[0]
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)

        ckpt, meta_graph = save_checkpoint(sess)
        graph_def = save_graph(sess)
        output_graph = here("frozen.pb")

        freeze_graph(graph_def, ckpt, meta_graph, output_names, output_graph)


def main():
    xor = XOR()
    inp, label, out, train, loss, output_names = xor.model()

    num_batches = 1000
    batch_size = 25

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(num_batches):
            x_batch, y_batch = xor.generate_batch(n=batch_size, low=-1, high=1)

            _, loss_batch = sess.run([train, loss], feed_dict={
                inp: x_batch,
                label: y_batch,
            })
            print("\033[K", end='\r')
            print("Loss: ", loss_batch, end='\r')
    
        print()

        ckpt, meta_graph = save_checkpoint(sess)
        graph_def = save_graph(sess)
        output_graph = here("frozen.pb")

        freeze_graph(graph_def, ckpt, meta_graph, output_names, output_graph)


if __name__ == '__main__':
    main2()