import os
import sys
import random
import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph as tf_freeze_graph


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


def freeze_from_checkpoint(ckpt, output_names):
    with tf.Session() as sess:
        meta_graph = ckpt + '.meta'
        saver = tf.train.import_meta_graph(meta_graph)
        saver.restore(sess, ckpt)

        graph_def = save_graph(sess)
        output_graph = here("frozen.pb")

        freeze_graph(graph_def, ckpt, meta_graph, output_names, output_graph)



if __name__ == '__main__':
    ckpt_path = sys.argv[1]
    output = sys.argv[2:]
#     output_names = ["fco/BiasAdd"]
    output_names = output
    freeze_from_checkpoint(ckpt_path, output_names)