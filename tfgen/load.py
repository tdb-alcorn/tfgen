import tensorflow as tf


def from_checkpoint(ckpt_path):
    sess = tf.Session()
    meta_graph = ckpt_path + '.meta'
    saver = tf.train.import_meta_graph(meta_graph)
    saver.restore(sess, ckpt_path)
    return sess