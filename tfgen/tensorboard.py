import tensorflow as tf


def write(ckpt_path):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)
        writer = tf.summary.FileWriter('output', sess.graph)
        # sess.run()
        writer.close()


if __name__ == '__main__':
    import sys
    ckpt = sys.argv[1]
    write(ckpt)