#!/usr/bin/env python3


import tensorflow as tf
from tensorflow.core.framework import graph_pb2


def graph_def(pb):
    graph_def = graph_pb2.GraphDef()

    with open(pb, 'rb') as b:
        graph_def.ParseFromString(b.read())

    return graph_def


if __name__ == '__main__':
    import sys

    main(sys.argv[1])