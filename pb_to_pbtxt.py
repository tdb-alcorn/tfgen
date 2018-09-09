#!/usr/bin/env python3


import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework.graph_io import write_graph
from google.protobuf import text_format


def main(pb):
    graph_def = graph_pb2.GraphDef()

    with open(pb, 'rb') as b:
        graph_def.ParseFromString(b.read())

    s = text_format.MessageToString(graph_def)

    with open(pb + 'txt', 'w') as txt:
        txt.truncate()
        txt.write(s)


if __name__ == '__main__':
    import sys

    main(sys.argv[1])