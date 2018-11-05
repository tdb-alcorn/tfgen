#!/usr/bin/env python3


import tensorflow as tf
from tensorflow.core.framework import graph_pb2


def parse_graph_def(pb):
    graph_def = graph_pb2.GraphDef()

    with open(pb, 'rb') as b:
        graph_def.ParseFromString(b.read())

    return graph_def

def list_ops(graph_def):
    return set(node.op for node in graph_def.node)

def get_usages(graph_def, op):
    return [node for node in graph_def.node if node.op == op]

def get_nodes_by_name(graph_def, name_fragment):
    return [node for node in graph_def.node if name_fragment in node.name]

if __name__ == '__main__':
    import sys

    pb_file = sys.argv[1]
    op_or_name = sys.argv[2]

    graph_def = parse_graph_def(pb_file)

    # print('\n'.join(list_ops(graph_def)))

    print('\n'.join(map(str, get_usages(graph_def, op_or_name))))
    # print('\n'.join(map(str, get_nodes_by_name(graph_def, op_or_name))))