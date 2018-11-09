#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework.graph_io import write_graph
from google.protobuf import text_format
from collections import namedtuple


TestCase = namedtuple('TestCase', ['arguments', 'expected'])


def main(pb):
    inputs = ["Placeholder"]
    outputs = ["output"]
    frozen = ["dense/kernel", "dense/bias", "dense_1/kernel", "dense_1/bias"]

    test_cases = [
        TestCase([
            [[1, 1]],
        ], [
            0,
        ]),
        TestCase([
            [[1, -1]],
        ], [
            1,
        ]),
        TestCase([
            [[-1, 1]],
        ], [
            1,
        ]),
        TestCase([
            [[-1, -1]],
        ], [
            0,
        ]),
    ]

    graph_def = graph_pb2.GraphDef()

    with open(pb, 'rb') as b:
        graph_def.ParseFromString(b.read())
    
    nodes = tf.import_graph_def(graph_def, return_elements=inputs+outputs+frozen)
    input_nodes = nodes[:len(inputs)]
    input_tensors = [node.outputs[0] for node in input_nodes]
    # graph = tf.get_default_graph()
    # input_tensors = [graph.get_tensor_by_name(node.name + ":0") for node in input_nodes]
    output_nodes = nodes[len(inputs):len(inputs)+len(outputs)]
    output_tensors = [node.outputs[0] for node in output_nodes]

    frozen_nodes = nodes[len(inputs)+len(outputs):]
    frozen_tensors = [node.outputs[0] for node in frozen_nodes]
    # print(input_nodes[0].outputs[0])
    print(input_nodes)
    print(input_tensors)
    print(output_nodes)
    print(output_tensors)

    print(frozen_nodes)
    # print(frozen_tensors)

    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        for tensor in frozen_tensors:
            print(tensor, tensor.eval(session=sess))
        for test in test_cases:
            feed_dict=dict(zip(input_tensors, test.arguments))
            actual = sess.run(output_tensors, feed_dict=feed_dict)
            print("arguments", test.arguments)
            print("actual", actual)
            print("expected", test.expected)



if __name__ == '__main__':
    import sys

    main(sys.argv[1])