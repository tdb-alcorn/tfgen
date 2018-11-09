#!/usr/bin/env python3

import argparse
import tensorflow as tf
from tensorflow.core.framework import graph_pb2


def parse_graph_def(pb):
    graph_def = graph_pb2.GraphDef()

    with open(pb, 'rb') as b:
        graph_def.ParseFromString(b.read())

    return graph_def

def list_ops(graph_def):
    return set(node.op for node in graph_def.node)

def list_children(graph_def, node):
    return [other for other in graph_def.node if node.name in other.input]

def list_inputs(graph_def, node):
    return [other for other in graph_def.node if other.name in node.input]

def get_usages(graph_def, op):
    return [node for node in graph_def.node if node.op == op]

def get_nodes_by_name(graph_def, name_fragment):
    return [node for node in graph_def.node if name_fragment in node.name]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query a tensorflow protobuf file')
    parser.add_argument('input_file', type=str, help='the protobuf file')
    parser.add_argument('command', type=str, help='list, op, name, inputs, children')
    parser.add_argument('op_or_name', type=str, default='', help='op type or node name')
    args = parser.parse_args()

    pb_file = args.input_file
    command = args.command
    op_or_name = args.op_or_name

    graph_def = parse_graph_def(pb_file)

    if command == 'list':
        print('\n'.join(list_ops(graph_def)))
        # print('\n'.join(map(lambda op: "- [ ] "+op, list_ops(graph_def))))
    elif command == 'op':
        print('\n'.join(map(str, get_usages(graph_def, op_or_name))))
    elif command == 'name':
        print('\n'.join(map(str, get_nodes_by_name(graph_def, op_or_name))))
    elif command == 'children':
        nodes = get_usages(graph_def, op_or_name)
        children = map(lambda node: list_children(graph_def, node), nodes)
        formatted_children = map(lambda cn: 'Num children: ' + str(len(cn)) + '\n' + ', '.join(map(str, cn)), children)
        print('\n'.join(formatted_children))
    elif command == 'inputs':
        nodes = get_usages(graph_def, op_or_name)
        inputs = map(lambda node: list_inputs(graph_def, node), nodes)
        formatted_inputs = map(lambda inp: 'Num inputs: ' + str(len(inp)) + '\n' + '\n'.join(map(str, inp)), inputs)
        print('\n\n'.join(formatted_inputs))
    
    # print('\n'.join(map(str, map(lambda n: list_children(graph_def, n), get_usages(graph_def, op_or_name)))))