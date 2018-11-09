from collections import Counter
import tensorflow as tf
import tfgen.load as load

def count_parameters(graph: tf.Graph):
    total_parameters = 0
    
    trainable = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    parameters = list()

    for variable in trainable:
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        parameters.append((variable, variable_parameters))
        total_parameters += variable_parameters

    by_scope = Counter()
    by_name = Counter()
    for variable, num_params in parameters:
        by_name[variable.op.name] += num_params
        by_scope[variable.op.name.split('/')[0]] += num_params

    return total_parameters, by_scope, by_name


def tabulate(counter):
    s = map(lambda c: '%s:\t%d' % c, counter.most_common())
    return '\n'.join(s)

def main(ckpt):
    sess = load.from_checkpoint(ckpt)
    total, by_scope, by_name = count_parameters(sess.graph)
    print("Parameters: %d" % total)
    print("By scope:\n")
    print(tabulate(by_scope))
    print()
    print("By name:\n")
    print(tabulate(by_name))


if __name__ == '__main__':
    import sys
    ckpt = sys.argv[1]
    main(ckpt)