import math
import numpy as np


def percentage_layers_same(weights_layers_a, weights_layers_b):
    assert_same_shape_and_type(weights_layers_a, weights_layers_b)
    same_layers_m = np.array([array_same_percentage(a, b) for a, b in zip(weights_layers_a, weights_layers_b)])
    weights_mean_for_layer_size = np.array([math.prod(x.shape) for x in weights_layers_a])
    return np.average(same_layers_m, weights=weights_mean_for_layer_size)


def assert_array_of_arrays_equal(x, y):
    for x_array, y_array in zip(x, y):
        np.testing.assert_array_equal(x_array, y_array)


def are_layers_same(weights_layers_a, weights_layers_b):
    return percentage_layers_same(weights_layers_a, weights_layers_b) == 1


def array_same_percentage(a, b):
    return np.mean(a == b)


def assert_same_shape_and_type(x, y):
    result, msg = check_same_shape_and_type(x, y)
    assert result, msg


def is_same_shape_and_type(x, y):
    return check_same_shape_and_type(x, y)[0]


def check_same_shape_and_type(x, y):
    for x_i, y_i in zip(x, y):
        if x_i.shape != y_i.shape:
            return False, f'Shapes not same: {x_i.shape} vs {y_i.shape}'
        if x_i.dtype != y_i.dtype:
            return False, f'Dtypes not same: {x_i.dtype} vs {y_i.dtype}'
    return True, ''
