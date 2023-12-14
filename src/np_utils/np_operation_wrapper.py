import math

import numpy as np


def diff_flat_with_remaining_start(array):
    """
    >>> diff_flat_with_remaining_start([1,2,3])
    array([1, 1, 1])
    """
    return np.diff(array, prepend=0)


def flatten_and_concatenate(value_in):
    return np.concatenate([np.ravel(v) for v in value_in])


def split_and_reshape(layers_concatenated, increasing_split, model_shapes, testing=False):
    re_split = split(layers_concatenated, increasing_split)
    if testing:
        assert_split_fits_to_model_shape(re_split, model_shapes)
    return reshape_split_to_model_shapes(re_split, model_shapes)


def reshape_split_to_model_shapes(re_split, model_shapes):
    return [v.reshape(s) for v, s in zip(re_split, model_shapes)]


def split(layers_concatenated, increasing_split):
    return np.split(layers_concatenated, indices_or_sections=increasing_split)


def assert_split_fits_to_model_shape(re_split, model_shapes):
    assert len(re_split) == len(model_shapes)
    for one_re_split, model_shape in zip(re_split, model_shapes):
        re_split_len = math.prod(one_re_split.shape)
        shape_prod = math.prod(model_shape)
        assert re_split_len == shape_prod, (re_split_len, shape_prod, one_re_split.shape, model_shape)


def get_split_upper_limit_for_shapes(model_shapes):
    per_layer_total_without_last = list(map(math.prod, model_shapes))[:-1]
    return np.cumsum(per_layer_total_without_last)
