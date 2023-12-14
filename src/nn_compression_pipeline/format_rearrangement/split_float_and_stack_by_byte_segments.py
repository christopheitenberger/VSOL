import math

import numpy as np

from np_utils import get_split_upper_limit_for_shapes, assert_split_fits_to_model_shape
from src.nn_compression_pipeline import CompressWithMeasurementsWithLayerInformation


class SplitFloatAndStackByByteSegments(CompressWithMeasurementsWithLayerInformation):
    """
    * flatt at end required to save memory
    * move axis required to save memory
    * rollaxis after concatenate slower
    * sections must be combined as such
    """

    def __init__(self, debug_print=False):
        super().__init__()
        self.debug_print = debug_print

        self.increasing_total_shape_prod_for_split = None

    def apply_function_to_layers_with_shape(self, value_in, func):
        return list(map(lambda x: func(x[0], x[1]), zip(value_in, self.model_shapes)))

    def compress(self, value_in: [np.ndarray], run_params):
        m = self.apply_function_to_layers_with_shape(value_in, self.compress_layer)
        m = np.concatenate(m, axis=-1)
        m = m.flatten()
        return m

    def compress_layer(self, last, shape):
        layer_decoded = np.frombuffer(last.tobytes(), dtype=np.byte)
        layer_decoded = layer_decoded.reshape((math.prod(shape), 4))
        byte_sections_beginning = np.moveaxis(layer_decoded, -1, 0)

        if self.debug_print:
            print(layer_decoded.shape, byte_sections_beginning.shape)
        return byte_sections_beginning

    def decompress(self, value_in, run_params):
        re = value_in.reshape((4, self.total_size))
        re_split = np.split(re, indices_or_sections=self.increasing_total_shape_prod_for_split, axis=-1)

        if self.testing:
            models_shape_with_byte_split =  [(4,) + v for v in self.model_shapes]
            assert_split_fits_to_model_shape(re_split, models_shape_with_byte_split)

        return self.apply_function_to_layers_with_shape(re_split, self.decompress_layer)

    def decompress_layer(self, last, shape):
        layer_decoded = np.moveaxis(last, 0, 1)
        layer_decoded = np.frombuffer(layer_decoded.tobytes(), dtype=np.single)
        return layer_decoded.reshape(shape)

    def reset_other(self):
        super().reset_other()
        self.increasing_total_shape_prod_for_split = get_split_upper_limit_for_shapes(self.model_shapes)
