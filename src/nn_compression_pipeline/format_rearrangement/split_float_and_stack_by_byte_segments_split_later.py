import math

import numpy as np

from src.nn_compression_pipeline import CompressWithMeasurementsWithLayerInformation


class SplitFloatAndStackByByteSegmentsSplitLater(CompressWithMeasurementsWithLayerInformation):
    """
    Similar to SplitFloatAndStackByByteSegments, see notes there, currently same memory but slower
    """

    def __init__(self):
        super().__init__()

        self.increasing_total_shape_prod_for_split = None

    def compress(self, value_in: [np.ndarray], run_params):
        k = np.concatenate([np.ravel(x) for x in value_in])
        layer_decoded = np.frombuffer(k.tobytes(), dtype=np.byte)
        layer_decoded = layer_decoded.reshape((-1, 4))
        byte_sections_beginning = np.moveaxis(layer_decoded, -1, 0)
        return byte_sections_beginning.flatten()

    def decompress(self, value_in, run_params):
        layer_decoded = np.reshape(value_in, (4, -1))
        layer_decoded = np.moveaxis(layer_decoded, 0, -1)
        layer_decoded = np.frombuffer(layer_decoded.tobytes(), dtype=np.single)
        layer_decoded = np.split(layer_decoded, indices_or_sections=self.increasing_total_shape_prod_for_split)
        return self.apply_function_to_layers_with_shape(layer_decoded, self.decompress_layer)

    def decompress_layer(self, last, shape):
        return last.reshape(shape)

    def apply_function_to_layers_with_shape(self, value_in, func):
        return list(map(lambda x: func(x[0], x[1]), zip(value_in, self.model_shapes)))

    def reset_other(self):
        super().reset_other()

        ttt = list(map(math.prod, self.model_shapes))
        for i in range(len(ttt))[1:]:
            ttt[i] += ttt[i - 1]

        self.increasing_total_shape_prod_for_split = ttt
