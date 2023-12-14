import numpy as np

from src.nn_compression_pipeline import CompressWithMeasurementsWithLayerInformation


class SplitFloatAndStack(CompressWithMeasurementsWithLayerInformation):
    """
    Executed experiments which failed:
    * Keeping each byte section of each layer in an own array and not combining it to one numpy array (no stacking)
        -> more memory required
    * Combine arrays flattend to one np array
        -> small memory improvement, but slower
    """

    def apply_function_to_layers_with_shape(self, value_in, func):
        return list(map(lambda x: func(x[0], x[1]), zip(value_in, self.model_shapes)))

    def compress(self, value_in: [np.ndarray], run_params):
        return self.apply_function_to_layers_with_shape(value_in, self.compress_layer)

    def decompress(self, value_in, run_params):
        return self.apply_function_to_layers_with_shape(value_in, self.decompress_layer)

    def compress_layer(self, last, shape, item_size_bytes=4):
        layer_decoded = np.frombuffer(last.tobytes(), dtype=np.byte)
        layer_decoded = layer_decoded.reshape(shape + (item_size_bytes,))
        return np.stack(np.split(layer_decoded, item_size_bytes, -1))

    def decompress_layer(self, last, shape, item_size_bytes=4, dtype=np.single):
        layer_reshaped = last.reshape((item_size_bytes,) + shape + (1,))
        layer_decoded = np.concatenate(np.split(layer_reshaped, item_size_bytes, 0), -1)
        layer_decoded = np.frombuffer(layer_decoded.tobytes(), dtype=dtype)
        return layer_decoded.reshape(shape)
