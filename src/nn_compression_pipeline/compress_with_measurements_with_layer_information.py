import math

from np_utils.testing import is_same_shape_and_type
from src.nn_compression_pipeline import Compress_With_Measurements


class CompressWithMeasurementsWithLayerInformation(Compress_With_Measurements):
    def __init__(self):
        super().__init__()
        self.model_shapes = None
        self.total_size = None
        self.shape_changed_in_this_run = True

    def get_shapes_per_layer_and_total_size(self, model_weights):
        model_shapes = list(map(lambda layer: layer.shape, model_weights))
        total_size = sum(map(math.prod, model_shapes))

        return model_shapes, total_size

    def reset_weights(self, model):
        super().reset_weights(model)

        weights = model.get_weights()
        self.reset_weights_without_model(weights)

    def reset_weights_without_model(self, weights):
        old_shapes = self.model_shapes
        self.model_shapes, self.total_size = self.get_shapes_per_layer_and_total_size(weights)

        self.shape_changed_in_this_run = self.is_weight_shapes_changed(old_shapes)

    def is_weight_shapes_changed(self, old_shapes):
        return old_shapes is None or not is_same_shape_and_type(old_shapes, self.model_shapes)
