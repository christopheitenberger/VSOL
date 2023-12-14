import numpy as np

from src.nn_compression_pipeline.delta_creation.compress_with_measurements_with_layer_information_and_weights_last_run import \
    CompressWithMeasurementsWithLayerInformationAndWeightsLastRun
from src.np_utils import percentage_layers_same


class Combine(CompressWithMeasurementsWithLayerInformationAndWeightsLastRun):
    dif_combiner = True

    # testing = True

    def __init__(self, combine_algorithm, decompress_algorithm=None):
        super().__init__()
        self.first = False
        self.combine_algorithm = combine_algorithm
        if decompress_algorithm:
            self.decompress_algorithm = decompress_algorithm
        elif combine_algorithm:
            self.decompress_algorithm = combine_algorithm

    def combine_two_models(self, values_zipped, combine_algorithm):
        return list(map(lambda x: combine_algorithm(x[0], x[1]), values_zipped))

    def recreate_weights_from_last_run_compress_from_last_lossy_weights(self):
        zipped = zip(self.last_lossy_layer_saved.weights_from_prev_run_compress,
                     self.last_lossy_layer_saved.lossy_weights)
        return self.combine_two_models(zipped, self.decompress_algorithm)

    def compress(self, value_in, run_params):
        zipped = zip(self.last_lossy_layer_saved.weights_from_last_run_compress, value_in)
        value_out = self.combine_two_models(zipped, self.combine_algorithm)
        return self.compress_operations_on_in_and_out_values(value_in, value_out, run_params)

    def decompress(self, value_in, run_params):
        zipped = zip(self.last_lossy_layer_saved.weights_from_last_run_decompress, value_in)
        value_out = self.combine_two_models(zipped, self.decompress_algorithm)

        if self.testing:
            assert sum(map(lambda x: np.count_nonzero(x), value_in)) > 0
            assert percentage_layers_same(value_out, self.last_lossy_layer_saved.weights_from_last_run_decompress) < 1
        return self.decompress_operations_on_in_and_out_values(value_in, value_out, run_params)

    def algorithm_name(self, dic_params=None):
        combine_algorithm_name = 'None' if self.combine_algorithm is None else self.combine_algorithm.__name__
        return super().algorithm_name({
            'alg': combine_algorithm_name,
        })
