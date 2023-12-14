from nn_compression_pipeline import Compress_With_Measurements
from itertools import product


class CompressionDifferentAlgorithmsPerLayersCombinationCreator:
    def __init__(self, compression_runners_with_different_algs_per_layer: [[Compress_With_Measurements]]):
        self.compression_runners_with_different_algs_per_layer = compression_runners_with_different_algs_per_layer

    def get_combinations_of_product_per_layer(self):
        range_of_indices_per_layer = list(map(lambda x: list(range(0, len(x))),
                                              self.compression_runners_with_different_algs_per_layer))
        all_combinations_indices = list(product(*range_of_indices_per_layer))

        return list(map(self.get_combination_from_ordered_indices, all_combinations_indices))

    def get_combination_from_ordered_indices(self, combination_indices):
        return [different_algs_of_layer[i] for i, different_algs_of_layer in
                zip(combination_indices, self.compression_runners_with_different_algs_per_layer)
                if different_algs_of_layer[i] is not None]
