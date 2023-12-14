import unittest

import np_utils
from nn_compression_pipeline import Combine, LossyWeightsHandoverClass
from np_utils.testing import assert_array_of_arrays_equal
from test_utils.fixture_neural_network_weights import weights_fixture, deep_copy_weights


class TestSplitFloatAndStack(unittest.TestCase):

    def setUp(self):
        self.compressor = Combine(np_utils.float_byte_wise_xor)
        handover_class = LossyWeightsHandoverClass()
        self.compressor.add_lossy_weights_handover(handover_class)

        self.first_weights = weights_fixture(111)
        self.list_of_weights_to_compress = [
            weights_fixture(1),
            weights_fixture(2),
            weights_fixture(3),
        ]
        self.weights_to_compress_copy_desired = [deep_copy_weights(w) for w in self.list_of_weights_to_compress]
        self.empty_run_params = {}

        handover_class.reset_weights(self.first_weights)
        self.compressor.reset_weights_without_model(self.first_weights)
        self.compressor.reset_other()

    def test_compressed_and_decompressed_weights_same_as_before(self):
        compressed_deltas = self.compress_list_of_weights_to_delta()
        decompressed_materialized_weights = self.decompress_list_of_weights_to_materialized_weights(compressed_deltas)

        for weight_set_desired, weight_set_decompressed in zip(self.weights_to_compress_copy_desired,
                                                               decompressed_materialized_weights):
            assert_array_of_arrays_equal(weight_set_decompressed, weight_set_desired)

    def compress_list_of_weights_to_delta(self):
        return [self.compressor.compress(w, self.empty_run_params) for w in self.list_of_weights_to_compress]

    def decompress_list_of_weights_to_materialized_weights(self, weight_deltas):
        return [self.compressor.decompress(w, self.empty_run_params) for w in weight_deltas]


if __name__ == '__main__':
    unittest.main()
