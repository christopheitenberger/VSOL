import unittest

from nn_compression_pipeline.compress_with_measurements_with_layer_information import \
    CompressWithMeasurementsWithLayerInformation
from np_utils.testing import assert_array_of_arrays_equal
from test_utils.fixture_neural_network_weights import weights_fixture, deep_copy_weights


class TestCaseCompressAndDecompressWeights(unittest.TestCase):

    def create_class_to_test(self) -> CompressWithMeasurementsWithLayerInformation:
        raise NotImplementedError()

    def setUp(self):
        self.compressor = self.create_class_to_test()

        self.weights_to_compress = weights_fixture()
        self.weights_to_compress_copy_desired = deep_copy_weights(self.weights_to_compress)
        self.empty_run_params = {}

        self.compressor.reset_weights_without_model(self.weights_to_compress)
        self.compressor.reset_other()

    def x_test_compressed_and_decompressed_weights_same_as_before(self):
        compressed = self.compressor.compress(self.weights_to_compress, self.empty_run_params)
        decompressed = self.compressor.decompress(compressed, self.empty_run_params)

        assert_array_of_arrays_equal(decompressed, self.weights_to_compress_copy_desired)


if __name__ == '__main__':
    unittest.main()
