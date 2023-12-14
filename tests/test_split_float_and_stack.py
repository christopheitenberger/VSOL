import unittest

from test_cases.testcase_compress_and_decompress_weights import TestCaseCompressAndDecompressWeights
from nn_compression_pipeline.format_rearrangement.split_float_and_stack import SplitFloatAndStack


class TestSplitFloatAndStack(TestCaseCompressAndDecompressWeights):

    def create_class_to_test(self):
        return SplitFloatAndStack()

    def test_compressed_and_decompressed_weights_same_as_before(self):
        self.x_test_compressed_and_decompressed_weights_same_as_before()


if __name__ == '__main__':
    unittest.main()
