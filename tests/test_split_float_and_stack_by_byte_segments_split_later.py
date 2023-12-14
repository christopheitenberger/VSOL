import unittest

from nn_compression_pipeline.format_rearrangement.split_float_and_stack_by_byte_segments_split_later import \
    SplitFloatAndStackByByteSegmentsSplitLater
from test_cases.testcase_compress_and_decompress_weights import TestCaseCompressAndDecompressWeights


class TestSplitFloatAndStackByByteSegmentsSplitLater(TestCaseCompressAndDecompressWeights):

    def create_class_to_test(self):
        return SplitFloatAndStackByByteSegmentsSplitLater()

    def test_compressed_and_decompressed_weights_same_as_before(self):
        self.x_test_compressed_and_decompressed_weights_same_as_before()


if __name__ == '__main__':
    unittest.main()
