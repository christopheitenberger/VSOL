import lzma

from src.nn_compression_pipeline import Compress_With_Measurements


class LZMAWithMeasurement(Compress_With_Measurements):
    def __init__(self):
        super().__init__()

    def compress(self, value_in: bytes, run_params):
        return lzma.compress(value_in)

    def decompress(self, value_in, run_params):
        return lzma.decompress(value_in)
