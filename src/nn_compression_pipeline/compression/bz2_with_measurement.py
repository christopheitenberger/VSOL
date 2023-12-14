import bz2

from src.nn_compression_pipeline import Compress_With_Measurements


class BZ2WithMeasurement(Compress_With_Measurements):
    def __init__(self, compression_rate):
        super().__init__()
        self.compression_rate = compression_rate

    def compress(self, value_in: bytes, run_params):
        return bz2.compress(value_in, compresslevel=self.compression_rate)

    def decompress(self, value_in, run_params):
        return bz2.decompress(value_in)

    def algorithm_name(self, dic_params=None):
        return super().algorithm_name({'cpr': self.compression_rate})
