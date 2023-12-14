import zstd

from src.nn_compression_pipeline import Compress_With_Measurements


class ZSTDWithMeasurement(Compress_With_Measurements):
    def __init__(self, compression_rate):
        super().__init__()
        self.compression_rate = compression_rate

    def compress(self, value_in: bytes, run_params):
        return zstd.compress(value_in, self.compression_rate)

    def decompress(self, value_in, run_params):
        return zstd.uncompress(value_in)

    def algorithm_name(self, dic_params=None):
        return super().algorithm_name({'cpr': self.compression_rate})
