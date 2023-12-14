import pickle

from src.nn_compression_pipeline import Compress_With_Measurements


class PickleDump(Compress_With_Measurements):
    def compress(self, value_in: bytes, run_params):
        return pickle.dumps(value_in)

    def decompress(self, value_in, run_params):
        return pickle.loads(value_in)
