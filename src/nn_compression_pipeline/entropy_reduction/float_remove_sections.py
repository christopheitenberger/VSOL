import numpy as np

from src.nn_compression_pipeline import Compress_With_Measurements
from src.np_utils.np_byte_manipulation import bit_range_as_single


class FloatRemoveSections(Compress_With_Measurements):
    """
        due to buildup of np.single per platform, platform dependent
    """
    lossy = True

    # testing = True

    def __init__(self, numb_start_bit_to_remove_big_e: int, numb_end_bit_to_remove_big_e_exclusive: int):
        super().__init__()
        self.numb_start_bit_to_remove_big_e = numb_start_bit_to_remove_big_e
        self.numb_end_bit_to_remove_big_e_exclusive = numb_end_bit_to_remove_big_e_exclusive

        print_if_testing = True

        self.debug_print = self.testing and print_if_testing

        self.and_mask_to_keep_bits_excluding_start_bits = bit_range_as_single(numb_start_bit_to_remove_big_e,
                                                                              numb_end_bit_to_remove_big_e_exclusive,
                                                                              np.uint32)
        if self.debug_print:
            print(f'start {self.and_mask_to_keep_bits_excluding_start_bits}')

    def compress(self, value_in, run_params):
        value_out = list(map(lambda x: self.compress_layer(x), value_in))
        return self.compress_operations_on_in_and_out_values(value_in, value_out, run_params)

    def compress_layer(self, value_in):
        old_shape = value_in.shape
        value_in = np.frombuffer(value_in.tobytes(), np.uint32) & self.and_mask_to_keep_bits_excluding_start_bits
        value_in = np.frombuffer(value_in.tobytes(), np.float32)

        return value_in.reshape(old_shape)

    def algorithm_name(self, dic_params=None):
        if dic_params:
            return super().algorithm_name(dic_params)

        return super().algorithm_name({
            'rs': self.numb_start_bit_to_remove_big_e,
            're': self.numb_end_bit_to_remove_big_e_exclusive
        })
