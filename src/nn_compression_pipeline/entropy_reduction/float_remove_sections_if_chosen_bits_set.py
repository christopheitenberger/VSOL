import numpy as np
import math

from src.nn_compression_pipeline import Compress_With_Measurements
from src.np_utils.np_byte_manipulation import float_byte_wise_func, bit_range_as_single


class FloatRemoveSectionsIfChosenBitsSet(Compress_With_Measurements):
    """
        due to buildup of np.single per platform, platform dependent

        Experiments for faster execution then using this class in series:
        * Keep overall mask for checking values and create new mask for selected values
            -> very slow probably due to creating new mask for filtering and modifying second mask
                which does not compensate the lower number of values to check
        * Reducing number of values not as mask but smaller array
            -> Numpy does not feature possibility to keep shallow smaller copy of values

        Experiments for better performance:
        * Selecting a random percentage instead of true bits, also used to compare effectiveness of bit selection
            -> does not seem to work significantly well, not extensively tested, compression or accuracy and speed down
        * Using minus before xor combination for diff
            -> does not seem to work significantly well, not extensively tested, compression and accuracy down
    """
    lossy = True

    # testing = True

    def __init__(self, numb_start_bit_to_check_big_e: int, numb_end_bit_to_check_big_e_exclusive: int,
                 numb_start_bit_to_remove_big_e: int, numb_end_bit_to_remove_big_e_exclusive: int):
        super().__init__()
        self.numb_start_bit_to_check_big_e = numb_start_bit_to_check_big_e
        self.numb_end_bit_to_check_big_e_exclusive = numb_end_bit_to_check_big_e_exclusive
        self.numb_start_bit_to_remove_big_e = numb_start_bit_to_remove_big_e
        self.numb_end_bit_to_remove_big_e_exclusive = numb_end_bit_to_remove_big_e_exclusive

        print_if_testing = False

        self.debug_print = self.testing and print_if_testing

        bit_range_mask_check = bit_range_as_single(numb_start_bit_to_check_big_e, numb_end_bit_to_check_big_e_exclusive)
        self.and_mask_to_check_start_fraction_bits = \
            self.float32_to_int_byte_wise_to_use_bit_operations(bit_range_mask_check[0])
        self.and_mask_to_keep_bits_excluding_start_bits = bit_range_as_single(numb_start_bit_to_remove_big_e,
                                                                              numb_end_bit_to_remove_big_e_exclusive)

        if self.debug_print:
            print(f'start {self.and_mask_to_check_start_fraction_bits}, '
                  f'end {self.and_mask_to_keep_bits_excluding_start_bits}')

    def compress(self, value_in, run_params):
        value_out = list(map(lambda x: self.compress_layer(x), value_in))
        return self.compress_operations_on_in_and_out_values(value_in, value_out, run_params)

    def compress_layer(self, value_in):
        value_in_orig_for_debugging = None

        if self.debug_print:
            value_in_orig_for_debugging = value_in

        value_in = value_in.copy()

        high_value_selected = self.check_start_fraction_if_bits_set_as_mask(value_in)

        value_in_masked = value_in[high_value_selected]

        and_mask_remove_start_bits_broad = np.broadcast_to(self.and_mask_to_keep_bits_excluding_start_bits,
                                                           value_in_masked.shape)
        value_in[high_value_selected] = float_byte_wise_func(value_in_masked, and_mask_remove_start_bits_broad,
                                                             np.bitwise_and)
        if self.debug_print:
            self.print_one_changed_value(value_in_orig_for_debugging, value_in, high_value_selected)

        if self.debug_print:
            mask_sum = high_value_selected.sum()
            print(f'{mask_sum}/{mask_sum / math.prod(value_in.shape) * 100:.1f}')

        return value_in

    def float32_to_int_byte_wise_to_use_bit_operations(self, value):
        return np.frombuffer(value.tobytes(), np.uint32)

    def check_start_fraction_if_bits_set_as_mask(self, value_in):
        v_in_masked = self.float32_to_int_byte_wise_to_use_bit_operations(value_in) \
                      & self.and_mask_to_check_start_fraction_bits

        return (v_in_masked > 0).reshape(value_in.shape)

    def print_one_changed_value(self, new, old, mask, index=0):
        if np.sum(mask) > index:
            print(f'old: {new[mask].ravel()[index]}, '
                  f'new: {old[mask].ravel()[index]} ')

    def algorithm_name(self, dic_params=None):
        if dic_params:
            return super().algorithm_name(dic_params)

        return super().algorithm_name({
            'cs': self.numb_start_bit_to_check_big_e,
            'ce': self.numb_end_bit_to_check_big_e_exclusive,
            'rs': self.numb_start_bit_to_remove_big_e,
            're': self.numb_end_bit_to_remove_big_e_exclusive
        })
