from typing import Tuple, List
import numpy as np


def float_byte_wise_xor(a: np.ndarray, b: np.ndarray):
    return float_byte_wise_func(a, b, np.bitwise_xor)


def float_byte_wise_func(a: np.ndarray, b: np.ndarray, func):
    assert a.shape == b.shape, f'shapes different, shape a: {a.shape}, shape b: {b.shape}'
    assert a.dtype == b.dtype, f'data types different, a: {a.dtype}, b: {b.dtype}'
    saved_shape = a.shape
    saved_dtype = a.dtype
    xor_bytes = func(np.frombuffer(a.tobytes(), np.byte), np.frombuffer(b.tobytes(), np.byte))
    xor_np = np.frombuffer(xor_bytes.tobytes(), saved_dtype)
    return xor_np.reshape(saved_shape)


def bit_range_as_single(start_big_e, end_big_e_exclusive, dtype=np.single):
    return bit_ranges_as_single([(start_big_e, end_big_e_exclusive)], dtype)


def bit_ranges_as_single(start_and_end_list: List[Tuple[int, int]], dtype=np.single):
    total_range = ranges_as_single_values_only_unique(start_and_end_list)
    decimal_number_of_bit_range = sum(map(lambda x: pow(2, x), total_range))

    bit_range_as_byte = decimal_number_of_bit_range.to_bytes(4, 'little')
    return np.frombuffer(bit_range_as_byte, dtype=dtype)


def ranges_as_single_values_only_unique(start_and_end_list) -> List[int]:
    numbs_to_add = []
    for start, end in start_and_end_list:
        numbs_to_add.extend(list(range(start, end)))
    return np.unique(numbs_to_add).tolist()
