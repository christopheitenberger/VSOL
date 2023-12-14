import time

import numpy as np

from src.nn_compression_pipeline import LossyWeightsHandoverClass, RunParamKeys


class Compress_With_Measurements:
    # layer of compression information
    lossy = False
    dif_combiner = False
    reset_saver_for_dif = False
    file_saver = False
    weights_can_repeat = False
    requires_pipeline = False
    trigger_callback_if_lossy = True

    testing = False

    def __init__(self):
        self.compression_sizes = []
        self.last_lossy_layer_saved: LossyWeightsHandoverClass = None
        self.name_of_algorithm = None

    def add_lossy_weights_handover(self, lossy_weights_handover_object):
        self.last_lossy_layer_saved = lossy_weights_handover_object

    def compress_and_measure(self, value_in, run_params):
        start_time = time.time()

        value_out = self.compress(value_in, run_params)

        self.save_process_time_to_param(start_time, run_params)

        return value_out

    def compress_operations_on_in_and_out_values(self, value_in, value_out, run_params):
        if self.lossy and self.last_lossy_layer_saved is not None and self.trigger_callback_if_lossy:
            self.last_lossy_layer_saved.set_lossy_weights_and_call_subscribers(value_out, run_params)
        return value_out

    def compress(self, value_in, run_params):
        return self.compress_operations_on_in_and_out_values(value_in, value_in, run_params)

    def decompress_and_measure(self, value_in, run_params):
        start_time = time.time()

        value_out = self.decompress(value_in, run_params)

        self.save_process_time_to_param(start_time, run_params)

        return value_out

    def decompress_operations_on_in_and_out_values(self, value_in, value_out, run_params):
        return value_out

    def decompress(self, value_in, run_params):
        return self.decompress_operations_on_in_and_out_values(value_in, value_in, run_params)

    def reset(self, model):
        if self.name_of_algorithm is None:
            self.name_of_algorithm = self.algorithm_name()

        self.reset_weights(model)
        self.reset_other()
        if self.last_lossy_layer_saved and self.last_lossy_layer_saved.lossy_weights:
            self.last_lossy_layer_saved.lossy_weights = None

    def reset_weights(self, model):
        pass

    def reset_weights_without_model(self, weights: [np.array]):
        """
        Used for testing to avoid having to create a full model
        :param weights: the weights of a model
        """
        pass

    def reset_other(self):
        pass

    def reset_measurements_and_other_after_run(self):
        self.compression_sizes = []
        self.last_lossy_layer_saved = None

    def algorithm_name(self, dic_params=None):
        if self.name_of_algorithm is not None:
            return self.name_of_algorithm

        algorithm_params_array = []
        if dic_params:
            algorithm_params_array = list(map(lambda x: f'{x[0]}_{str(x[1]).replace("-", "*")}', dic_params.items()))

        return '#'.join([self.__class__.__name__, *algorithm_params_array])

    def save_process_time_to_param(self, last_time, run_params):
        process_time = self.__get_time(last_time)
        self.save_fixed_process_time_to_param(process_time, run_params)

    def save_fixed_process_time_to_param(self, process_time_fixed, run_params):
        run_params[RunParamKeys.PROCESS_TIME_PER_COMPRESSOR].append(
            [self.algorithm_name(), process_time_fixed])

    def save_compressed_time_to_last_save(self, last_time, run_params):
        process_time = self.__get_time(last_time)
        for name_and_process_time in run_params[RunParamKeys.PROCESS_TIME_PER_COMPRESSOR]:
            if name_and_process_time[0] == self.algorithm_name():
                name_and_process_time[1] = name_and_process_time[1] + process_time
                return None

        raise IndexError()

    def save_compression_size(self, size_before_compression_bytes, size_after_compression_bytes):
        self.compression_sizes.append([size_before_compression_bytes, size_after_compression_bytes])

    def __get_time(self, last_time):
        return time.time() - last_time
