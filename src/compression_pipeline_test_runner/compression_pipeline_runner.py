import os
import hashlib
import pickle
import shutil
import time
import math
import sys
from statistics import mean
from typing import Optional
from visualization_utils.constants import *

import numpy as np

from nn_compression_pipeline.data_saver.dif_reset_saver import LoadDirection
from src.nn_compression_pipeline import Compress_With_Measurements, LossyWeightsHandoverClass, RunParamKeys
from src.np_utils import percentage_layers_same


class CompressionPipelineRunner:
    """
    Class that orchestrates compression pipeline runner.
    Uses algorithms and steps (runners) to compress and decompress weights.
    Tracks all required metrics during execution.
    Is savable to file.
    Tracks deployment timestamps and can reload weights accordingly.
    Partially verifies that the weights are properly compressed and decompressed.
    Tracks which types of runners are used.

    Attributes:
        weights_last_run (list): List of weights from the last run. Only used for and during debugging
        name_online_training_run (str): Name of the online training run.
        path_to_overall_model_folder (str): Path to the overall model folder.
        runners ([Compress_With_Measurements]): List of compression runners.
        debug_asserts (bool): Flag for enabling debug asserts.
        measurements_finished_and_loaded_to_class (bool): Flag indicating if measurements are finished and loaded.
        loss_metric_during_training (list): List of loss metric values during training.
        metrics (list): List of next batch accuracy.
        metrics_last_batch (list): List of last batch accuracy .
        timestamps_of_saves (list): List of timestamps of saves.
        timestamps_of_runs (list): List of timestamps of runs.
            Its length is equal or higher than the timestamps_of_saves since not every run must lead to a save
        compression_time_from_runners (list): List of compression times from runners.
        decompression_time_from_runners (list): List of decompression times from runners.
        training_time (list): List of training times.
        compression_sizes_from_runners (list): List of compression sizes from runners.
        decompress_run_counter (int): Counter for decompression runs.
        decompress_run_counter_timestamp (int): Timestamp of decompression runs.
        lossy_weights_handover_class (LossyWeightsHandoverClass): LossyWeightsHandoverClass instance.
            Tracks previous weights
        saved_model_file_names_and_sizes_from_folder (str): Path to saved model file names and sizes.
        combined_algorithm_name (str): Combined algorithm name.
        combined_run_name (str): Combined run name with algorithm name
        combined_run_name_with_hash (str): Combined run name with algorithm name and hash.
        path_to_model_folder (str): Path to model folder where weights are saved to.
        testing (bool): Flag indicating if pipeline is testing.
        lossy (bool): Flag indicating if pipeline is lossy. Inferred by used runners.
        weights_can_repeat (bool): Flag indicating if weights can repeat. Inferred by used runners.
        reset_saver_for_dif (bool): Flag indicating if reset saver is used. Inferred by used runners.
    """
    weights_last_run: []

    def __init__(self, name_online_training_run: str, path_to_overall_model_folder: str,
                 runners: [Compress_With_Measurements], debug_asserts=True):

        file_writer = None
        for runner in reversed(runners):
            if runner.reset_saver_for_dif:
                continue
            if runner.file_saver:
                file_writer = runner
                break

            raise Exception('last runner must be file saver')

        if not os.path.exists(path_to_overall_model_folder):
            raise Exception(f'overall path {path_to_overall_model_folder} does not exist')

        self.name_online_training_run = name_online_training_run
        self.path_to_overall_model_folder = path_to_overall_model_folder
        self.runners: [Compress_With_Measurements] = runners
        self.debug_asserts = debug_asserts
        self.measurements_finished_and_loaded_to_class = False
        self.loss_metric_during_training = []
        self.metrics = []
        self.metrics_last_batch = []
        self.saved_models_count = 0
        self.timestamps_of_saves = []
        self.timestamps_of_runs = []
        self.compression_time_from_runners = []
        self.decompression_time_from_runners = []
        self.training_time = []
        self.compression_sizes_from_runners = []
        self.decompress_run_counter = 0
        self.decompress_run_counter_timestamp = 0
        self.lossy_weights_handover_class = LossyWeightsHandoverClass()
        self.saved_model_file_names_and_sizes_from_folder = None

        self.combined_algorithm_name = '-'.join(list(map(lambda runner: runner.algorithm_name(), runners)))
        self.combined_run_name = '~'.join([self.combined_algorithm_name, self.name_online_training_run])
        combined_run_name_for_hashing_with_legacy_errors = self.combined_run_name.replace('LSTM', 'LTSM')
        self.combined_run_name_hash = hashlib.sha256(
            combined_run_name_for_hashing_with_legacy_errors.encode('utf-8')).hexdigest()

        self.path_to_model_folder = f'{self.path_to_overall_model_folder}/{self.combined_run_name_hash}'
        self.combined_run_name_with_hash = f'{self.combined_run_name}({self.combined_run_name_hash})'

        if not os.path.exists(self.path_to_model_folder):
            os.makedirs(self.path_to_model_folder)

        file_writer.path_to_file = self.path_to_model_folder

        self.testing = any(map(lambda x: x.testing, runners))
        self.lossy = any(map(lambda x: x.lossy, runners))
        self.weights_can_repeat = any(map(lambda x: x.weights_can_repeat, runners))
        self.reset_saver_for_dif = any(map(lambda x: x.reset_saver_for_dif, runners))
        if self.testing:
            print('pipe_is_testing')
        if self.lossy and self.testing:
            print('pipe_is_lossy')
        if self.weights_can_repeat and self.testing:
            print('pipe_weights_can_repeat')

        self.set_lossy_weights_handover_to_lowest_if_in_and_out_runner_present()

    def set_lossy_weights_handover_to_lowest_if_in_and_out_runner_present(self):
        last_lossy_runner_found = False
        for compress in reversed(self.runners):
            if compress.reset_saver_for_dif or compress.requires_pipeline:
                compress.add_runners(self.runners)

            if not compress.lossy or compress.weights_can_repeat:
                compress.add_lossy_weights_handover(self.lossy_weights_handover_class)
            elif not last_lossy_runner_found:
                compress.add_lossy_weights_handover(self.lossy_weights_handover_class)
                self.lossy_weights_handover_class.is_weights_from_original_run_saved_and_used = False
                last_lossy_runner_found = True
            else:
                compress.add_lossy_weights_handover(self.lossy_weights_handover_class)

    def load_this_class_from_dump_file_if_already_run(self):
        if not os.path.exists(self.get_dump_file_path()):
            return None

        loaded_class = self.load_this_class_from_dump_file()

        if loaded_class is None:
            return None

        try:
            loaded_class.get_model_filenames_and_size_and_assert_if_missing()
            return loaded_class
        except AssertionError:
            return None

    def get_dump_file_path(self):
        return f'{self.path_to_model_folder}.dmp'

    def save_objects_for_measurements_and_reset_runners(self, save_to_file):
        for runner in self.runners:
            name = runner.algorithm_name()
            self.compression_sizes_from_runners.append((name, runner.compression_sizes))

        self.reset_measurements_for_runners()
        self.save_model_file_names_and_sizes_from_model_folder()
        self.measurements_finished_and_loaded_to_class = True
        if save_to_file:
            try:
                with open(self.get_dump_file_path(), 'wb') as f:
                    pickle.dump(self, f)
                self.remove_folder_with_saved_weights()
            except Exception as e:
                print(f'Error while saving, run was not saved, see description: {e}')

    def remove_folder_with_saved_weights(self):
        assert self.is_model_file_names_and_sizes_saved()
        shutil.rmtree(self.path_to_model_folder)
        print(f'Removed class saved model folder: {self.combined_run_name_with_hash}')

    def load_this_class_from_dump_file(self) -> Optional['CompressionPipelineRunner']:
        try:
            with open(self.get_dump_file_path(), 'rb') as f:
                loaded_class: 'CompressionPipelineRunner' = pickle.load(f)

            if not loaded_class.is_model_file_names_and_sizes_saved():
                loaded_class.save_model_file_names_and_sizes_from_model_folder()
                try:
                    with open(loaded_class.get_dump_file_path(), 'wb') as f:
                        pickle.dump(loaded_class, f)
                    print(f'Resaved class with new data: {loaded_class.combined_run_name_with_hash}')
                    loaded_class.remove_folder_with_saved_weights()
                except Exception as e:
                    print(f'Error while saving updated file params, run was not saved, see description: {e}')
            return loaded_class

        except Exception as e:
            print(f'Error loading class, has to run again, see description: {e}')
            return None

    def compress_and_decompress_and_verify_if_testing(self, value_in, data_of_current_batch, loss_numb):
        if self.debug_asserts:
            percentage_same = percentage_layers_same(self.weights_last_run, value_in)
            assert percentage_same < 0.9, f'weights should change for each learn step, percentage: {percentage_same}'
            self.weights_last_run = value_in

        if loss_numb:
            self.loss_metric_during_training.append(loss_numb)

        run_params = {
            RunParamKeys.TRAINING_DATA: data_of_current_batch,
            RunParamKeys.RUN_NUMBER_TO_USE: len(self.timestamps_of_saves),
            RunParamKeys.LOSS: loss_numb,
        }
        self.compress_and_measure(value_in, run_params)

        if self.lossy_weights_handover_class.is_weights_from_original_run_saved_and_used:
            self.lossy_weights_handover_class.weights_from_last_run_compress = value_in

        if self.testing:
            # long runtime, therefore only executed when testing
            self.verify_and_assert_compressed_weights_by_decompression(run_params)

    def verify_and_assert_compressed_weights_by_decompression(self, run_params_of_compression_run):
        compression_not_saved = RunParamKeys.PIPELINE_FINISHED in run_params_of_compression_run \
                                and run_params_of_compression_run[RunParamKeys.PIPELINE_FINISHED]

        value_expected = self.lossy_weights_handover_class.weights_from_last_run_compress
        if self.weights_can_repeat or self.reset_saver_for_dif:
            value_out = self.decompress_and_measure_last_save()
        else:
            run_params = {}
            value_out = self.decompress_and_measure(run_params)

        if not compression_not_saved:
            np.testing.assert_equal(value_out, value_expected)

    def compress_and_measure(self, value_in, run_params):

        run_params[RunParamKeys.PROCESS_TIME_PER_COMPRESSOR] = []
        compression_was_aborted = False
        for compress in self.runners:
            value_in = compress.compress_and_measure(value_in, run_params)
            if RunParamKeys.PIPELINE_FINISHED in run_params and run_params[RunParamKeys.PIPELINE_FINISHED]:
                compression_was_aborted = True
                if self.testing:
                    print(f'Stop compression Pipeline@{type(compress).__name__}')
                break

        time_stamp_after_run = time.time_ns()
        self.timestamps_of_runs.append(time_stamp_after_run)
        if not compression_was_aborted:
            self.saved_models_count += 1
            self.timestamps_of_saves.append(time_stamp_after_run)

        self.compression_time_from_runners.append(
            run_params[RunParamKeys.PROCESS_TIME_PER_COMPRESSOR])

    def get_next_timestamp_of_saves_safe(self, before_next):
        next_count = 1 + before_next
        if next_count >= len(self.timestamps_of_saves):
            if self.testing:
                print('last decompress next, using infinity')
            next_time_stamp_of_save = sys.maxsize
        else:
            next_time_stamp_of_save = self.timestamps_of_saves[next_count]

        return next_time_stamp_of_save

    def decompress_and_measure_next(self, run_params=None):
        time_stamp_of_run_to_load = self.timestamps_of_runs[self.decompress_run_counter]
        next_time_stamp_of_save = self.get_next_timestamp_of_saves_safe(self.decompress_run_counter_timestamp)
        if time_stamp_of_run_to_load >= next_time_stamp_of_save:
            self.decompress_run_counter_timestamp += 1

        if self.testing:
            time_stamp_of_selected_run_to_load = self.timestamps_of_saves[self.decompress_run_counter_timestamp]
            next_time_stamp_of_save = self.get_next_timestamp_of_saves_safe(self.decompress_run_counter_timestamp)
            assert time_stamp_of_selected_run_to_load <= time_stamp_of_run_to_load < next_time_stamp_of_save, \
                f'to_load: {self.decompress_run_counter:4.0f}, {time_stamp_of_run_to_load} ' \
                f'| selected: {self.decompress_run_counter_timestamp:4.0f}, {time_stamp_of_selected_run_to_load}' \
                f'| {self.timestamps_of_runs} {self.timestamps_of_saves}'
            assert self.decompress_run_counter_timestamp < len(self.timestamps_of_saves)

        self.decompress_run_counter += 1

        testing_and_dif_reset_present_and_run_selected = \
            self.testing and self.reset_saver_for_dif \
            and self.decompress_run_counter == math.floor(len(self.timestamps_of_runs) / 2)
        if testing_and_dif_reset_present_and_run_selected:
            self.assert_weights_loaded_from_both_directions_same(self.decompress_run_counter_timestamp)

        return self.decompress_and_measure_run_number(self.decompress_run_counter_timestamp, run_params)

    def assert_weights_loaded_from_both_directions_same(self, iteration_to_check):
        run_params = {RunParamKeys.DIFF_RESET_LOAD_DIRECTION: LoadDirection.FORWARD,
                      RunParamKeys.DONT_SAVE_PROCESSING_TIME: True}
        decompressed_weights_forward = self.decompress_and_measure_run_number(iteration_to_check, run_params)

        run_params = {RunParamKeys.DIFF_RESET_LOAD_DIRECTION: LoadDirection.BACKWARD,
                      RunParamKeys.DONT_SAVE_PROCESSING_TIME: True}
        decompressed_weights_backward = self.decompress_and_measure_run_number(iteration_to_check, run_params)

        same_between_f_and_b = percentage_layers_same(decompressed_weights_forward, decompressed_weights_backward)
        assert same_between_f_and_b == 1, \
            f'weights change from loading direction, diff is {same_between_f_and_b} from iteration {iteration_to_check}'

    def decompress_and_measure_last_save(self):
        save_to_load = len(self.timestamps_of_saves) - 1

        return self.decompress_and_measure_run_number(save_to_load)

    def decompress_and_measure_run_number(self, run_number, run_params=None):
        if run_params is None:
            run_params = {}

        run_params[RunParamKeys.RUN_NUMBER_TO_USE] = run_number

        return self.decompress_and_measure(run_params)

    def load_weights_at_timestamp(self, timestamp: int):
        run_number_of_timestamp = self.get_run_number_for_timestamp(timestamp)
        return self.decompress_and_measure_run_number(run_number_of_timestamp)

    def get_run_number_for_timestamp(self, timestamp):
        size_of_list = len(self.timestamps_of_saves)
        max_runs = math.ceil(math.log(size_of_list)) + 1
        run_count = 0

        next_step_size = math.ceil(size_of_list / 2)
        run_to_check = next_step_size

        while run_count <= max_runs:
            first_interval = False
            last_interval = False
            if run_to_check <= 0:
                run_to_check = 0
                first_interval = True
            elif run_to_check >= (size_of_list - 1):
                run_to_check = size_of_list - 2
                last_interval = True

            timestamp_before_or_equal = self.timestamps_of_saves[run_to_check]
            timestamp_after = self.timestamps_of_saves[run_to_check + 1]

            next_step_size = math.ceil(next_step_size / 2)
            if timestamp_before_or_equal <= timestamp:
                if timestamp < timestamp_after:
                    if self.testing:
                        print(f'Found timestamp {timestamp} for run {run_to_check} after {run_count}/{max_runs} tries')
                    return run_to_check
                elif last_interval:
                    return run_to_check + 1
                else:
                    run_to_check += next_step_size
            elif first_interval:
                raise LookupError(f'given timestamp {timestamp} before first save {timestamp_before_or_equal}')
            else:
                run_to_check -= next_step_size

            run_count += 1

        raise AssertionError(f'run too long ({run_count}/{max_runs})')

    def decompress_and_measure(self, run_params):
        run_params[RunParamKeys.PROCESS_TIME_PER_COMPRESSOR] = []

        value_out = None
        for decompress in reversed(self.runners):
            value_out = decompress.decompress_and_measure(value_out, run_params)
            if RunParamKeys.PIPELINE_FINISHED in run_params and run_params[RunParamKeys.PIPELINE_FINISHED]:
                if self.testing:
                    print(f'Stop decompression Pipeline@{type(decompress).__name__}')
                break

        save_processing_time = not (RunParamKeys.DONT_SAVE_PROCESSING_TIME in run_params
                                    and run_params[RunParamKeys.DONT_SAVE_PROCESSING_TIME])

        if save_processing_time:
            self.decompression_time_from_runners.append(
                run_params[RunParamKeys.PROCESS_TIME_PER_COMPRESSOR])

        return value_out

    def compression_call_count(self):
        return len(self.timestamps_of_runs)

    def reset(self, model):
        self.decompress_run_counter = 0
        self.decompress_run_counter_timestamp = 0

        self.weights_last_run = model.get_weights()
        self.lossy_weights_handover_class.reset_weights(self.weights_last_run)
        for compress in self.runners:
            compress.reset(model)

    def reset_measurements_for_runners(self):
        for compress in self.runners:
            compress.reset_measurements_and_other_after_run()

    def get_next_accuracies(self):
        return list(map(lambda x: x['accuracy'], self.metrics))

    def get_last_accuracies(self):
        return list(map(lambda x: x[1], self.metrics_last_batch))

    def get_measurements(self):
        compression_time_total = self.get_time(self.compression_time_from_runners)
        decompression_time_total = self.get_time(self.decompression_time_from_runners)
        number_to_compare_to = len(self.timestamps_of_runs) if self.weights_can_repeat else self.saved_models_count
        assert len(compression_time_total) == number_to_compare_to

        number_of_decompression_runs = len(decompression_time_total) / number_to_compare_to
        assert number_of_decompression_runs % 1 == 0 and number_of_decompression_runs <= 2

        if number_of_decompression_runs == 2:
            half_of_decompression_measurements = int(len(decompression_time_total) / number_of_decompression_runs)
            decompression_time_total_only_from_real_test_run = decompression_time_total[
                                                               half_of_decompression_measurements:]
        else:
            decompression_time_total_only_from_real_test_run = decompression_time_total

        assert len(decompression_time_total_only_from_real_test_run) == number_to_compare_to

        return {
            DISK_USAGE_MEASUREMENT: self.get_disk_space_usage(),
            ACC_NEXT_MEAN_MEASUREMENT: 1 - np.mean(self.get_next_accuracies()),
            ACC_LAST_MEAN_MEASUREMENT: 1 - np.mean(self.get_last_accuracies()),
            COMPRESSION_TIME_MEAN_MEASUREMENT: np.mean(compression_time_total),
            DECOMPRESSION_TIME_MEAN_MEASUREMENT: np.mean(decompression_time_total_only_from_real_test_run),
            DECOMPRESSION_TIME_MAX_MEASUREMENT: np.max(decompression_time_total_only_from_real_test_run),
            COMPRESSION_TIME_MAX_MEASUREMENT: np.max(compression_time_total),
        }

    def get_time(self, array_with_tuple_name_measurement):
        return list(map(lambda x: self.get_time_sum_per_execution(x), array_with_tuple_name_measurement))

    def get_time_sum_per_execution(self, array_array):
        return sum(list(map(lambda x: x[1], array_array)))

    def get_disk_space_usage(self):
        disk_space_bytes = sum(map(lambda a: a[1], self.get_model_filenames_and_size_and_assert_if_missing()))
        disk_space_k_bytes = disk_space_bytes / pow(10, 3)
        return disk_space_k_bytes

    def get_file_sizes_of_full_and_diff_saves(self):
        full = []
        diff = []

        def add_filesize_to_full_or_diff(dir_entry):
            name, size = dir_entry
            if 'full' in name:
                full.append(size)
            else:
                diff.append(size)

        [add_filesize_to_full_or_diff(f) for f in self.get_model_filenames_and_size_and_assert_if_missing()]

        return full, diff

    def get_model_filenames_and_size_and_assert_if_missing(self):
        if not self.is_model_file_names_and_sizes_saved():
            self.save_model_file_names_and_sizes_from_model_folder()

        return self.saved_model_file_names_and_sizes_from_folder

    def save_model_file_names_and_sizes_from_model_folder(self):
        file_names_and_sizes = self.get_model_filenames_and_sizes_from_model_folder()
        assert file_names_and_sizes is not None
        self.saved_model_file_names_and_sizes_from_folder = file_names_and_sizes

    def get_model_filenames_and_sizes_from_model_folder(self):
        if not os.path.exists(self.path_to_model_folder):
            return None

        with os.scandir(self.path_to_model_folder) as model_files:
            file_names_and_sizes = [(m_file.name, os.path.getsize(m_file)) for m_file in model_files]
            if file_names_and_sizes is not None and len(file_names_and_sizes) > 1:
                return file_names_and_sizes

    def is_model_file_names_and_sizes_saved(self):
        return hasattr(self, 'saved_model_file_names_and_sizes_from_folder') \
            and self.saved_model_file_names_and_sizes_from_folder is not None \
            and len(self.saved_model_file_names_and_sizes_from_folder) > 0

    def get_size_prop_difference_of_full_and_diff_save(self):
        if not self.reset_saver_for_dif:
            return None

        full, diff = self.get_file_sizes_of_full_and_diff_saves()

        return mean(full) / mean(diff)

    def append_to_metrics(self, metric):
        self.metrics.append(metric)

    def append_to_metrics_last_batch(self, metric):
        self.metrics_last_batch.append(metric)

    def append_to_training_times(self, batch_training_time):
        self.training_time.append(batch_training_time)

    def get_training_times(self):
        # first excluded since start time very high
        return self.training_time[1:]
