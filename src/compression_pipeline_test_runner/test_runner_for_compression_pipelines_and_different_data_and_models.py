import math

import numpy as np
import pandas as pd
from tqdm.contrib.itertools import product

from src.compression_pipeline_test_runner.compression_keras_callbacks import CompressModel, DecompressModelCallback
from src.compression_pipeline_test_runner.compression_pipeline_runner import CompressionPipelineRunner
from src.models_with_training_and_evaluation import ModelWrapperAndLoader
from src.nn_compression_pipeline import Compress_With_Measurements
from visualization_utils import ALGORITHM_NAME_COL, RUN_NAME_COL, NN_TYPE_COL


class TestRunnerForCompressionPipelinesAndDifferentDataAndModels:
    """
    Class to evaluate compression pipelines with different models.
    First executes compression during online learning measuring compression time and used storage space.
    Secondly executes decompression during loading the saved models measuring decompression time and next batch accuracy.
    Creates CompressionPipelineRunner for algorithm combinations
    Able to output data for all CompressionPipelineRunner


    """
    text_break = '\n\n--- '

    def __init__(self, path_to_overall_model_folder: str,
                 model_wrapper_with_data: ModelWrapperAndLoader,
                 compressionRunnerCombinations: [[Compress_With_Measurements]], verbose_lvl=1):
        self.model_wrapper_with_data = model_wrapper_with_data
        self.path_to_overall_model_folder = path_to_overall_model_folder
        self.override_online_runs = False
        self.compressionRunnerCombinations = compressionRunnerCombinations
        self.verbose_lvl = verbose_lvl

        self.compression_pipeline_runners = {}

    def add_compression_pipeline_runner_to_dict(self, compression_pipeline_run: CompressionPipelineRunner):
        combined_algorithm_name = compression_pipeline_run.combined_algorithm_name
        if combined_algorithm_name not in self.compression_pipeline_runners.keys():
            self.compression_pipeline_runners[combined_algorithm_name] = []

        self.compression_pipeline_runners[combined_algorithm_name].append(compression_pipeline_run)

    def set_label_for_data_split_on_off_and_return_online_run_name(self, class_label):
        data_split_on_off = self.model_wrapper_with_data.data_split
        data_split_on_off.set_selected_group(class_label)
        self.gen = data_split_on_off.get_online_training_batch_as_epoch_sequence()
        self.get_val = data_split_on_off.get_online_validation_batch_as_epoch_sequence()
        self.val_seq = data_split_on_off.get_online_validation_sequence()
        self.val_last_batch = data_split_on_off.get_online_validation_on_last_batch_sequence()

        return data_split_on_off.get_name_for_online_training_run()

    def create_compression_pipeline_runner(self, runners: [Compress_With_Measurements],
                                           path_to_overall_model_folder: str,
                                           name_online_training_run: str) -> CompressionPipelineRunner:
        return CompressionPipelineRunner(name_online_training_run, path_to_overall_model_folder, runners)

    def run(self, override_online_runs=False):
        self.override_online_runs = override_online_runs
        all_class_labels = self.model_wrapper_with_data.data_split.get_all_class_labels()

        last_class_label = None
        name_online_training_run = None
        for class_label, compression_runner_combination in product(all_class_labels,
                                                                   self.compressionRunnerCombinations):
            if last_class_label != class_label:
                last_class_label = class_label
                name_online_training_run = self.set_label_for_data_split_on_off_and_return_online_run_name(class_label)

            compression_pipeline_run = self.create_and_run_compression_pipeline_runner(
                compression_runner_combination, name_online_training_run)
            self.add_compression_pipeline_runner_to_dict(compression_pipeline_run)

    def create_and_run_compression_pipeline_runner(
            self, compression_runner_combination: [Compress_With_Measurements],
            name_online_training_run: str) -> CompressionPipelineRunner:
        compression_pipelines_to_test = self.create_compression_pipeline_runner(
            compression_runner_combination, self.path_to_overall_model_folder, name_online_training_run)
        name_of_run = compression_pipelines_to_test.combined_run_name_with_hash

        if not self.override_online_runs and not compression_pipelines_to_test.testing:
            if compression_pipelines_to_test.measurements_finished_and_loaded_to_class:
                print(f'{self.text_break}Skip {name_of_run} because already run, already loaded')
                return compression_pipelines_to_test
            loaded_class_if_was_present = compression_pipelines_to_test.load_this_class_from_dump_file_if_already_run()
            if loaded_class_if_was_present is not None:
                print(f'{self.text_break}Skip {name_of_run} because already run, now loading...')
                return loaded_class_if_was_present

        batch_size, number_of_online_batches = self.model_wrapper_with_data.data_split.get_batch_size_and_number_of_online_batches()

        print(f'{self.text_break}Executing training for {name_of_run}')
        model = self.reset_all(compression_pipelines_to_test)
        assert number_of_online_batches <= self.gen.epoch_len(), (number_of_online_batches, self.gen.epoch_len())

        history = model.fit(self.gen, batch_size=batch_size, epochs=number_of_online_batches,
                            validation_data=self.get_val,
                            verbose=self.verbose_lvl, shuffle=False,
                            callbacks=[CompressModel(compression_pipelines_to_test, self.gen)]
                            )

        print(f'{self.text_break}Executing access and performance measurements for {name_of_run}')
        model = self.reset_all(compression_pipelines_to_test)

        decompress_model_callback = DecompressModelCallback(compression_pipelines_to_test, self.val_last_batch)
        history = model.evaluate(self.val_seq, batch_size=batch_size,
                                 verbose=self.verbose_lvl, callbacks=[decompress_model_callback])

        print(f'{self.text_break}Saving measurements for {name_of_run}')
        save_to_file = not compression_pipelines_to_test.testing
        compression_pipelines_to_test.save_objects_for_measurements_and_reset_runners(save_to_file)

        return compression_pipelines_to_test

    def reset_all(self, compression_pipelines_to_test: CompressionPipelineRunner):
        model = self.model_wrapper_with_data.load_weights_from_folder_or_train_if_missing()
        self.gen.reset_sequence()
        self.get_val.reset_sequence()
        self.val_last_batch.reset_sequence()
        compression_pipelines_to_test.reset(model)
        self.model_wrapper_with_data.data_split.set_random_seed_combined()

        return model

    def get_model_name(self):
        return self.model_wrapper_with_data.data_split.model_name

    def traverse_algs(self):
        for runs_of_alg in self.compression_pipeline_runners.values():
            for run in runs_of_alg:
                yield run

    def get_mean_and_max_training_time_in_s(self):
        training_times_per_execution = np.array([run.get_training_times() for run in self.traverse_algs()])
        training_times_per_execution_mean = training_times_per_execution.mean(axis=-1)
        training_times_per_execution_max = training_times_per_execution.max(axis=-1)

        def mean_and_from_ms_to_s(v):
            return v.mean() / math.pow(10, 9)

        return mean_and_from_ms_to_s(training_times_per_execution_mean), \
            mean_and_from_ms_to_s(training_times_per_execution_max)

    def get_metrics_df(self, different_alg_names=None):
        rows = []
        columns = None

        if different_alg_names is None:
            alg_names_with_runners = self.compression_pipeline_runners.items()
        else:
            assert len(different_alg_names) == len(self.compression_pipeline_runners), \
                (len(different_alg_names), len(self.compression_pipeline_runners))
            alg_names_with_runners = zip(different_alg_names, self.compression_pipeline_runners.values())

        for algorithm_name, list_of_runners in alg_names_with_runners:
            for runner in list_of_runners:
                runner_measurements = runner.get_measurements()
                if not columns:
                    columns = [ALGORITHM_NAME_COL, RUN_NAME_COL, NN_TYPE_COL] + list(runner_measurements.keys())

                nn_type = runner.name_online_training_run.split('#', 1)[0]
                run_related_names = [algorithm_name, runner.name_online_training_run, nn_type]
                row = run_related_names + list(runner_measurements.values())
                rows.append(row)

        df = pd.DataFrame(data=rows, columns=columns)

        return df
