import math
from enum import IntEnum

from src.nn_compression_pipeline import Compress_With_Measurements, RunParamKeys
from src.np_utils import percentage_layers_same


class LoadDirection(IntEnum):
    DYNAMIC = 0,
    FORWARD = 1,
    BACKWARD = 2,


class DifResetSaver(Compress_With_Measurements):
    reset_saver_for_dif = True

    # testing = True

    # max_decompression_time_in_s not deterministic
    def __init__(self, max_decompression_time_in_s=None, fixed_number_of_iterations=None):
        super().__init__()

        time_present = max_decompression_time_in_s is not None
        iterations_present = fixed_number_of_iterations is not None

        assert time_present != iterations_present, 'only one parameter should be set instead of both or non'

        self.max_decompression_time_in_s = max_decompression_time_in_s
        self.fixed_number_of_iterations = fixed_number_of_iterations

        # same as reset_other
        self.runs_without_saving_dif = 0
        self.current_max_total_compression_time_for_runners = 0
        self.iteration_numb_when_reset_point_to_create = self.fixed_number_of_iterations
        self.save_file_name_counter = 0

        self.runners_without_this_class: list[Compress_With_Measurements] = None
        self.runners_for_full_diff: list[Compress_With_Measurements] = None
        self.runners_name_for_decompression_time: list[str] = None

    def add_runners(self, runners):
        assert any(map(lambda x: x.dif_combiner, runners)), 'No DifCombiner runner, therefore DifResetSaver unnecessary'
        self.runners_without_this_class = list(filter(lambda x: not x.reset_saver_for_dif, runners))
        self.runners_name_for_decompression_time = list(
            map(lambda x: x.algorithm_name(),
                filter(lambda x: not x.lossy, self.runners_without_this_class)))
        self.runners_for_full_diff = list(
            filter(lambda x: not (x.lossy or x.dif_combiner), self.runners_without_this_class))

    def compress(self, value_in, run_params):
        if self.save_file_name_counter == 0:
            self.save_full_diff()
            return

        self.runs_without_saving_dif += 1
        if self.iteration_numb_when_reset_point_to_create is None:
            self.save_iteration_numb_if_max_compression_time_overstepped(run_params)

        if self.iteration_numb_when_reset_point_to_create is not None \
                and self.runs_without_saving_dif >= self.iteration_numb_when_reset_point_to_create:
            self.save_full_diff()

    def decompress(self, value_in, run_params):
        if run_params is None or RunParamKeys.RUN_NUMBER_TO_USE not in run_params:
            return None

        load_direction = LoadDirection.DYNAMIC
        if RunParamKeys.DIFF_RESET_LOAD_DIRECTION in run_params:
            load_direction = run_params[RunParamKeys.DIFF_RESET_LOAD_DIRECTION]
            assert isinstance(load_direction, LoadDirection)

        run_params[RunParamKeys.PIPELINE_FINISHED] = True

        run_number = run_params[RunParamKeys.RUN_NUMBER_TO_USE]

        dif_res_to_use = 0
        run_recreated = 0
        if self.iteration_numb_when_reset_point_to_create is not None:
            dif_res_to_use_in_between = run_number / self.iteration_numb_when_reset_point_to_create
            dif_res_to_use = self.round_with_dirction_for_load_direction(dif_res_to_use_in_between, load_direction)
            use_last_dif_since_next_does_not_exists = self.save_file_name_counter <= dif_res_to_use
            if use_last_dif_since_next_does_not_exists:
                dif_res_to_use -= 1
            run_recreated = dif_res_to_use * self.iteration_numb_when_reset_point_to_create

        param_dic = {RunParamKeys.FILE_NAME_MIDDLE_OVERRIDE: self.get_file_name_middle(dif_res_to_use)}
        nearest_full_run_diff = None
        for runner in reversed(self.runners_for_full_diff):
            nearest_full_run_diff = runner.decompress(nearest_full_run_diff, param_dic)

        self.last_lossy_layer_saved.set_last_decompress_weights(nearest_full_run_diff)

        if run_recreated == run_number:
            if self.testing:
                print(f'{run_recreated} diff recreate directly')
            return nearest_full_run_diff

        if run_recreated > run_number:
            diffs_to_load_for_recreation = list(reversed(range(run_number + 1, run_recreated + 1)))
        else:
            diffs_to_load_for_recreation = list(range(run_recreated + 1, run_number + 1))

        if self.testing or load_direction != LoadDirection.DYNAMIC:
            print(f'checkp: {run_recreated}, diffs: {diffs_to_load_for_recreation}, '
                  f'run_number: {run_number}, | '
                  f'({dif_res_to_use} * ({self.iteration_numb_when_reset_point_to_create})), ')

        value_last_test = nearest_full_run_diff

        value_out = None
        for diff_run_to_load in diffs_to_load_for_recreation:
            param_dic = {RunParamKeys.RUN_NUMBER_TO_USE: diff_run_to_load}
            for runner in reversed(self.runners_without_this_class):
                value_out = runner.decompress(value_out, param_dic)

            if self.testing:
                same = percentage_layers_same(value_last_test, value_out)
                assert same < 1
                value_last_test = value_out

        # fixed issue where decompress and comrpess different
        # self.last_lossy_layer_saved.set_last_weights_and_call_subscribers(value_out)

        return value_out

    def round_with_dirction_for_load_direction(self, dif_res_to_use_in_between, load_direction):
        match load_direction:
            case LoadDirection.DYNAMIC:
                dif_res_to_use = round(dif_res_to_use_in_between)
            case LoadDirection.FORWARD:
                dif_res_to_use = math.ceil(dif_res_to_use_in_between)
            case LoadDirection.BACKWARD:
                dif_res_to_use = math.floor(dif_res_to_use_in_between)
            case _:
                raise TypeError(f'Given load direction {load_direction} not valid')
        return dif_res_to_use

    def save_iteration_numb_if_max_compression_time_overstepped(self, run_params):
        if self.testing:
            assert self.iteration_numb_when_reset_point_to_create is None

        compression_times_from = filter(lambda x: x[0] in self.runners_name_for_decompression_time,
                                        run_params[RunParamKeys.PROCESS_TIME_PER_COMPRESSOR])
        max_compression_time_last_run = sum(map(lambda x: x[1], compression_times_from))

        if self.current_max_total_compression_time_for_runners < max_compression_time_last_run:
            self.current_max_total_compression_time_for_runners = max_compression_time_last_run

        current_estimated_max_decompression_time = self.current_max_total_compression_time_for_runners * \
                                                   round((self.runs_without_saving_dif / 2) + 1)

        if current_estimated_max_decompression_time >= self.max_decompression_time_in_s:
            self.iteration_numb_when_reset_point_to_create = self.runs_without_saving_dif
            if self.testing:
                print(f'---Number of iterations after saving set, following values found:\n'
                      f'Iterations after full save: {self.iteration_numb_when_reset_point_to_create}, \n'
                      f'max time per compression used: {self.current_max_total_compression_time_for_runners}\n'
                      f'estimated_max_decompression_time: '
                      f'{current_estimated_max_decompression_time}/{self.max_decompression_time_in_s} (/target)'
                      f'---')

    def save_full_diff(self, weights_to_save=None):
        if weights_to_save is None:
            weights_to_save = self.last_lossy_layer_saved.weights_from_last_run_compress

        self.runs_without_saving_dif = 0
        last_run_saved_in = weights_to_save
        param_dic = {RunParamKeys.FILE_NAME_MIDDLE_OVERRIDE: self.get_file_name_middle(self.save_file_name_counter)}
        for runner in self.runners_for_full_diff:
            last_run_saved_in = runner.compress(last_run_saved_in, param_dic)

        self.save_file_name_counter += 1

    def get_file_name_middle(self, count):
        return f'full_{count:05}'

    def reset_other(self):
        self.runs_without_saving_dif = 0
        self.current_max_total_compression_time_for_runners = 0

    def reset_measurements_and_other_after_run(self):
        super().reset_measurements_and_other_after_run()
        self.iteration_numb_when_reset_point_to_create = self.fixed_number_of_iterations
        self.save_file_name_counter = 0

    def algorithm_name(self, dic_params=None):
        if self.fixed_number_of_iterations is not None:
            return super().algorithm_name({'fni': self.fixed_number_of_iterations})
        else:
            return super().algorithm_name({'mdt': self.max_decompression_time_in_s})
