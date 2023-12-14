from src.nn_compression_pipeline import RunParamKeys
from src.nn_compression_pipeline.compress_with_measurements_and_saves_when_skip_saves import \
    CompressWithMeasurementsAndSavesWhenSkipSaves


class SkipSavingModels(CompressWithMeasurementsAndSavesWhenSkipSaves):
    lossy = True
    weights_can_repeat = True

    # testing = True
    def __init__(self, number_of_runs_to_save=1, number_of_runs_to_not_load=1, full_diff=False):
        super().__init__(full_diff)
        self.number_of_runs_to_save = number_of_runs_to_save
        self.number_of_runs_to_not_load = number_of_runs_to_not_load

        self.run_counter = 0

    def compress(self, value_in, run_params):
        run_round = self.run_counter % (self.number_of_runs_to_save + self.number_of_runs_to_not_load)
        skip_run_weight_loading = run_round >= self.number_of_runs_to_save

        if self.testing:
            print(run_round, self.number_of_runs_to_save, skip_run_weight_loading)

        if skip_run_weight_loading:
            run_params[RunParamKeys.PIPELINE_FINISHED] = True

        self.run_counter += 1
        return self.compress_operations_on_in_and_out_values(value_in, value_in, run_params)

    def combine_with_last_combined(self, value_in):
        zipped = zip(self.last_combined, value_in)
        return self.dif_combiner.combine_two_models(zipped, self.dif_combiner.combine_algorithm)

    def reset_other(self):
        super().reset_other()
        self.run_counter = 0

    def algorithm_name(self, dic_params=None):
        return super().algorithm_name({
            'nrs': self.number_of_runs_to_save,
            'nnl': self.number_of_runs_to_not_load,
        })
