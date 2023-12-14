from src.nn_compression_pipeline import Compress_With_Measurements


class SkipSavedModelsDuringEvaluation(Compress_With_Measurements):
    """
    Used to measure which impact on e.g. acc it would have if models would be rolled out not after each training
    """
    lossy = True
    weights_can_repeat = True

    # testing = True
    def __init__(self, number_of_runs_to_not_load):
        super().__init__()
        self.number_of_runs_to_not_load = number_of_runs_to_not_load
        self.run_counter = 0
        self.weights_from_used_run = None

    def decompress(self, value_in, run_params):
        skip_run_weight_loading = (self.run_counter % (self.number_of_runs_to_not_load + 1)) != 0
        if skip_run_weight_loading:
            value_out = self.weights_from_used_run
        else:
            self.weights_from_used_run = value_in
            value_out = value_in

        self.run_counter += 1
        return self.decompress_operations_on_in_and_out_values(value_in, value_out, run_params)

    def reset_other(self):
        super().reset_other()
        self.run_counter = 0
        self.weights_from_used_run = None

    def algorithm_name(self, dic_params=None):
        return super().algorithm_name({'nnl': self.number_of_runs_to_not_load})
