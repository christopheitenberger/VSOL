from src.nn_compression_pipeline import Compress_With_Measurements


class PassOnInitialWeightsAndIgnoreInput(Compress_With_Measurements):
    """To have acc baseline which should not be undercut since always should improve over no change"""
    lossy = True
    initial_weights: []

    def compress_and_measure(self, value_in, run_params):
        value_out = self.compress(value_in, run_params)
        self.save_fixed_process_time_to_param(1000, run_params)  # so does not mess up order of other measurements
        return value_out

    def compress(self, value_in, run_params):
        return self.compress_operations_on_in_and_out_values(value_in, self.initial_weights, run_params)

    def decompress_and_measure(self, value_in, run_params):
        value_out = self.decompress(value_in, run_params)
        self.save_fixed_process_time_to_param(1000, run_params)  # so does not mess up order of other measurements
        return value_out

    def decompress(self, value_in, run_params):
        return self.decompress_operations_on_in_and_out_values(value_in, self.initial_weights, run_params)

    def reset_weights(self, model):
        self.initial_weights = model.get_weights()
