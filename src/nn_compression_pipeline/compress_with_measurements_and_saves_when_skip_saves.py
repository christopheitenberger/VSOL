from src.nn_compression_pipeline import Compress_With_Measurements, RunParamKeys


class CompressWithMeasurementsAndSavesWhenSkipSaves(Compress_With_Measurements):
    lossy = True
    weights_can_repeat = True
    requires_pipeline = True

    def __init__(self, full_diff=False):
        super().__init__()
        self.full_diff = full_diff

        self.dif_combiner = None

        self.last_combined = None

    def add_runners(self, runners):
        assert any(map(lambda x: x.dif_combiner, runners)), 'No DifCombiner runner, therefore DifResetSaver unnecessary'
        self.dif_combiner = list(filter(lambda x: x.dif_combiner, runners))[0]

    def compress_operations_on_in_and_out_values(self, value_in, value_out, run_params):
        pipeline_finished = RunParamKeys.PIPELINE_FINISHED in run_params and run_params[RunParamKeys.PIPELINE_FINISHED]
        # acts like diffing between last saved weights and current learned weights with lossy
        if pipeline_finished and not self.full_diff:
            self.last_lossy_layer_saved.reset_to_previous_weights()

        # saves all weights inbetween as ony save by combining each diff
        if self.full_diff:
            if pipeline_finished:
                if self.last_combined is None:
                    self.last_combined = value_in
                else:
                    self.last_combined = self.combine_with_last_combined(value_in)
            elif self.last_combined:
                value_out = self.combine_with_last_combined(value_in)
                self.last_combined = None

        return value_out

    def combine_with_last_combined(self, value_in):
        zipped = zip(self.last_combined, value_in)
        return self.dif_combiner.combine_two_models(zipped, self.dif_combiner.combine_algorithm)

    def reset_other(self):
        super().reset_other()
        self.last_combined = None

    def algorithm_name(self, dic_params=None):
        if dic_params is None:
            dic_params = {}
        dic_params.update({'fd': self.full_diff})
        return super().algorithm_name(dic_params)
