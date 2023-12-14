from src.nn_compression_pipeline import CompressWithMeasurementsWithLayerInformation


class CompressWithMeasurementsWithLayerInformationAndWeightsLastRun(CompressWithMeasurementsWithLayerInformation):
    def __init__(self):
        super().__init__()

    def compress_operations_on_in_and_out_values(self, value_in, value_out, run_params):
        super().compress_operations_on_in_and_out_values(value_in, value_out, run_params)
        self.last_lossy_layer_saved.set_new_compress_weights(value_in)
        return value_out

    def decompress_operations_on_in_and_out_values(self, value_in, value_out, run_params):
        super().decompress_operations_on_in_and_out_values(value_in, value_out, run_params)
        self.last_lossy_layer_saved.set_last_decompress_weights(value_out)
        return value_out

    def add_lossy_weights_handover(self, lossy_weights_handover_object):
        super().add_lossy_weights_handover(lossy_weights_handover_object)
        if not self.last_lossy_layer_saved.is_weights_from_original_run_saved_and_used:
            self.last_lossy_layer_saved.add_subscriber_to_weight_changes(
                self.recreate_weights_from_last_run_compress_from_last_lossy_weights_and_measure)

    def recreate_weights_from_last_run_compress_from_last_lossy_weights_and_measure(self, run_params):
        self.last_lossy_layer_saved.override_compressed_weights(
            self.recreate_weights_from_last_run_compress_from_last_lossy_weights())

    def recreate_weights_from_last_run_compress_from_last_lossy_weights(self):
        raise NotImplementedError()