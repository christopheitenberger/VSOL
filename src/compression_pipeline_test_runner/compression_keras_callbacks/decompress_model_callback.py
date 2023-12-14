from keras import callbacks

from ..compression_pipeline_runner import CompressionPipelineRunner
from src.np_utils import percentage_layers_same


class DecompressModelCallback(callbacks.Callback):
    """
    Callback for decompressing and evaluating model during testing.
    Changes the model weights after each batch to simulate the weights during online learning.

    Attributes:
        compression_pipeline (CompressionPipelineRunner): The compression pipeline runner object.
        val_last_batch: The validation last batch for getting last batch accuracy
        runs_weights_did_not_change (list): A list to store the percentage of layers that remained the same after decompression. Used for debugging.

    """

    def __init__(self, compression_pipeline: CompressionPipelineRunner, val_last_batch):
        super().__init__()
        self.call_count = 0
        self.compression_pipeline = compression_pipeline
        self.val_last_batch = val_last_batch
        self.val_last_batch.number_of_batches_override = 1

        self.runs_weights_did_not_change = []

    def on_test_batch_begin(self, batch, logs=None):
        decompressed_weights = self.compression_pipeline.decompress_and_measure_next()

        self.runs_weights_did_not_change.append(percentage_layers_same(decompressed_weights, self.model.get_weights()))

        self.model.set_weights(decompressed_weights)
        self.call_count += 1

    def on_test_batch_end(self, batch, logs=None):
        self.append_metric_to_pipeline(logs)
        history = self.model.evaluate(self.val_last_batch, batch_size=self.val_last_batch.batch_size, verbose=2)
        self.compression_pipeline.append_to_metrics_last_batch(history)

    def append_metric_to_pipeline(self, logs):
        self.compression_pipeline.append_to_metrics(logs)

    def on_test_end(self, logs=None):
        self.assert_number_of_compression_and_decompression_calls_correct()
        self.call_count = 0

        if not self.compression_pipeline.weights_can_repeat:
            self.assert_weights_have_changed()
        else:
            print(self.runs_weights_did_not_change)

    def assert_number_of_compression_and_decompression_calls_correct(self):
        assert (self.compression_pipeline.decompress_run_counter_timestamp + 1) == \
               len(self.compression_pipeline.timestamps_of_saves)
        call_count_pipeline = self.compression_pipeline.compression_call_count()
        assert call_count_pipeline == self.call_count, \
            f'model_count: {call_count_pipeline} real_count: {self.call_count}'

    def assert_weights_have_changed(self):
        sum_changed_layers = sum([x < 1 for x in self.runs_weights_did_not_change])
        len_changed_layers = len(self.runs_weights_did_not_change)
        assert len_changed_layers == sum_changed_layers or sum_changed_layers == 0, \
            f'weights for some runs have changed {sum_changed_layers}/{len_changed_layers} ' \
            f'{self.runs_weights_did_not_change}'
