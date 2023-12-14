from keras import callbacks
import tensorflow as tf
import time

from ..compression_pipeline_runner import CompressionPipelineRunner


class CompressModel(callbacks.Callback):
    """
    Callback class for compressing model weights and saving them during training.
    Integrates CompressionPipelineRunner into Keras through callbacks.Callback methods on_batch_end and on_batch_begin.

    Attributes:
        compression_pipeline (CompressionPipelineRunner): Instance of CompressionPipelineRunner class
            for compressing model weights.
        training_seq (tf.keras.utils.Sequence): Training sequence for retrieving training data.
    """

    def __init__(self, compression_pipeline: CompressionPipelineRunner,
                 training_seq: tf.keras.utils.Sequence):
        super(CompressModel, self).__init__()
        self.compression_pipeline = compression_pipeline
        self.training_seq: tf.keras.utils.Sequence = training_seq
        self.batch_start_time = None

    def on_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time_ns()

    def on_batch_end(self, batch, logs=None):
        training_time_of_batch = time.time_ns() - self.batch_start_time
        self.compression_pipeline.append_to_training_times(training_time_of_batch)
        self.save_weights(batch, logs['loss'])

    def save_weights(self, batch_numb, loss_numb):
        training_data = self.training_seq.__getitem__(batch_numb)
        self.compression_pipeline.compress_and_decompress_and_verify_if_testing(
            self.model.get_weights(), training_data, loss_numb)
