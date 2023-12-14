import numpy.typing as npt
from keras import Model
from keras.utils import Sequence
from keras.models import clone_model

from compression_pipeline_test_runner import CompressModel
from compression_pipeline_test_runner.compression_pipeline_runner_factory_from_performance_requirements import \
    CompressionPipelineRunnerFactoryFromPerformanceRequirements
from compression_pipeline_test_runner.performance_requirement_settings_data import PerformanceRequirementSettingsData


class OnlineLearningVersionSaver(CompressModel):
    """
    OnlineLearningVersionSaver

    This class is responsible for saving and loading weights of a model at different timestamps
    during an online learning process.
    It extends the CompressModel class by featuring easy to use methods to load weights for a timestamp


    """

    def __init__(self, model: Model, training_seq: Sequence, path_to_overall_model_folder: str,
                 performance_requirement_settings: PerformanceRequirementSettingsData = PerformanceRequirementSettingsData()):
        compression_pipeline = CompressionPipelineRunnerFactoryFromPerformanceRequirements.create_and_select_algorithms_from_settings_and_configure(
            performance_requirement_settings, model, path_to_overall_model_folder)
        super().__init__(compression_pipeline, training_seq)

    def load_weights_at_timestamp(self, timestamp: int) -> [npt.ArrayLike]:
        return self.compression_pipeline.load_weights_at_timestamp(timestamp)

    def clone_model_and_load_weights_at_timestamp(self, timestamp: int, model: Model):
        weights = self.load_weights_at_timestamp(timestamp)
        clones_model_with_loaded_weights = clone_model(model)
        clones_model_with_loaded_weights.set_weights(weights)
        return clones_model_with_loaded_weights
