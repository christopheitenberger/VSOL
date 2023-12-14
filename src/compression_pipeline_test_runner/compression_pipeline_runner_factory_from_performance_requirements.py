from keras import Model

import np_utils
from compression_pipeline_test_runner import CompressionPipelineRunner
from compression_pipeline_test_runner.performance_requirement_settings_data import PerformanceRequirementSettingsData, \
    CompressionSpeed, AccuracyLimitPercent
from nn_compression_pipeline import DifResetSaver, Compress_With_Measurements, ZSTDWithMeasurement, PickleDump, \
    Combine, PickleToFile, SplitFloatAndStackByByteSegments, FloatRemoveSections, TopK, GCXS, \
    SkipSaveWhenAccuracyStillGood


class CompressionPipelineRunnerFactoryFromPerformanceRequirements:
    """
    This class provides a factory method to create and
    configure a CompressionPipelineRunner based on performance requirements settings.
    The resulting possible settings origin from evaluating them
    for the neural networks TwoBidirectionalLSTM and TwoByTwoConvLayeredNN

    """

    @staticmethod
    def create_and_select_algorithms_from_settings_and_configure(
            performance_requirement_settings: PerformanceRequirementSettingsData,
            model: Model, path_to_overall_model_folder: str) -> CompressionPipelineRunner:

        compression_runner_combination = CompressionPipelineRunnerFactoryFromPerformanceRequirements.__get_compression_runner_combination_from_settings(
            performance_requirement_settings)
        compression_pipeline = CompressionPipelineRunner(
            'PRODUCTION_RUN_WITHOUT_SETTINGS',
            path_to_overall_model_folder,
            compression_runner_combination)
        compression_pipeline.reset(model)
        return compression_pipeline

    @staticmethod
    def __get_compression_runner_combination_from_settings(
            performance_requirement_settings: PerformanceRequirementSettingsData) -> [Compress_With_Measurements]:
        to_byte_and_general_compression_and_file_saver_and_dif_reset = [
            PickleDump(),
            ZSTDWithMeasurement(1),
            PickleToFile(),
            DifResetSaver(
                max_decompression_time_in_s=performance_requirement_settings.max_decompression_time_in_s)
        ]

        if performance_requirement_settings.lossy == performance_requirement_settings.lossy.LOSSLESS:
            return [
                Combine(np_utils.float_byte_wise_xor),
                SplitFloatAndStackByByteSegments(),
                *to_byte_and_general_compression_and_file_saver_and_dif_reset,
            ]

        # Lossy

        float_removal_lower_compression_settings = (18, 32)
        float_removal_high_compression_settings = (20, 32)

        top_k_lower_compression_settings = (0.0750, 0.001, True)
        top_k_high_compression_settings = (0.0750, 0.001, -0.5)

        skip_save_lower_compression_settings = (0, False)
        skip_save_high_compression_settings = (0.01, False)

        gcxs_settings_all_true = (True, True, True, True)

        if performance_requirement_settings.compression_speed == CompressionSpeed.FAST:

            if performance_requirement_settings.accuracy_limit_percent == AccuracyLimitPercent.ZERO:
                return [
                    Combine(np_utils.float_byte_wise_xor),
                    FloatRemoveSections(*float_removal_high_compression_settings),
                    SplitFloatAndStackByByteSegments(),
                    *to_byte_and_general_compression_and_file_saver_and_dif_reset,
                ]

            if performance_requirement_settings.accuracy_limit_percent == AccuracyLimitPercent.ZERO_POINT_ONE:
                return [
                    TopK(*top_k_lower_compression_settings),
                    Combine(np_utils.float_byte_wise_xor),
                    FloatRemoveSections(*float_removal_lower_compression_settings),
                    GCXS(*gcxs_settings_all_true),
                    *to_byte_and_general_compression_and_file_saver_and_dif_reset,
                ]

            if performance_requirement_settings.accuracy_limit_percent == AccuracyLimitPercent.ONE:
                return [
                    TopK(*top_k_high_compression_settings),
                    Combine(np_utils.float_byte_wise_xor),
                    FloatRemoveSections(*float_removal_high_compression_settings),
                    GCXS(*gcxs_settings_all_true),
                    *to_byte_and_general_compression_and_file_saver_and_dif_reset,
                ]
        elif performance_requirement_settings.compression_speed == CompressionSpeed.SLOW:
            if performance_requirement_settings.accuracy_limit_percent == AccuracyLimitPercent.ZERO:
                return [
                    Combine(np_utils.float_byte_wise_xor),
                    FloatRemoveSections(*float_removal_high_compression_settings),
                    SkipSaveWhenAccuracyStillGood(*skip_save_lower_compression_settings),
                    SplitFloatAndStackByByteSegments(),
                    *to_byte_and_general_compression_and_file_saver_and_dif_reset,
                ]

            if performance_requirement_settings.accuracy_limit_percent == AccuracyLimitPercent.ZERO_POINT_ONE:
                return [
                    Combine(np_utils.float_byte_wise_xor),
                    FloatRemoveSections(*float_removal_high_compression_settings),
                    SkipSaveWhenAccuracyStillGood(*skip_save_high_compression_settings),
                    SplitFloatAndStackByByteSegments(),
                    *to_byte_and_general_compression_and_file_saver_and_dif_reset,
                ]

            if performance_requirement_settings.accuracy_limit_percent == AccuracyLimitPercent.ONE:
                return [
                    TopK(*top_k_high_compression_settings),
                    Combine(np_utils.float_byte_wise_xor),
                    FloatRemoveSections(*float_removal_high_compression_settings),
                    SkipSaveWhenAccuracyStillGood(*skip_save_high_compression_settings),
                    GCXS(*gcxs_settings_all_true),
                    *to_byte_and_general_compression_and_file_saver_and_dif_reset,
                ]

        raise ValueError(f'PerformanceRequirementSettingsData is miss configured')
