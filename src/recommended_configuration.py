from compression_pipeline_test_runner import PerformanceRequirementSettingsData, LossSetting, CompressionSpeed, \
    AccuracyLimitPercent

recommended_setting = PerformanceRequirementSettingsData(
    # should be set by user as high as acceptable to increase compression ratio
    max_decompression_time_in_s=15,
    lossy=LossSetting.LOSSY,
    compression_speed=CompressionSpeed.FAST,
    accuracy_limit_percent=AccuracyLimitPercent.ZERO
)
