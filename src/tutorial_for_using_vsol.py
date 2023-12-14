import time

from compression_pipeline_test_runner import PerformanceRequirementSettingsData, LossSetting, CompressionSpeed, \
    AccuracyLimitPercent, OnlineLearningVersionSaver
from conv_model_loader_for_tutorial import \
    get_mnist_model_with_data_and_settings_for_online_run_example

path_to_overall_model_folder = '../notebooks/model_save_runs'
model, data_seq_online_run, batch_size, number_of_online_batches = (
    get_mnist_model_with_data_and_settings_for_online_run_example())

# choosing the default defensive settings to get started quickly
settings_default = PerformanceRequirementSettingsData()

# choosing a specific value for each type of setting
settings_for = PerformanceRequirementSettingsData(
    max_decompression_time_in_s=15,
    lossy=LossSetting.LOSSY,
    compression_speed=CompressionSpeed.FAST,
    accuracy_limit_percent=AccuracyLimitPercent.ZERO_POINT_ONE)

# creating online learning saver from chosen settings
saver = OnlineLearningVersionSaver(
    model,
    data_seq_online_run,
    path_to_overall_model_folder,
    settings_for)

# to simulate a classification timestamp, a timestamp during training is needed
# this is done by creating a timestamp before training and adding 5 seconds
timestamp_before_run = time.time_ns()

# using the online learning saver is as easy as adding
# it as a callback function with the learning function
# the 'saver' internally recorded the deployment timestamp
model.fit(data_seq_online_run,
          batch_size=batch_size,
          epochs=number_of_online_batches,
          verbose=1,
          shuffle=False,
          callbacks=[saver]
          )

# adding 5 seconds to the timestamp of the beginning of the training results
# in a timestamp during training, simulating a classification timestamp
# the deployment timestamp is recorded by the 'saver
simulated_classification_timestamp = timestamp_before_run + (5 * pow(10, 9))

# the weights can be loaded simply by using a timestamp
loaded_weights = saver.load_weights_at_timestamp(simulated_classification_timestamp)

# the model can be also cloned and loaded
# with the weights from the timestamp
model = saver.clone_model_and_load_weights_at_timestamp(
    simulated_classification_timestamp, model)
