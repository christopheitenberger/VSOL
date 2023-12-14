import numpy as np
import tensorflow as tf
from statistics import mean

from src.nn_compression_pipeline import RunParamKeys
from src.nn_compression_pipeline.compress_with_measurements_and_saves_when_skip_saves import \
    CompressWithMeasurementsAndSavesWhenSkipSaves


class SkipSaveWhenAccuracyStillGood(CompressWithMeasurementsAndSavesWhenSkipSaves):
    lossy = True
    weights_can_repeat = True

    # testing = True
    def __init__(self, percentage_different_leads_to_save=0.001, evaluate_only_last_batch=False):
        super().__init__(True)
        self.percentage_different_leads_to_save = percentage_different_leads_to_save
        self.model_clone_for_evaluation = None
        self.evaluate_only_last_batch = evaluate_only_last_batch

        self.weights_from_last_save = None
        self.accuracy_sum_of_last_save_for_new_data = 0
        self.x_test_save_for_new_data = None
        self.y_test_save_for_new_data = None
        self.run_without_save = 0
        self.batch_size = None

        self.save_skipped_counter = []

    def compress(self, value_in, run_params):
        if RunParamKeys.TRAINING_DATA not in run_params:
            raise IndexError(f'{RunParamKeys.TRAINING_DATA} info missing in run_params')

        if self.run_without_save >= 1:
            if self.are_deployed_weights_still_acceptable(run_params):
                self.save_skipped_counter.append(self.run_without_save)
                run_params[RunParamKeys.PIPELINE_FINISHED] = True
            else:
                self.run_without_save = 0

        if self.run_without_save == 0:
            self.save_skipped_counter.append(0)
            assert len(self.last_lossy_layer_saved.weights_from_last_run_compress) >= 1
            self.reset_saved_weight_related_values()
            self.weights_from_last_save = self.last_lossy_layer_saved.weights_from_last_run_compress

        self.run_without_save += 1
        return self.compress_operations_on_in_and_out_values(value_in, value_in, run_params)

    def are_deployed_weights_still_acceptable(self, run_params):
        x_new, y_new = run_params[RunParamKeys.TRAINING_DATA]

        if self.batch_size is None:
            self.batch_size = len(x_new)

        if self.evaluate_only_last_batch:
            x_for_new_weights = x_new
            y_for_new_weights = y_new
        else:
            if self.x_test_save_for_new_data is None:
                self.x_test_save_for_new_data = x_new
                self.y_test_save_for_new_data = y_new
            else:
                self.x_test_save_for_new_data = np.append(x_new, self.x_test_save_for_new_data, axis=0)
                self.y_test_save_for_new_data = np.append(y_new, self.y_test_save_for_new_data, axis=0)

            x_for_new_weights = self.x_test_save_for_new_data
            y_for_new_weights = self.y_test_save_for_new_data

        accuracy_last_save = self.set_weights_and_get_prediction_accuracy(self.weights_from_last_save, x_new, y_new)

        if self.evaluate_only_last_batch:
            accuracy_old_weights = accuracy_last_save
        else:
            self.accuracy_sum_of_last_save_for_new_data += accuracy_last_save
            accuracy_old_weights = self.accuracy_sum_of_last_save_for_new_data / self.run_without_save

        accuracy = self.set_weights_and_get_prediction_accuracy(
            self.last_lossy_layer_saved.weights_from_last_run_compress, x_for_new_weights, y_for_new_weights)

        numb_of_data_to_eval = len(x_for_new_weights)
        percentage_different_leads_to_save_rounded_for_number_of_samples = round(
            self.percentage_different_leads_to_save * numb_of_data_to_eval) / numb_of_data_to_eval
        skip_save_model_due_to_accuracy_acceptable = \
            (accuracy_old_weights + percentage_different_leads_to_save_rounded_for_number_of_samples) >= accuracy
        if self.testing:
            print(f'use_last_model: {skip_save_model_due_to_accuracy_acceptable}'
                  f'{self.run_without_save:4.0f} deployed_a: {accuracy_old_weights:.4f}, '
                  f'best_a: {accuracy:.4f} ')

        return skip_save_model_due_to_accuracy_acceptable

    def set_weights_and_get_prediction_accuracy(self, weights, x, y_true):
        self.model_clone_for_evaluation.set_weights(weights)
        y_pred = self.model_clone_for_evaluation.predict(x, batch_size=self.batch_size, verbose=0)
        accuracy = mean(tf.keras.metrics.categorical_accuracy(y_true, y_pred).numpy())
        return accuracy

    def reset_saved_weight_related_values(self):
        self.weights_from_last_save = None
        self.accuracy_sum_of_last_save_for_new_data = 0
        self.x_test_save_for_new_data = None
        self.y_test_save_for_new_data = None
        self.run_without_save = 0

    def reset_other(self):
        super().reset_other()

        if len(self.save_skipped_counter) > 0:
            number_skipped_runs = np.count_nonzero(self.save_skipped_counter)
            print(f'Number of skipped runs {number_skipped_runs}/{len(self.save_skipped_counter)}\n'
                  f'for iterations: {self.save_skipped_counter}')
        self.save_skipped_counter = []
        self.batch_size = None

        self.reset_saved_weight_related_values()

    def reset_weights(self, model):
        super().reset_weights(model)
        self.model_clone_for_evaluation = tf.keras.models.clone_model(model)

    def reset_measurements_and_other_after_run(self):
        self.model_clone_for_evaluation = None

    def algorithm_name(self, dic_params=None):
        return super().algorithm_name({
            'pds': self.percentage_different_leads_to_save,
            'elb': self.evaluate_only_last_batch,
        })
