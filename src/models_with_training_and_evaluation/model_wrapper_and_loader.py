import math
import os
from typing import TypedDict

import tensorflow as tf
from keras.callbacks import EarlyStopping

from src.data_splitter import DataSplitOnOff


class OnlineNextAndLastAccuracies(TypedDict):
    next_accs: list[float]
    last_accs: list[float]


class ModelWrapperAndLoader:
    def __init__(self, data_split: DataSplitOnOff, root_offline_model_folder, epochs, verbose_level=0):
        self.root_offline_model_folder = root_offline_model_folder
        self.epochs = epochs
        self.data_split = data_split
        self.verbose_level = verbose_level

        self.model: tf.keras.Model = None
        self.weights_after_initialization = None

    def model_compile_if_not_saved(self):
        if self.model is None or self.weights_after_initialization is None:
            self.data_split.set_seed()
            self.model = self.model_compile()
            self.weights_after_initialization = self.model.get_weights()

    def model_compile(self) -> tf.keras.Model:
        pass

    def accuracy_for_model(self):
        self.load_weights_from_folder_or_train_if_missing()
        X_train_off, Y_train_off, X_val, Y_val = self.data_split.get_data_for_offline_training()
        X_on_last_batch, Y_on_last_batch = self.data_split.get_data_online_validation_last_batch()
        X_test, Y_test = self.data_split.get_test_data_split_like_offline_split()
        return {
            'train': self.__get_eval_measurement(X_train_off, Y_train_off),
            'val': self.__get_eval_measurement(X_val, Y_val),
            'last_batch': self.__get_eval_measurement(X_on_last_batch, Y_on_last_batch),
            'test': self.__get_eval_measurement(X_test, Y_test)
        }

    def __get_eval_measurement(self, x, y, value_for_seed_additional_to_seed=None):
        self.data_split.set_random_seed_combined(value_for_seed_additional_to_seed)
        batch_size, _ = self.data_split.get_batch_size_and_number_of_online_batches()
        history = self.model.evaluate(x, y, batch_size=self.data_split.batch_size, verbose=self.verbose_level)
        return history[1]


    def get_validation_accuracy_for_full_and_offline_only_model_with_all_data(self):
        self.load_weights_from_folder_or_train_if_missing(True, [
            EarlyStopping(monitor="val_accuracy", restore_best_weights=True, patience=5)
        ])
        _, _, X_val, Y_val = self.data_split.get_data_for_full_and_only_offline_training()
        assert Y_val is not None, f'Test data is missing for model {self.data_split.get_offline_only_model_name()}'
        return self.__get_eval_measurement(X_val, Y_val, 0)

    def get_online_next_and_last_accs(self) -> OnlineNextAndLastAccuracies:
        self.load_weights_from_folder_or_train_if_missing()
        batch_size, number_of_online_batches = self.data_split.get_batch_size_and_number_of_online_batches()

        gen = self.data_split.get_online_validation_batch_as_epoch_sequence()
        x, y = self.data_split.get_data_online_validation_last_batch()
        val_data_on = (x, y)
        history = self.model.fit(gen, batch_size=batch_size, epochs=number_of_online_batches,
                                 validation_data=val_data_on,
                                 verbose=self.verbose_level, shuffle=False)

        return {
            'next_accs': history.history['accuracy'],
            'last_accs': history.history['val_accuracy'],
        }

    def get_total_params(self, load_full_and_only_offline_model=False):
        self.load_weights_from_folder_or_train_if_missing(load_full_and_only_offline_model)
        shapes = [layer.shape for layer in self.model.get_weights()]
        params_per_layer = [math.prod(shape) for shape in shapes]
        prod = int(math.fsum(params_per_layer))
        return prod, params_per_layer, shapes, self.get_model_summary_as_string()

    def get_model_summary_as_string(self):
        summary_per_line = []
        self.model.summary(print_fn=lambda x: summary_per_line.append(x))
        return '\n'.join(summary_per_line)

    def load_weights_from_folder_or_train_if_missing(self, load_full_and_only_offline_model=False, callbacks=None):
        if not load_full_and_only_offline_model:
            assert self.data_split.selected_group is not None
        self.model_compile_if_not_saved()

        if load_full_and_only_offline_model:
            model_name = self.data_split.get_offline_only_model_name()
        else:
            model_name = self.data_split.get_name_for_model()

        offline_model_folder_data_run = f'{self.root_offline_model_folder}/{model_name}/'
        path_to_existing_folder = offline_model_folder_data_run
        if not os.path.exists(path_to_existing_folder):
            os.mkdir(path_to_existing_folder)

        if os.path.exists(f'{path_to_existing_folder}.index'):
            self.model.load_weights(path_to_existing_folder)
        else:
            self.model.set_weights(self.weights_after_initialization)
            batch_size, _ = self.data_split.get_batch_size_and_number_of_online_batches()
            if load_full_and_only_offline_model:
                data_to_use_for_training = self.data_split.get_data_for_full_and_only_offline_training()
                self.data_split.set_random_seed_combined(0)
            else:
                data_to_use_for_training = self.data_split.get_data_for_offline_training()
                self.data_split.set_random_seed_combined()

            X_train_off, Y_train_off, X_val, Y_val = data_to_use_for_training
            history = self.model.fit(X_train_off, Y_train_off,
                                     batch_size=batch_size,
                                     epochs=self.epochs,
                                     validation_data=(X_val, Y_val), callbacks=callbacks, verbose=self.verbose_level)
            self.model.save_weights(path_to_existing_folder)

        return self.model
