import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical  # convert to one-hot-encoding
import tensorflow as tf
import math

from src.model_data_loader import ModelDataLoader
from src.online_data_keras_sequence import BatchesAsEpochSequence, RegularSequence, OneBatchAsEpoch


class DataSplitOnOff:
    """
    Class for splitting data into offline and online training sets.
    UPMonline, TBonline, UPoff and Poff are used to set the amount of used data for offline and online training.

    Attributes
        model_data_loader: An instance of ModelDataLoader that provides the training data.
        number_classes: The number of classes in the data.
        batch_size: The size of each training batch.
        random_seed: The random seed when shuffling the data.
        model_name: The name of the model.
        UPMonline: The maximum number of selected label percentage.
        TBonline: The number of online batches.
        UPoff: The number of instances of selected label to use for offline training.
        Poff: The number of instances of other labels used for offline training.
            Mostly set dynamically.
        number_of_online_batches: The number of batches to use for online training of the TBonline, limiting it.
        number_of_labels_to_use: The number of labels to use for training.
        debug_assert: Whether to enable debug assertions.
        debug_print: Whether to enable debug printing.
    """

    def __init__(self, model_data_loader: ModelDataLoader, number_classes, batch_size, random_seed, model_name,
                 UPMonline, TBonline=None, UPoff=None, Poff=None,
                 number_of_online_batches=None,
                 number_of_labels_to_use=None, debug_assert=True, debug_print=False):
        self.model_data_loader = model_data_loader
        self.x_train_original, self.y_train_as_label_int_original, self.x_test, self.y_test_as_label_int = model_data_loader.get_x_train_and_y_train_as_label_int()
        assert len(self.x_train_original) == len(self.y_train_as_label_int_original)
        # len(np.unique(y_train_as_label_int))
        self.number_classes = number_classes
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.model_name = model_name
        self.UPMonline = UPMonline
        self.TBonline = TBonline
        self.UPoff = UPoff
        self.Poff = Poff
        self.number_of_online_batches = number_of_online_batches
        self.number_of_labels_to_use = number_of_labels_to_use
        self.debug_print = debug_print
        self.debug_assert = debug_assert
        self.is_number_of_online_batches_set = number_of_online_batches is not None

        absolute_value_splits = self.__validate_or_set_max_split_param_and_return_absolute_values_to_use(UPMonline,
                                                                                                         TBonline,
                                                                                                         UPoff, Poff)

        if absolute_value_splits is None:
            raise AttributeError('Value missing')

        self.offline_other_label_number, self.offline_selected_labels_number, \
            self.online_other_labels_number, self.online_selected_labels_number = absolute_value_splits

        self.y_train_original = to_categorical(self.y_train_as_label_int_original, num_classes=number_classes)
        self.y_test = None
        if self.y_test_as_label_int is not None:
            self.y_test = to_categorical(self.y_test_as_label_int, num_classes=number_classes)
        self.x_train = None
        self.y_train_as_label_int = None
        self.y_train = None
        self.selected_group = None
        self.X_train_off = None
        self.Y_train_off = None
        self.X_val_off = None
        self.Y_val_off = None
        self.X_train_on = None
        self.Y_train_on = None

        self.temp_verify = {}

    def __max_number_of_online_batches(self):
        numb_online_data = self.online_selected_labels_number + self.online_other_labels_number
        return math.floor(numb_online_data / self.batch_size)

    def reset_number_online_batches(self):
        self.number_of_online_batches = None
        self.is_number_of_online_batches_set = False

    def set_number_online_batches(self, number_online_batches_to_set):
        self.number_of_online_batches = number_online_batches_to_set
        self.is_number_of_online_batches_set = True

    def __validate_number_of_online_batches_if_present_or_set_to_max(self):
        max_number_of_batches_for_data = self.__max_number_of_online_batches() - 1
        if self.is_number_of_online_batches_set:
            assert self.number_of_online_batches <= max_number_of_batches_for_data
        else:
            self.number_of_online_batches = max_number_of_batches_for_data

        if self.debug_print:
            print(f'{self.number_of_online_batches}/{max_number_of_batches_for_data}(max) online batches used')

    def override_split_settings(self, UPMonline=None, TBonline=None, UPoff=None, Poff=None):
        if UPMonline is None:
            UPMonline = self.UPMonline
        if TBonline is None:
            TBonline = self.TBonline
        if UPoff is None:
            UPoff = self.UPoff
        if Poff is None:
            Poff = self.Poff
        absolute_value_splits = self.__validate_or_set_max_split_param_and_return_absolute_values_to_use(UPMonline,
                                                                                                         TBonline,
                                                                                                         UPoff, Poff)

        if absolute_value_splits is None:
            raise AttributeError('Value missing')

        self.UPMonline, self.TBonline, self.UPoff, self.Poff = (UPMonline, TBonline, UPoff, Poff)
        self.offline_other_label_number, self.offline_selected_labels_number, \
            self.online_other_labels_number, self.online_selected_labels_number = absolute_value_splits
        self.set_selected_group(self.selected_group)

    def get_all_class_labels(self):
        r = range(self.number_classes)
        if self.number_of_labels_to_use:
            end = min(self.number_of_labels_to_use, self.number_classes)
            return r[0: end]
        else:
            return r

    def set_selected_group(self, selected_group):
        self.__validate_number_of_online_batches_if_present_or_set_to_max()
        max_label_number = self.number_classes - 1
        self.selected_group = min(selected_group, max_label_number)
        self.X_train_off, self.Y_train_off, self.X_val_off, self.Y_val_off, self.X_train_on, self.Y_train_on = self.__get_data_sets_for_group()

    def get_batch_size_and_number_of_online_batches(self):
        return self.batch_size, self.number_of_online_batches

    def get_data_for_offline_training(self):
        return self.X_train_off, self.Y_train_off, self.X_val_off, self.Y_val_off

    def get_data_for_full_and_only_offline_training(self):
        return self.x_train_original, self.y_train_original, self.x_test, self.y_test

    def get_online_training_batch_as_epoch_sequence(self):
        end = -self.batch_size
        return BatchesAsEpochSequence(self.X_train_on[:end], self.Y_train_on[:end], self.batch_size, self.debug_print)

    def get_online_validation_batch_as_epoch_sequence(self):
        start = self.batch_size
        return BatchesAsEpochSequence(self.X_train_on[start:], self.Y_train_on[start:], self.batch_size,
                                      self.debug_print)

    def get_online_validation_sequence(self):
        return RegularSequence(self.X_train_on[self.batch_size:], self.Y_train_on[self.batch_size:], self.batch_size,
                               self.number_of_online_batches, self.debug_print)

    def get_data_online_validation_last_batch(self):
        end = self.batch_size * self.__max_number_of_online_batches()
        start = end - self.batch_size
        return self.X_train_on[start:end], self.Y_train_on[start:end]

    def get_online_validation_on_last_batch_sequence(self):
        x, y = self.get_data_online_validation_last_batch()
        return OneBatchAsEpoch(x, y, self.batch_size, self.number_of_online_batches, self.debug_print)

    def set_seed(self):
        tf.keras.utils.set_random_seed(self.random_seed)

    def set_random_seed_combined(self, value_for_seed_additional_to_seed=None):
        if value_for_seed_additional_to_seed is None:
            value_for_seed_additional_to_seed = self.selected_group

        combined_seed = self.random_seed + value_for_seed_additional_to_seed
        tf.keras.utils.set_random_seed(combined_seed)
        return combined_seed

    def __validate_or_set_max_split_param_and_return_absolute_values_to_use(self, UPMonline, TBonline=None,
                                                                            UPoff=None, Poff=None):
        any_value_not_set = np.any(np.array([UPMonline, TBonline, UPoff, Poff]) == None)
        total = len(self.y_train_as_label_int_original)
        unique, label_counts = np.unique(self.y_train_as_label_int_original, return_counts=True)
        selected_label_total = min(label_counts)
        other_label_total = total - selected_label_total

        selected_label_left = selected_label_total
        other_label_left = other_label_total

        selected_label_total_batches = selected_label_total / self.batch_size

        # A = (a*b)/2
        # A*2 = a*b
        # (A*2)/a = b
        maxTBonline = (selected_label_total_batches * 2) / UPMonline
        maxTBonline = math.floor(maxTBonline)

        if TBonline is None:
            print(f'max of TBonline: {maxTBonline}')
            TBonline = maxTBonline
        elif TBonline > maxTBonline:
            print(f'TBonline not possible, maxTBonline: {maxTBonline}, tried TBonline: {TBonline},')
            return
        elif self.debug_print:
            print(f'maxTBonline would have been: {maxTBonline}')

        online_selected_labels_number = (TBonline * UPMonline) / 2 * self.batch_size
        online_selected_labels_number = math.ceil(online_selected_labels_number)
        selected_label_left -= online_selected_labels_number

        online_other_labels_number = (TBonline * self.batch_size) - online_selected_labels_number
        assert online_other_labels_number % 1 == 0
        online_other_labels_number = int(online_other_labels_number)
        other_label_left -= online_other_labels_number

        online_data_number = online_selected_labels_number + online_other_labels_number
        assert online_data_number % self.batch_size == 0, \
            [online_data_number, online_data_number % self.batch_size, online_selected_labels_number, self.batch_size]

        maxUPoff = selected_label_left / total

        if UPoff is None:
            print(f'max of UPoff: {maxUPoff}')
            UPoff = maxUPoff
        elif UPoff > maxUPoff:
            print(f'UPoff not possible, maxUPoff: {maxUPoff}, tried UPoff: {UPoff},')
            return
        elif self.debug_print:
            print(f'maxUPoff would have been: {maxUPoff}')

        offline_selected_labels_number = math.floor(UPoff * total)
        assert offline_selected_labels_number % 1 == 0
        offline_selected_labels_number = int(offline_selected_labels_number)
        selected_label_left -= offline_selected_labels_number

        min_total_other_label = total - max(label_counts)
        min_open_values = (min_total_other_label - online_other_labels_number) + offline_selected_labels_number

        maxPoff = (min_open_values / total)

        if Poff is None:
            print(f'max of Poff: {maxPoff}')
            Poff = maxPoff
        elif Poff > maxPoff:
            print(f'Poff not possible, maxPoff: {maxPoff}, tried Poff: {Poff},')
            return
        elif self.debug_print:
            print(f'maxPoff would have been: {maxPoff}')

        offline_other_label_number = math.floor(Poff * total) - offline_selected_labels_number
        assert offline_other_label_number % 1 == 0
        offline_other_label_number = int(offline_other_label_number)
        other_label_left -= offline_other_label_number

        if self.debug_print:
            if any_value_not_set:
                print(f'UPMonline: {UPMonline:1.3f}, TBonline: {TBonline:1.3f}, UPoff: {UPoff:1.3f}, Poff: {Poff:1.3f}')
            print(f'off: {offline_selected_labels_number}/{offline_other_label_number}(s/o), \
            on: {online_other_labels_number}/{online_selected_labels_number}')
            print(
                f'selected label data unused: {selected_label_left:.0f}, other label data unused: {other_label_left:.0f}')

        return offline_other_label_number, offline_selected_labels_number, \
            online_other_labels_number, online_selected_labels_number

    def __get_data_sets_for_group(self):
        combined_seed = self.set_random_seed_combined()

        idx_for_shuffle = np.arange(self.x_train_original.shape[0])
        np.random.shuffle(idx_for_shuffle)

        self.x_train = self.x_train_original[idx_for_shuffle]
        self.y_train_as_label_int = self.y_train_as_label_int_original[idx_for_shuffle]
        self.y_train = self.y_train_original[idx_for_shuffle]

        X_train_off, Y_train_off, offline_training_data_mask = self.__split_offline()
        X_train_on, Y_train_on, class_with_lower_representation_on, other_classes_training_on = self.__split_online(
            offline_training_data_mask)
        X_val_off, Y_val_off = self.__split_val_offline(class_with_lower_representation_on, other_classes_training_on)

        if self.debug_print:
            selected_class_batch_off = self.__selected_class_per_batch_course(Y_train_off)
            selected_class_batch_on = self.__selected_class_per_batch_course(Y_train_on)
            selected_class_batch_off_then_on = np.append(selected_class_batch_off, selected_class_batch_on)
            plt.plot(selected_class_batch_off_then_on)
            plt.show()

        # verify shuffle stability
        if self.debug_assert and False:
            run_name = self.get_name_for_online_training_run()
            new_training_datas = (X_train_off, Y_train_off, X_train_on, Y_train_on)
            if run_name in self.temp_verify:
                assert len(self.temp_verify[run_name]) == 4
                for last, new in zip(self.temp_verify[combined_seed], new_training_datas):
                    np.testing.assert_array_equal(last, new)

            self.temp_verify[run_name] = new_training_datas

        return X_train_off, Y_train_off, X_val_off, Y_val_off, X_train_on, Y_train_on

    def __get_mask_for_group_split(self, y_as_label_int, group, lower_number, other_number, mask_size=None):
        if mask_size is None:
            mask_size = y_as_label_int.size
        class_with_lower_representation = np.where(y_as_label_int == group)[0]
        other_classes_training = np.where(y_as_label_int != group)[0]

        data_mask_split = np.zeros(mask_size, dtype=bool)

        data_mask_split[class_with_lower_representation[:lower_number]] = True
        data_mask_split[other_classes_training[:other_number]] = True

        return data_mask_split

    def __split_offline(self):
        ## split training and online training set with lower representation of one group in training, virtual drift
        # unique_groups, groups = np.unique(Y_train_labels, return_inverse=True)
        offline_training_data_mask = self.__get_mask_for_group_split(self.y_train_as_label_int, self.selected_group,
                                                                     self.offline_selected_labels_number,
                                                                     self.offline_other_label_number)

        X_train_off, Y_train_off = self.apply_mask(self.x_train, self.y_train, offline_training_data_mask)

        # shuffle data to prevent pure batches
        idx_for_shuffle = np.random.permutation(len(X_train_off))
        X_train_off, Y_train_off = self.apply_mask(X_train_off, Y_train_off, idx_for_shuffle)

        return X_train_off, Y_train_off, offline_training_data_mask

    def __split_online(self, offline_training_data_mask):
        index_to_for_on_training = np.logical_not(offline_training_data_mask)
        number_of_on_training = self.online_selected_labels_number + self.online_other_labels_number
        class_with_lower_representation_on = \
            np.where(index_to_for_on_training & (self.y_train_as_label_int == self.selected_group))[0]
        other_classes_training_on = \
            np.where(index_to_for_on_training & (self.y_train_as_label_int != self.selected_group))[0]

        class_lower_counter = 0
        other_class_counter = 0
        mapping_array_counter = 0
        map_to_rising_array = np.empty(number_of_on_training, dtype=np.intc)

        lower_class_count_per_batch = self.__linear(number_of_on_training, self.online_selected_labels_number)
        for lower_class_count_for_current_batch in lower_class_count_per_batch:
            map_to_rising_array[
            mapping_array_counter:mapping_array_counter + lower_class_count_for_current_batch] = class_with_lower_representation_on[
                                                                                                 class_lower_counter: class_lower_counter + lower_class_count_for_current_batch]
            mapping_array_counter += lower_class_count_for_current_batch
            class_lower_counter += lower_class_count_for_current_batch
            other_class_count_for_current_batch = self.batch_size - lower_class_count_for_current_batch

            map_to_rising_array[
            mapping_array_counter:mapping_array_counter + other_class_count_for_current_batch] = other_classes_training_on[
                                                                                                 other_class_counter: other_class_counter + other_class_count_for_current_batch]

            mapping_array_counter += other_class_count_for_current_batch
            other_class_counter += other_class_count_for_current_batch

        map_to_rising_array = np.delete(map_to_rising_array, np.s_[mapping_array_counter:])

        assert number_of_on_training == mapping_array_counter, [number_of_on_training, mapping_array_counter]

        X_train_on, Y_train_on = self.apply_mask(self.x_train, self.y_train, map_to_rising_array)

        return X_train_on, Y_train_on, class_with_lower_representation_on, other_classes_training_on

    def get_test_data_split_like_offline_split(self):
        class_with_lower_representation = np.where(self.y_test_as_label_int == self.selected_group)[0]
        other_classes_training = np.where(self.y_test_as_label_int != self.selected_group)[0]
        mask = self.__split_val_offline_mask(class_with_lower_representation, other_classes_training,
                                             self.y_test_as_label_int.size)
        return self.apply_mask(self.x_test, self.y_test, mask)

    def __split_val_offline(self, class_with_lower_representation_on, other_classes_training_on):
        mask = self.__split_val_offline_mask(class_with_lower_representation_on, other_classes_training_on,
                                             self.y_train_as_label_int.size)
        return self.apply_mask(self.x_train, self.y_train, mask)

    def __split_val_offline_mask(self, class_with_lower_representation_on, other_classes_training_on, size_mask):
        expected_ratio = self.offline_selected_labels_number / self.offline_other_label_number
        ratio_of_online_data = len(class_with_lower_representation_on) / len(other_classes_training_on)

        if ratio_of_online_data < expected_ratio:
            selected_label_number = len(class_with_lower_representation_on)
            other_label_number = int(
                selected_label_number / self.offline_selected_labels_number * self.offline_other_label_number)
        else:
            other_label_number = len(other_classes_training_on)
            selected_label_number = int(
                other_label_number / self.offline_other_label_number * self.offline_selected_labels_number)

        mask_split_like_online = np.zeros(size_mask, dtype=bool)

        mask_split_like_online[other_classes_training_on[:other_label_number]] = True
        mask_split_like_online[class_with_lower_representation_on[:selected_label_number]] = True

        resulting_group_ratio_in_training = selected_label_number / mask_split_like_online.sum()
        expected_group_ratio = self.offline_selected_labels_number / (
                self.offline_selected_labels_number + self.offline_other_label_number)
        assert math.isclose(resulting_group_ratio_in_training, expected_group_ratio, rel_tol=0.1, abs_tol=0.0005), \
            f'{resulting_group_ratio_in_training:2.6f}/{self.UPoff:2.6f}(actual/expected)'

        return mask_split_like_online

    def apply_mask(self, x, y, mask):
        return x[mask], y[mask]

    def __selected_class_per_batch_course(self, array_y):
        t = to_categorical([self.selected_group], num_classes=self.number_classes)[0]
        kk = array_y == t
        kk = np.all(kk, 1)
        t = math.floor(len(kk) / self.batch_size) * self.batch_size
        return np.reshape(kk[:t], (-1, self.batch_size)).sum(axis=1) / self.batch_size

    def __linear(self, number_of_data_for_online_training, number_of_data_of_chosen_class):
        total_batches = number_of_data_for_online_training / self.batch_size  # some other class values may be lost
        assert total_batches % 1 == 0
        total_batches = int(total_batches)
        # y = z*x
        # area = (z/2)*x^2 | differentiate
        # z = (area*2)/x^2 | solver for z
        z = (number_of_data_of_chosen_class * 2) / math.pow(total_batches, 2)
        area = lambda x: (z / 2) * math.pow(x + 1, 2)

        class_lower_counter = 0

        lower_class_count_per_batch = []
        for n in range(total_batches):
            lower_class_count_for_current_batch = math.floor(area(n) - class_lower_counter)

            if (n + 1) == total_batches:
                if (class_lower_counter + lower_class_count_for_current_batch) > number_of_data_of_chosen_class:
                    lower_class_count_for_current_batch = number_of_data_of_chosen_class - class_lower_counter
                else:
                    class_other_left = (number_of_data_for_online_training - number_of_data_of_chosen_class) - (
                            (n * self.batch_size) - class_lower_counter)

                    lower_class_count_for_current_batch = max(self.batch_size - class_other_left,
                                                              lower_class_count_for_current_batch)

            class_lower_counter += lower_class_count_for_current_batch
            lower_class_count_per_batch.append(lower_class_count_for_current_batch)

        return lower_class_count_per_batch

    def get_offline_only_model_name(self):
        return '#'.join([self.model_name, str(self.batch_size), str(self.random_seed)])

    def get_name_for_model(self, ):
        return '#'.join([self.model_name, str(self.selected_group),
                         str(self.UPMonline), str(self.TBonline), str(self.UPoff), str(self.Poff),
                         str(self.batch_size), str(self.random_seed)])

    def get_name_for_online_training_run(self) -> str:
        return '#'.join([self.get_name_for_model(), str(self.number_of_online_batches)])
