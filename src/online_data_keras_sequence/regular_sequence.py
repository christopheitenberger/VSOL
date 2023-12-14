import math
import tensorflow as tf


class RegularSequence(tf.keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size, number_of_batches_override=None, debug_print=False):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.debug_print = debug_print
        assert len(self.x) == len(self.y)
        assert (len(self.x) % self.batch_size) == 0

        self.number_of_batches_override = None
        if number_of_batches_override:
            if number_of_batches_override > self.__len__():
                raise Exception(
                    f'not enough data for overriden batches number, '
                    f'override:{number_of_batches_override} calc: {self.__len__()}')
            else:
                self.number_of_batches_override = number_of_batches_override

    def __len__(self):
        if self.number_of_batches_override:
            return self.number_of_batches_override

        return math.floor(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        assert idx < len(self.x), f'idx: {idx}, len(x): {len(self.x)}'
        start = idx * self.batch_size
        end = (idx + 1) * self.batch_size
        batch_x = self.x[start:end]
        batch_y = self.y[start:end]
        if self.debug_print:
            print(f'index {idx:04} [{start}:{end}]')

        return batch_x, batch_y

    def reset_sequence(self):
        pass
