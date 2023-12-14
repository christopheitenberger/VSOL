from src.online_data_keras_sequence import RegularSequence


class BatchesAsEpochSequence(RegularSequence):
    """
    A class that represents the sequence of batches as epochs.
    Each batch has to be one epoch to retrieve the accuracy per batch and not per epoch.

    """
    def __init__(self, x_set, y_set, batch_size, debug_print=False):
        super().__init__(x_set, y_set, batch_size, debug_print=debug_print)
        self.fake_epoch_count = 0

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        assert idx == 0
        return super().__getitem__(self.fake_epoch_count)

    def epoch_len(self):
        return super().__len__()

    def on_epoch_end(self):
        self.fake_epoch_count += 1

    def reset_sequence(self):
        self.fake_epoch_count = 0
