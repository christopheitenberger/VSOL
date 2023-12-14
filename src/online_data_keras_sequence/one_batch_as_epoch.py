from src.online_data_keras_sequence import RegularSequence


class OneBatchAsEpoch(RegularSequence):
    """
    This class represents a custom data generator to reuse a single batch for several epochs.
    Used to measure last batch accuracy.

    """

    def __init__(self, x_set, y_set, batch_size, number_of_batches_override, debug_print=False):
        assert len(x_set) == batch_size, f'batch_size: {batch_size}, len(x): {len(x_set)}'
        super().__init__(x_set, y_set, batch_size, debug_print)
        self.number_of_batches_override = number_of_batches_override

    def __len__(self):
        return self.number_of_batches_override

    def __getitem__(self, idx):
        return super().__getitem__(0)
