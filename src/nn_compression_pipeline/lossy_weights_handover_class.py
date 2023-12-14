class LossyWeightsHandoverClass:
    def __init__(self):
        self.subscribers_when_weights_change = []

        self.lossy_weights = None
        self.weights_from_last_run_compress = []
        self.weights_from_prev_run_compress = []
        self.weights_from_last_run_decompress = []

        self.is_weights_from_original_run_saved_and_used = True

    def set_lossy_weights_and_call_subscribers(self, new_lossy_weights, run_params):
        if len(self.subscribers_when_weights_change) == 0:
            self.weights_from_last_run_compress = new_lossy_weights
            return None

        self.lossy_weights = new_lossy_weights
        for subscriber in self.subscribers_when_weights_change:
            subscriber(run_params)
        self.lossy_weights = None

    def override_compressed_weights(self, weights):
        self.weights_from_last_run_compress = weights

    def set_new_compress_weights(self, new_weights):
        self.weights_from_prev_run_compress = self.weights_from_last_run_compress
        self.weights_from_last_run_compress = new_weights

    def set_last_decompress_weights(self, new_last_weights):
        self.weights_from_last_run_decompress = new_last_weights

    def reset_to_previous_weights(self):
        self.weights_from_last_run_compress = self.weights_from_prev_run_compress
        self.weights_from_prev_run_compress = None
        self.lossy_weights = None

    def add_subscriber_to_weight_changes(self, subscriber_class_func):
        self.subscribers_when_weights_change.append(subscriber_class_func)

    def reset_weights(self, weights):
        self.weights_from_last_run_compress = weights
        self.weights_from_last_run_decompress = weights
        self.weights_from_prev_run_compress = None
        self.lossy_weights = None
