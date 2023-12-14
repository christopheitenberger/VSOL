class ModelLabelAndIterationSelection:
    def __init__(self, models, labels=None, models_to_use=None, batches=None):
        self.models = models
        self.labels = None
        self.models_to_use = None
        self.batches = None

        self.validate_and_set_params(labels, models_to_use, batches)

    def get_models(self, labels=None, models_to_use=None, batches=None):
        self.validate_and_set_params(labels, models_to_use, batches)

        if self.models_to_use is None:
            models_to_return = [*self.models]
        else:
            models_to_return = self.models[0:self.models_to_use]

        for model_to_change in models_to_return:
            model_to_change.data_split.number_of_labels_to_use = self.labels

        if self.batches is None:
            for model_to_change in models_to_return:
                model_to_change.data_split.reset_number_online_batches()
        else:
            for model_to_change in models_to_return:
                model_to_change.data_split.set_number_online_batches(20)

        return models_to_return

    def validate_and_set_params(self, labels=None, models_to_use=None, batches=None):
        self.assert_is_in_range_or_none(labels, 4)
        self.labels = labels

        self.assert_is_in_range_or_none(batches, 1)
        self.batches = batches

        self.assert_is_in_range_or_none(models_to_use, len(self.models))
        self.models_to_use = models_to_use

    def assert_is_in_range_or_none(self, value_to_check, upper_limit):
        assert value_to_check in list(range(1, upper_limit + 1)) or value_to_check is None
