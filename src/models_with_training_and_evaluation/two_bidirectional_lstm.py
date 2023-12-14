from keras.models import Sequential
from keras import layers

import model_data_loader
from src.data_splitter import DataSplitOnOff
from src.model_data_loader import AGNewsDataLoader
from src.models_with_training_and_evaluation.model_wrapper_and_loader import ModelWrapperAndLoader


class TwoBidirectionalLSTM(ModelWrapperAndLoader):
    def __init__(self, data_split: DataSplitOnOff, root_offline_model_folder, epochs, verbose_level=0):
        super().__init__(data_split, root_offline_model_folder, epochs, verbose_level)

    def model_compile(self):
        # source, modified: https://www.kaggle.com/code/ishandutta/ag-news-classification-lstm
        data_loader_from_split = self.data_split.model_data_loader
        assert isinstance(data_loader_from_split,
                          (model_data_loader.ag_new_data_loader.AGNewsDataLoader, AGNewsDataLoader))
        vocab_size = data_loader_from_split.vocab_size
        input_length = data_loader_from_split.sentence_length
        model = Sequential([
            layers.Embedding(vocab_size, 128, input_length=input_length),
            layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.GlobalMaxPooling1D(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.25),
            layers.Dense(256, activation='relu'),
            layers.Dense(4, activation='softmax'),
        ])

        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=["accuracy"])

        return model
