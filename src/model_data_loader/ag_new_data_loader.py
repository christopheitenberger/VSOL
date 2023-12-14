from model_data_loader import ModelDataLoader
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

import pandas as pd
import numpy as np
import os


class AGNewsDataLoader(ModelDataLoader):

    def __init__(self, vocab_size, sentence_length, path_to_data):
        self.vocab_size = vocab_size
        self.sentence_length = sentence_length
        super().__init__(path_to_data)

    def load_and_pre_process_data(self, path_to_data):
        # source, modified: https://www.kaggle.com/code/ishandutta/ag-news-classification-lstm
        assert os.path.exists(f'{path_to_data}/train.csv'), \
            f'download and unzip dataset from https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset into {path_to_data}'
        data = pd.read_csv(f'{path_to_data}/train.csv')
        testdata = pd.read_csv(f'{path_to_data}/test.csv')

        def extract_train_data(data):
            x_train = data['Title'] + " " + data['Description']
            y_train = data['Class Index'].apply(lambda x: x - 1).values
            return x_train, y_train

        x_train, y_train = extract_train_data(data)
        x_test, y_test = extract_train_data(testdata)

        tokenizer = Tokenizer(num_words=self.vocab_size)
        tokenizer.fit_on_texts(np.append(x_train.values, x_test.values))

        word_index = tokenizer.word_index
        total_words = len(word_index) + 1
        maxlen = 100

        def tokenize(data, tokenizer, maxlen):
            tokenized = tokenizer.texts_to_sequences(data)
            return pad_sequences(tokenized, maxlen)

        X_train_final = tokenize(x_train, tokenizer, self.sentence_length)
        X_test_final = tokenize(x_test, tokenizer, self.sentence_length)

        return X_train_final, y_train, X_test_final, y_test
