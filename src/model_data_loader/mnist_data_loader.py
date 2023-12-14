import numpy as np
import pandas as pd

from model_data_loader import ModelDataLoader


class MNISTDataLoader(ModelDataLoader):

    # data source: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    # unpack archive into ./data/mnist
    def __init__(self, path_to_data):
        super().__init__(path_to_data)

    def load_and_pre_process_data(self, path_to_data):
        # [original source, modified](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)
        X_train, Y_train = self.__pre_process_to_x_y(pd.read_csv(f'{path_to_data}/mnist_train.csv'))
        X_test, Y_test = self.__pre_process_to_x_y(pd.read_csv(f'{path_to_data}/mnist_test.csv'))

        return X_train, Y_train, \
            X_test, Y_test

    def __pre_process_to_x_y(self, pd_loaded):
        # [original source, modified](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)
        Y = pd_loaded["label"]

        # Drop 'label' column
        X = pd_loaded.drop(labels=["label"], axis=1)

        number_of_empty_values_of_x = np.count_nonzero(~X.isnull())
        assert number_of_empty_values_of_x != 0, number_of_empty_values_of_x

        # free some space
        del pd_loaded

        # Note:
        # We perform a grayscale normalization to reduce the effect of illumination's differences.
        # Moreover the CNN converg faster on [0..1] data than on [0..255].

        # Normalize the data
        X = X / 255.0

        # Note:
        # Train and test images (28px x 28px) has been stock into pandas.Dataframe as 1D vectors of 784 values. We reshape all data to 28x28x1 3D matrices.
        # Keras requires an extra dimension in the end which correspond to channels. MNIST images are gray scaled so it use only one channel. For RGB images, there is 3 channels, we would have reshaped 784px vectors to 28x28x3 3D matrices.

        # Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
        X = X.values.reshape(-1, 28, 28, 1)

        return X, Y
