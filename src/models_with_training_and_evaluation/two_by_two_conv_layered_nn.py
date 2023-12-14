from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import rmsprop_v2 as RMSprop

from src.data_splitter import DataSplitOnOff
from src.models_with_training_and_evaluation.model_wrapper_and_loader import ModelWrapperAndLoader


class TwoByTwoConvLayeredNN(ModelWrapperAndLoader):
    def __init__(self, data_split: DataSplitOnOff, root_offline_model_folder, epochs, verbose_level=0):
        super().__init__(data_split, root_offline_model_folder, epochs, verbose_level)

    def model_compile(self):
        # [original source, modified](https://www.kaggle.com/code/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)
        # Set the CNN model
        # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation="softmax"))

        # Define the optimizer
        optimizer = RMSprop.RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

        # Compile the model
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

        # learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
        #                                             patience=3,
        #                                             verbose=1,
        #                                             factor=0.5,
        #                                             min_lr=0.00001)

        return model
