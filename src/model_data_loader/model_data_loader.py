from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ModelDataLoader(ABC):
    def __init__(self, path_to_data: str):
        self.x_train, self.y_train_as_label_int, self.x_test, self.y_test_as_label_int = self.load_and_pre_process_data(
            path_to_data)
        assert len(self.x_train) == len(self.y_train_as_label_int)

    @abstractmethod
    def load_and_pre_process_data(self, path_to_data):
        pass

    def print_label_distribution(self):
        unique, counts = np.unique(self.y_train_as_label_int, return_counts=True)
        sns.barplot(x=unique, y=counts).set(title=type(self).__name__)
        plt.show()

    def get_x_train_and_y_train_as_label_int(self):
        return self.x_train, self.y_train_as_label_int, self.x_test, self.y_test_as_label_int
