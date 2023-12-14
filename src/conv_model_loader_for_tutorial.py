from data_splitter import DataSplitOnOff
from model_data_loader import MNISTDataLoader
from models_with_training_and_evaluation import TwoByTwoConvLayeredNN
from keras import Model
from keras.utils import Sequence


def get_mnist_model_with_data_and_settings_for_online_run_example() -> (Model, Sequence, int, int):
    mnist_data_loader = MNISTDataLoader(path_to_data='../data/mnist')
    offline_model_folder = '../notebooks/model_offline_save'

    # split settings
    random_seed = 2
    upm_online = 0.5
    tb_online = 150
    u_poff_mnist = 0.0001
    number_of_online_batches = None

    data_split_mnist = DataSplitOnOff(mnist_data_loader,
                                      number_classes=10, batch_size=86,
                                      random_seed=random_seed,
                                      UPMonline=upm_online,
                                      TBonline=tb_online,
                                      UPoff=u_poff_mnist,
                                      model_name=TwoByTwoConvLayeredNN.__name__,
                                      number_of_online_batches=number_of_online_batches,
                                      number_of_labels_to_use=4,
                                      )
    model_cnn = TwoByTwoConvLayeredNN(data_split_mnist, offline_model_folder, 10)

    data_split_on_off = model_cnn.data_split
    data_split_on_off.set_selected_group(0)

    model = model_cnn.load_weights_from_folder_or_train_if_missing()
    data_seq_online_run = data_split_on_off.get_online_training_batch_as_epoch_sequence()

    batch_size, number_of_online_batches = model_cnn.data_split.get_batch_size_and_number_of_online_batches()

    return model, data_seq_online_run, batch_size, number_of_online_batches
