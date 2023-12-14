import pickle


class ProcessingCache:
    """
    ProcessingCache class is responsible for saving and retrieving values based on a given key.
    If they key is not present, the given function is executed and the resulting value is saved for the next execution.
    It also provides the functionality to save and load the cache from a file.
    """

    def __init__(self, values=None, file_folder=None, file_name='cache_map', debug_print=False):
        self.file_folder = file_folder
        self.file_name = file_name
        self.rerun = False
        self.debug_print = debug_print
        if values is not None:
            self.values = values
        elif self.__get_file_path():
            self.load_from_file()
        else:
            self.values = {}

    def get(self, key, value_retriever):
        if hasattr(value_retriever, '__name__'):
            key = f'{key}_{value_retriever.__name__}'
        if key not in self.values or self.rerun:
            if self.debug_print:
                print(f'{key} has to be retrieved')
            self.values[key] = value_retriever(key)
            self.save_to_file()
            if self.debug_print:
                print(f'saved after executing {key} ')

        return self.values[key]

    def __get_file_path(self):
        if self.file_folder and self.file_name:
            return self.file_folder + self.file_name
        else:
            return None

    def save_to_file(self):
        with open(self.__get_file_path(), 'wb') as file:
            pickle.dump(self.values, file)

    def load_from_file(self):
        with open(self.__get_file_path(), 'rb') as file:
            self.values = pickle.load(file)
