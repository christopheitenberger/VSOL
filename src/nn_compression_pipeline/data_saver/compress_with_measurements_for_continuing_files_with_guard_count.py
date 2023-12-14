from src.nn_compression_pipeline import RunParamKeys, Compress_With_Measurements


class CompressWithMeasurementsForContinuingFilesWithGuardCount(Compress_With_Measurements):
    file_saver = True

    def __init__(self):
        super().__init__()
        self.path_to_file = None
        self.counter = 0
        self.max_counter = None

    def decompress_and_measure(self, value_in, run_params):
        if self.max_counter and self.counter >= self.max_counter:
            return None
        value_out = super().decompress_and_measure(value_in, run_params)
        self.counter += 1
        return value_out

    def reset_other(self):
        self.max_counter = self.counter
        self.counter = 0

    def compress(self, value_in: bytes, run_params) -> None:
        with open(self.get_file_path(run_params), 'wb') as f:
            self.write_to_file(value_in, f)

    def write_to_file(self, value_in: bytes, file):
        pass

    def decompress(self, value_in=None, run_params=None) -> bytes:
        with open(self.get_file_path(run_params), 'rb') as f:
            value_out = self.read_from_file(f)

        return value_out

    def read_from_file(self, file):
        pass

    def get_file_path(self, run_params):
        if RunParamKeys.FILE_NAME_MIDDLE_OVERRIDE in run_params:
            file_name_middle = run_params[RunParamKeys.FILE_NAME_MIDDLE_OVERRIDE]
        else:
            if RunParamKeys.RUN_NUMBER_TO_USE in run_params:
                counter_to_use = run_params[RunParamKeys.RUN_NUMBER_TO_USE]
            else:
                counter_to_use = self.counter
            file_name_middle = f'{counter_to_use:05}'

        return f'{self.path_to_file}/model.{file_name_middle}.dump'
