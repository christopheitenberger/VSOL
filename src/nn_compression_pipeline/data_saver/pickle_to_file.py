import pickle

from src.nn_compression_pipeline.data_saver.compress_with_measurements_for_continuing_files_with_guard_count \
    import CompressWithMeasurementsForContinuingFilesWithGuardCount


class PickleToFile(CompressWithMeasurementsForContinuingFilesWithGuardCount):
    def write_to_file(self, value_in: bytes, file) -> None:
        pickle.dump(value_in, file)

    def read_from_file(self, file):
        return pickle.load(file)
