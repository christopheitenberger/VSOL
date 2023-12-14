import pandas as pd

from nn_compression_pipeline import Compress_With_Measurements
from visualization_utils import get_non_overlapping_parts, ALGORITHM_NAME_COL


class CompressionPerLayerCombinationCreator:

    def __init__(self, compression_runners_for_all_layers_with_connections: [(str, Compress_With_Measurements)],
                 order_of_layers_to_add: [[str]], debug_print=False):
        dependency_order_flat = self.flatten_array(order_of_layers_to_add)
        dependency_order_flat_set = set(dependency_order_flat)
        assert len(dependency_order_flat_set) == len(dependency_order_flat)

        connections, compression_runners = zip(*compression_runners_for_all_layers_with_connections)
        assert len(set(connections).union(dependency_order_flat_set)) == len(set(connections))

        self.compression_runners_for_all_layers = compression_runners_for_all_layers_with_connections
        self.dependency_order = order_of_layers_to_add
        self.debug_print = debug_print

        self.compression_runners = compression_runners
        self.connections = connections

    def flatten_array(self, array_of_arrays):
        return [el for array in array_of_arrays for el in array]

    def get_per_layer_combination_from_runners(self):
        indexes_of_lower_layer_indices = []
        combinations = []
        for n in self.dependency_order:
            indexes_for_current_layer_grouped = [self.get_indexes_for_group_string(s) for s in n]
            indexes_for_current_layer = self.flatten_array(indexes_for_current_layer_grouped)

            for index_of_cur_layer_to_use in indexes_for_current_layer_grouped:
                indexes_unordered = index_of_cur_layer_to_use + indexes_of_lower_layer_indices
                combinations.append(self.get_combination_from_list_of_indexes(indexes_unordered))

            indexes_of_lower_layer_indices = indexes_of_lower_layer_indices + indexes_for_current_layer

            if len(indexes_for_current_layer_grouped) >= 2:
                combinations.append(self.get_combination_from_list_of_indexes(indexes_of_lower_layer_indices))

        return combinations

    def get_combination_from_list_of_indexes(self, indexes_unordered):
        indexes_unordered.sort()
        return [self.compression_runners[i] for i in indexes_unordered]

    def get_indexes_for_group_string(self, group_string: str):
        indexes_of_cur_layers_to_use_for_comb = filter(lambda x: x[1] == group_string, enumerate(self.connections))
        return list(map(lambda x: x[0], indexes_of_cur_layers_to_use_for_comb))

    def get_number_of_combinations_per_layer_ordered_as_combinations(self):
        def get_number(el):
            number_of_elements = len(el)

            if number_of_elements >= 2:
                return number_of_elements + 1
            else:
                return number_of_elements

        return list(map(get_number, self.dependency_order))

    def get_measurements_per_layer_and_total_diff(self, df, dic_value):
        n_combinations_per_layer = self.get_number_of_combinations_per_layer_ordered_as_combinations()
        chosen_measurement_names = list(dic_value.keys())

        df_group_alg = df.groupby([ALGORITHM_NAME_COL], sort=False)
        df_means = df_group_alg.mean(numeric_only=True)
        df_means_only_selected = df_means[chosen_measurement_names]

        assert len(df_means_only_selected.index) == sum(n_combinations_per_layer), \
            (len(df_means_only_selected.index), sum(n_combinations_per_layer))
        assert n_combinations_per_layer[0] == 1
        del n_combinations_per_layer[0]

        first_row = df_means_only_selected.iloc[0]

        row_count = 0

        measurements = []
        measurements_first_row = []
        for o in n_combinations_per_layer:
            row_of_last_layer = df_means_only_selected.iloc[row_count]

            for k in range(o):
                row_count += 1
                last_row = df_means_only_selected.iloc[row_count]

                measurements.append(self.get_all_measurements(last_row, row_of_last_layer, dic_value))
                measurements_first_row.append(self.get_all_measurements(last_row, first_row, dic_value))

        columns = [ALGORITHM_NAME_COL] + chosen_measurement_names
        df_diff_to_last_layer = pd.DataFrame(data=measurements, columns=columns)
        df_diff_to_first = pd.DataFrame(data=measurements_first_row, columns=columns)

        return df_diff_to_last_layer, df_diff_to_first

    def get_all_measurements(self, row_to_compare_to, current_row, dic_value):
        diff_of_name = get_non_overlapping_parts(row_to_compare_to.name, current_row.name)[0]

        def get_measurement_diff_with_name_with_rows(measurement_name, combine_func):
            return self.get_measurement_diff_with_name(row_to_compare_to, current_row, measurement_name, combine_func)

        measurements_of_row = [get_measurement_diff_with_name_with_rows(measurement_name, combine_func)
                               for measurement_name, combine_func in dic_value.items()]
        measurements_of_row_with_name = [diff_of_name] + measurements_of_row
        return measurements_of_row_with_name

    def get_measurement_diff_with_name(self, current_row, row_to_compare_to, measurement_name, combine_func):
        current_layer_measurement = current_row[measurement_name]
        last_layer_measurement = row_to_compare_to[measurement_name]
        diff_from_func = combine_func(last_layer_measurement, current_layer_measurement)
        if self.debug_print:
            print(f'{measurement_name}, current: {current_layer_measurement}, last:  {last_layer_measurement}'
                  f', diff: {diff_from_func}')
        return diff_from_func
