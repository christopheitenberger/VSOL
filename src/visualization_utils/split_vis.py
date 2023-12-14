import numbers
from itertools import product
from typing import List, NamedTuple

import numpy as np
import pandas as pd
import seaborn as sns

from np_utils import ProcessingCache
from .constants import *

from models_with_training_and_evaluation import ModelWrapperAndLoader, OnlineNextAndLastAccuracies

MEAN = 'mean'
LABEL_NUMBER = 'label_number'
PARAMETER_NAME = 'pname'
Online_Data_Split_Value_Name = 'value'
NEURAL_NETWORK_TYPE_COL = 'NN Type'
BATCH_NUMBER = 'Batch Number'
ACCURACY_VALUE_NORMALIZED_COLUMN_NAME = 'Last Accuracy Normalized'


class LastAccuracyOfOnlineRunForSplitParameter(NamedTuple):
    online_last_accuracy_for_run: list[float]
    name_of_changed_split_parameter: str
    value_of_parameter: str
    run_name: str


def get_run_data_for_different_params_for_all_labels(param_name, values, models: List[ModelWrapperAndLoader],
                                                     save_values: ProcessingCache
                                                     ) -> list[LastAccuracyOfOnlineRunForSplitParameter]:
    accuracies_of_online_run_for_split_parameter_list = []
    for model in models:
        for label in model.data_split.get_all_class_labels():
            model.data_split.set_selected_group(label)

            for v in values:
                accuracies_of_online_run_for_split_parameter_list.append(
                    get_accuracies_of_online_run_from_cache_or_run_online_run(model, param_name, v, save_values))

    return accuracies_of_online_run_for_split_parameter_list


def process_table_to_overview_per_param(
        accuracy_of_online_run_for_parameter_list: list[LastAccuracyOfOnlineRunForSplitParameter], name_of_value):
    mean_accuracy_table = create_mean_table_from_rest(accuracy_of_online_run_for_parameter_list)

    curve_diff_mean, run_values = create_diff_cross_by_absolute_diff_of_curve(accuracy_of_online_run_for_parameter_list,
                                                                              mean_accuracy_table)

    df = pd.DataFrame({
        name_of_value: run_values,
        'Mean Curve Diff': curve_diff_mean,
    })

    df = df.set_index(name_of_value)
    df.index = df.index.astype(str)

    df.columns.name = df.index.name
    df.index.name = None

    return df


def show_line_graph_of_table(accuracy_of_online_run_for_parameter_list: [LastAccuracyOfOnlineRunForSplitParameter],
                             selected_value=None, label_number=None):
    if isinstance(label_number, numbers.Number):
        label_number = str(label_number)

    mean_accuracy_table = create_full_table(accuracy_of_online_run_for_parameter_list)
    parameter_name = mean_accuracy_table['pname'].iloc[1]

    if selected_value is None:
        selected_value = mean_accuracy_table[Online_Data_Split_Value_Name].unique()

    if label_number is not None:
        mean_accuracy_table = mean_accuracy_table[
            (mean_accuracy_table[Online_Data_Split_Value_Name].isin(selected_value)) & (
                        mean_accuracy_table[LABEL_NUMBER] == label_number)]
        g = sns.FacetGrid(mean_accuracy_table, col=Online_Data_Split_Value_Name, hue=NEURAL_NETWORK_TYPE_COL,
                          sharex=False)
    else:
        mean_accuracy_table = mean_accuracy_table[
            (mean_accuracy_table[Online_Data_Split_Value_Name].isin(selected_value))]
        g = sns.FacetGrid(mean_accuracy_table, col=Online_Data_Split_Value_Name, hue=NEURAL_NETWORK_TYPE_COL,
                          sharex=False,
                          row=LABEL_NUMBER)

    g.map(sns.lineplot, BATCH_NUMBER, ACCURACY_VALUE_NORMALIZED_COLUMN_NAME)
    g.set_titles(col_template=f"{parameter_name}: {{col_name}}", row_template="{row_name}")
    g.add_legend()


def create_cross_table_from_run(
        accuracy_of_online_run_for_parameter_list: list[LastAccuracyOfOnlineRunForSplitParameter]):
    mean_accuracy_table = create_mean_table_from_rest(accuracy_of_online_run_for_parameter_list)
    return create_cross_table(accuracy_of_online_run_for_parameter_list, mean_accuracy_table)


def index_and_column_names_to_string_percent_from_fraction(table):
    table.columns = (table.columns * 100).astype(str) + '%'
    table.index = (table.index * 100).astype(str) + '%'
    return table


def create_cross_table(accuracy_of_online_run_for_parameter_list, tt_f):
    grouped = tt_f.groupby([NEURAL_NETWORK_TYPE_COL, Online_Data_Split_Value_Name], sort=False).mean(
        numeric_only=True).reset_index()
    nn_types = '_' + tt_f[NEURAL_NETWORK_TYPE_COL].unique()

    merge = grouped.merge(grouped, how='cross', suffixes=nn_types)
    merge = merge[
        (merge[f'{NEURAL_NETWORK_TYPE_COL}_{NN_CONV}'] == NN_CONV) & (
                merge[f'{NEURAL_NETWORK_TYPE_COL}_{NN_LSTM}'] == NN_LSTM)]

    def get_curve_diff_mean(x):
        nn_type_conv, value_conv, _, nn_type_lstm, value_lstm, _ = x
        nn_type_and_run_value_pairs = [(nn_type_conv, value_conv), (nn_type_lstm, value_lstm)]
        return create_indices_for_cross_list_and_diff_each_curve_mean(
            nn_type_and_run_value_pairs, accuracy_of_online_run_for_parameter_list, tt_f)

    merge['diff'] = merge.apply(get_curve_diff_mean, axis=1)

    to_drop = NEURAL_NETWORK_TYPE_COL + nn_types
    merge = merge.drop(to_drop, axis=1)
    to_drop = MEAN + nn_types
    merge = merge.drop(to_drop, axis=1)

    merge = merge.pivot(index=f'value_{NN_CONV}', columns=f'value_{NN_LSTM}', values='diff')

    merge.columns.name = f'{NN_CONV}\\textbackslash {NN_LSTM}'

    return merge


# private methods

def get_accuracies_of_online_run_from_cache_or_run_online_run(
        model_to_check: ModelWrapperAndLoader, p_name: str,
        value: str, save_values: ProcessingCache) -> LastAccuracyOfOnlineRunForSplitParameter:
    model_to_check.data_split.override_split_settings(**{p_name: value})
    run_name = model_to_check.data_split.get_name_for_online_training_run()

    def get_online_accs(run_name) -> OnlineNextAndLastAccuracies:
        return model_to_check.get_online_next_and_last_accs()

    online_accuracies: OnlineNextAndLastAccuracies = save_values.get(run_name, get_online_accs)
    return LastAccuracyOfOnlineRunForSplitParameter(online_accuracies['last_accs'], p_name, value, run_name)


def normalize_measurements(x):
    return normalize_measurements_with_bounds(x)[0]


def normalize_measurements_with_bounds(x: OnlineNextAndLastAccuracies):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    min_of_x, max_of_x = np.min(x), np.max(x)
    normalized_array = (x - min_of_x) / (max_of_x - min_of_x)
    return normalized_array, min_of_x, max_of_x


def create_mean_table_from_rest(tt: list[LastAccuracyOfOnlineRunForSplitParameter]):
    mean_accuracy_table = create_full_table(tt)
    mean_accuracy_table = mean_accuracy_table.groupby(
        [PARAMETER_NAME, Online_Data_Split_Value_Name, RUN_NAME_COL, NEURAL_NETWORK_TYPE_COL, LABEL_NUMBER],
        sort=False).agg(
        {ACCURACY_VALUE_NORMALIZED_COLUMN_NAME: MEAN}).reset_index()
    mean_accuracy_table = mean_accuracy_table.rename({ACCURACY_VALUE_NORMALIZED_COLUMN_NAME: MEAN}, axis=1)
    return mean_accuracy_table


def create_full_table(accuracy_of_online_run_for_parameter_list: list[LastAccuracyOfOnlineRunForSplitParameter]):
    table_average = [vi for v in map(get_avg, accuracy_of_online_run_for_parameter_list) for vi in v]
    df_table_average = pd.DataFrame(table_average,
                                    columns=[BATCH_NUMBER, ACCURACY_VALUE_NORMALIZED_COLUMN_NAME, PARAMETER_NAME,
                                             Online_Data_Split_Value_Name,
                                             RUN_NAME_COL])
    df_table_average[[NEURAL_NETWORK_TYPE_COL, RUN_NAME_COL]] = df_table_average[RUN_NAME_COL].str.split('#', n=1,
                                                                                                         expand=True)
    df_table_average[[LABEL_NUMBER, RUN_NAME_COL]] = df_table_average[RUN_NAME_COL].str.split('#', n=1, expand=True)

    df_table_average[NEURAL_NETWORK_TYPE_COL] = df_table_average[NEURAL_NETWORK_TYPE_COL].map(NN_TYPE_TO_REPLACE)
    return df_table_average


def get_avg(accuracy_of_online_run_for_parameter: LastAccuracyOfOnlineRunForSplitParameter):
    next_accs_zero, p_name, value, run_name = accuracy_of_online_run_for_parameter
    next_accs_zero_normalized, _, _ = normalize_measurements_with_bounds(next_accs_zero)

    return [[count, v_n, p_name, value, run_name] for count, v_n in enumerate(next_accs_zero_normalized)]


def create_diff_cross_by_absolute_diff_of_curve(accuracy_of_online_run_for_parameter_list, tt_f):
    run_values = tt_f[Online_Data_Split_Value_Name].unique()
    nn_types = tt_f[NEURAL_NETWORK_TYPE_COL].unique()
    diff_cross = []

    for run_value in run_values:
        nn_type_and_run_value_pairs = list(product(nn_types, [run_value]))
        diff_cross.append(create_indices_for_cross_list_and_diff_each_curve_mean(
            nn_type_and_run_value_pairs, accuracy_of_online_run_for_parameter_list, tt_f))

    return diff_cross, run_values


def create_indices_for_cross_list_and_diff_each_curve_mean(nn_type_and_run_value_pairs,
                                                           accuracy_of_online_run_for_parameter_list, tt_f):
    return create_indices_for_cross_list_and_diff_each_curve(nn_type_and_run_value_pairs,
                                                             accuracy_of_online_run_for_parameter_list, tt_f).mean()


def create_indices_for_cross_list_and_diff_each_curve(nn_type_and_run_value_pairs,
                                                      accuracy_of_online_run_for_parameter_list, tt_f):
    indices_to_compare = create_indices_cross_list_for_nn_type_and_run_value_pairs(nn_type_and_run_value_pairs, tt_f)
    return get_array_of_all_diffs(indices_to_compare, accuracy_of_online_run_for_parameter_list)


def create_indices_cross_list_for_nn_type_and_run_value_pairs(nn_type_and_run_value_pairs, tt_f):
    indices = []
    for nn_type, run_value in nn_type_and_run_value_pairs:
        v = tt_f.index[
            (tt_f[Online_Data_Split_Value_Name] == run_value) & (tt_f[NEURAL_NETWORK_TYPE_COL] == nn_type)].tolist()
        indices.append(v)
    indices_to_compare = list(product(*indices))
    return indices_to_compare


def get_array_of_all_diffs(indices_to_compare, accuracy_of_online_run_for_parameter_list):
    array_of_diffs = []
    for first_index, second_index in indices_to_compare:
        diff_abs_mean = get_values_to_compare_and_diff_abs(first_index, second_index,
                                                           accuracy_of_online_run_for_parameter_list)
        array_of_diffs.append(diff_abs_mean)
    return np.array(array_of_diffs)


def get_values_to_compare_and_diff_abs(first_index, second_index, accuracy_of_online_run_for_parameter_list):
    first = accuracy_of_online_run_for_parameter_list[first_index][0]
    second = accuracy_of_online_run_for_parameter_list[second_index][0]
    diff = normalize_measurements(first) - normalize_measurements(second)
    diff_abs_mean = np.abs(diff)
    return diff_abs_mean
