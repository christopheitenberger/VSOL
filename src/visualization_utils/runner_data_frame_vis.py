import math

import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.groupby import DataFrameGroupBy
from tabulate import tabulate

from nn_compression_pipeline import TopK
from .constants import *
from .name_formatting import remove_overlapping_string_junks_from_split, get_last_two_upper_case_letters_except_first


def get_grouped_df_without_baselines_and_reset_frequency_as_algorithm_name_sorted(df_vis: pd.DataFrame):
    df_grouped = create_grouped_table_for_different_nn(df_vis)
    df_grouped = df_grouped.drop(get_baseline_indices_by_list(df_grouped[ALGORITHM_NAME_COL]))
    df_grouped[ALGORITHM_NAME_COL] = df_grouped[ALGORITHM_NAME_COL].astype('str').str.extract(r'\D*(\d+)\D*')
    df_grouped = df_grouped.sort_values(by=ALGORITHM_NAME_COL, key=lambda x: x.astype('int'))
    return df_grouped


def scatterplot_for_nn_types_grouped(df_group_alg, x_measurement=COMPRESSION_RATIO_COL,
                                     y_measurement=DECOMPRESSION_TIME_MAX_COL):
    x = sns.scatterplot(data=df_group_alg, x=x_measurement, y=y_measurement, hue=ALGORITHM_NAME_COL, style=NN_TYPE_COL)
    columns_to_use = math.ceil(df_group_alg[ALGORITHM_NAME_COL].nunique() / 15)
    x.legend(ncol=columns_to_use, loc='upper left', bbox_to_anchor=(1, 1))


def lineplot_for_topk_loss_adaptive_curve_settings(curve_settings):
    def curve_from_top_k(topk):
        return [topk.get_percentage_to_use_for_loss_from_selected_function(r / 100) for r in range(0, 101)]

    curve_settings_with_values_for_curve = {c: curve_from_top_k(TopK(0.1, 0.1, c))
                                            for c in curve_settings}
    curve_settings_plot = sns.lineplot(curve_settings_with_values_for_curve, )
    curve_settings_plot.set(xlabel='Percentage of K Range to use', ylabel='Normalized Loss Value')
    return curve_settings_plot


def show_stats(df, measurement=DISK_USAGE_MEASUREMENT, reverse_sorted=False, measurement_type='mean'):
    m = group_by_use_several_measurements(df, [ALGORITHM_NAME_COL])

    m.sort(key=sort_df(measurement, measurement_type), reverse=reverse_sorted)

    for name, t in m:
        print(f'\n---{name}')
        print(tabulate(t, floatfmt=".4f", headers='keys'))


def sort_df(measurement, agg_func_name):
    return lambda x: x[1].loc[[measurement]][agg_func_name][0]


def create_grouped_comparison_table_for_different_nn_with_acc(dfs_from_different_nn, runners, print_acc_std=False):
    return create_grouped_comparison_table_for_different_nn(dfs_from_different_nn, runners, print_acc=True,
                                                            print_acc_std=print_acc_std)


def create_grouped_comparison_table_for_different_nn_with_acc_without_decompression_time(dfs_from_different_nn, runners,
                                                                                         print_acc_std=False):
    return create_grouped_comparison_table_for_different_nn(dfs_from_different_nn, runners, print_acc=True,
                                                            print_acc_std=print_acc_std,
                                                            print_decompression_time_max=False)


def create_grouped_table_for_different_nn(dfs_from_different_nn, runners: list = None, print_acc=False,
                                          print_acc_std=False, print_decompression_time_max=True):
    """
    runners: list[TestRunnerForCompressionPipelinesAndDifferentDataAndModels]
    """
    measurements_to_unroll_name_and_type_pair = [
        [DISK_USAGE_MEASUREMENT, 'mean_max_prop', COMPRESSION_RATIO_COL],
        [COMPRESSION_TIME_MEAN_MEASUREMENT, 'mean', COMPRESSION_TIME_COL],
        [COMPRESSION_TIME_MAX_MEASUREMENT, 'mean', COMPRESSION_TIME_MAX_COL],
        [DECOMPRESSION_TIME_MEAN_MEASUREMENT, 'mean', DECOMPRESSION_TIME_COL],
    ]

    if print_decompression_time_max:
        measurements_to_unroll_name_and_type_pair.append(
            [DECOMPRESSION_TIME_MAX_MEASUREMENT, 'mean', DECOMPRESSION_TIME_MAX_COL])

    if print_acc:
        measurements_to_unroll_name_and_type_pair.insert(1, [ACC_NEXT_MEAN_MEASUREMENT, 'mean_bl_diff',
                                                             MEAN_ACCURACY_DIFF_COL])
        if print_acc_std:
            measurements_to_unroll_name_and_type_pair.insert(1, [ACC_NEXT_MEAN_MEASUREMENT, 'std', 'ACC STD'])

    baseline_alg_names_keys_to_replace = {}

    columns_to_group = [NN_TYPE_COL, ALGORITHM_NAME_COL]
    grouped_and_unrolled_measurements = \
        [group_and_unroll_some_measurements(df, measurements_to_unroll_name_and_type_pair, columns_to_group)
         for df in dfs_from_different_nn]

    index_baseline = -1

    if runners and False:
        training_time_bl_name = 'BL-Training Time'
        grouped_and_unrolled_measurements = grouped_and_unrolled_measurements + [get_training_time_as_df(runners)]
        baseline_alg_names_keys_to_replace[training_time_bl_name] = training_time_bl_name
        index_baseline -= len(runners)

    df_comb = pd.concat(grouped_and_unrolled_measurements, ignore_index=True, sort=False)

    baseline_algorithm_name = df_comb[ALGORITHM_NAME_COL].values[index_baseline]
    if baseline_algorithm_name in POTENTIAL_BASELINE_ALG_NAMES_KEYS_TO_REPLACE:
        baseline_alg_names_keys_to_replace[baseline_algorithm_name] = \
            POTENTIAL_BASELINE_ALG_NAMES_KEYS_TO_REPLACE[baseline_algorithm_name]

    alg_names_without_baseline = np.setdiff1d(df_comb[ALGORITHM_NAME_COL],
                                              list(baseline_alg_names_keys_to_replace.keys()))
    shortened_alg_name_keys_to_replace = shorten_algorithm_names_in_index(alg_names_without_baseline)
    names_to_replace = shortened_alg_name_keys_to_replace | baseline_alg_names_keys_to_replace
    df_comb[ALGORITHM_NAME_COL] = df_comb[ALGORITHM_NAME_COL].replace(names_to_replace)

    if MEAN_ACCURACY_DIFF_COL in df_comb.columns:
        df_comb[MEAN_ACCURACY_DIFF_COL] = (df_comb[MEAN_ACCURACY_DIFF_COL] * 100).round(2) * -1

    return df_comb


def create_grouped_comparison_table_for_different_nn(dfs_from_different_nn, runners: list = None, print_acc=False,
                                                     print_acc_std=False, print_decompression_time_max=True):
    df_comb = create_grouped_table_for_different_nn(dfs_from_different_nn, runners, print_acc, print_acc_std,
                                                    print_decompression_time_max)
    pivot_table = df_comb.pivot_table(columns=NN_TYPE_COL, index=ALGORITHM_NAME_COL, sort=False)

    pivot_table.columns.names = ['Short Algorithm Name Differences', None]

    return pivot_table


def only_keep_table_rows_containing_substrings_and_baseline(df: pd.DataFrame, substrings: [str]):
    substrings_with_bl = substrings + [BASELINE_PREFIX]
    indices_to_remove_since_missing_phrase = [v for v in df.index
                                              if not string_contains_any_of_substrings(v, substrings_with_bl)]
    return df.drop(indices_to_remove_since_missing_phrase)


def only_keep_table_rows_containing_baseline_and_not_substrings(df: pd.DataFrame, substrings: [str]):
    indices_to_remove_since_missing_phrase = [v for v in df.index
                                              if not BASELINE_PREFIX in v
                                              and string_contains_any_of_substrings(v, substrings)]
    return df.drop(indices_to_remove_since_missing_phrase)


def string_contains_any_of_substrings(v, substrings):
    return any(substring in v for substring in substrings)


def sort_and_fill_na(table: pd.DataFrame, value, nn_type, ascending=False):
    table = table.sort_values(by=(value, nn_type), ascending=ascending)
    table = move_baseline_row_to_first_row(table)
    return table.fillna('')


def move_baseline_row_to_first_row(table):
    target_row = get_baseline_indices_by_list(table.index)
    idx = target_row + [i for i in range(len(table)) if i not in target_row]
    table = table.iloc[idx]
    return table


def get_baseline_indices_by_list(list_to_check):
    return [e for e, v in enumerate(list_to_check) if BASELINE_PREFIX in v]


def sort_groups_by_regex_and_fill_na(table, value, nn_type, regex, ascending=False):
    def extract_string_by_regex_and_pass_others(x):
        s_type = x.dtype
        if s_type == object or s_type == str:
            return x.str.extract(regex, expand=False)

        return x

    table = table.sort_values(by=[ALGORITHM_NAME_COL, (value, nn_type)], ascending=[False, ascending],
                              key=extract_string_by_regex_and_pass_others)
    table = move_baseline_row_to_first_row(table)
    return table.fillna('')


def get_training_time_as_df(runners):
    training_time = [[*runner.get_mean_and_max_training_time_in_s(), runner.get_model_name(), 'BL-Training Time']
                     for runner in runners]
    return pd.DataFrame(training_time,
                        columns=[COMPRESSION_TIME_COL, COMPRESSION_TIME_MAX_COL, NN_TYPE_COL, ALGORITHM_NAME_COL])


def shorten_algorithm_names_in_index(alg_names, remove_overlapping_parts=True):
    alg_names_ref_or_shortened = alg_names
    if remove_overlapping_parts:
        alg_names_ref_or_shortened = remove_overlapping_string_junks_from_split(alg_names_ref_or_shortened)

    shortened_alg_names = list(
        map(shorten_algorithm_name_for_table_by_using_only_the_first_letters, alg_names_ref_or_shortened))
    return dict(zip(alg_names, shortened_alg_names))


def remove_shortened_algorithm_from_index_of_pivot_metrics_table(df: pd.DataFrame, names_to_remove_list: [str]):
    for name_to_remove_list in names_to_remove_list:
        df.index = df.index.str.replace(name_to_remove_list, '')

    return df


def group_and_unroll_some_measurements(df, measurements_to_unroll_name_and_type_pair, columns_to_group_by):
    grouped_listed_measurements = group_by_use_several_measurements(df, columns_to_group_by)
    return unroll_group_by_for_some_measurements(grouped_listed_measurements, columns_to_group_by,
                                                 measurements_to_unroll_name_and_type_pair)


def unroll_group_by_for_some_measurements(df_grouped, group_by_columns, measurements_to_unroll_name_and_type_pair):
    data = {}
    for t in [n[2] for n in measurements_to_unroll_name_and_type_pair] + group_by_columns:
        data[t] = []

    for name, df_df in df_grouped:
        for i in range(len(group_by_columns)):
            data[group_by_columns[i]].append(name[i])

        for (measurement, agg_func_name, col_name) in measurements_to_unroll_name_and_type_pair:
            data[col_name].append(df_df.loc[[measurement]][agg_func_name][0])

    return pd.DataFrame(data)


def group_by_use_several_measurements(df: pd.DataFrame, group_by_columns):
    df = df.replace({NN_TYPE_COL: NN_TYPE_TO_REPLACE})

    df_group_alg = df.groupby(group_by_columns, sort=False)
    assert_that_all_group_by_values_have_same_length(df_group_alg)

    mean_mins = df_group_alg.mean(numeric_only=True).min()
    mean_max_es = df_group_alg.mean(numeric_only=True).max()
    bl_mean = df_group_alg.mean(numeric_only=True).iloc[-1]

    def create_table_from_gro(t: (str, DataFrameGroupBy)):
        (name, group_by) = t
        group_means = group_by.mean(numeric_only=True)
        p = {
            'mean_min_prop': group_means / mean_mins,
            'mean_max_prop': mean_max_es / group_means,
            'mean_min_diff': group_means - mean_mins,
            'mean_bl_diff': group_means - bl_mean,
            'mean': group_means,
            'median': group_by.median(numeric_only=True),
            'min': group_by.min(numeric_only=True),
            'max': group_by.max(numeric_only=True),
            'std': group_by.std(numeric_only=True),
        }
        t = pd.DataFrame(p.values()).T
        t.columns = p.keys()
        return name, t

    return list(map(create_table_from_gro, df_group_alg))


def assert_that_all_group_by_values_have_same_length(group_by_object: DataFrameGroupBy):
    length_per_group = [len(v) for v in group_by_object.indices.values()]
    assert len(np.unique(length_per_group)) == 1, \
        f'Group by lengths unequal, see list:\n{list(zip(length_per_group, group_by_object.indices.keys()))}'


def shorten_algorithm_name_for_table_by_using_only_the_first_letters(s):
    def shorten_params(param):
        param_split = param.split('_')
        param_shortened = param_split[0][:1].lower()
        if len(param_split) == 2:
            param_value = param_split[1]
            if param_value == 'True':
                param_value = 'T'
            elif param_value == 'False':
                param_value = 'F'
            elif is_string_float_percentage(param_value):
                param_value = float_percentage_string_to_full_percentage(param_value)

            param_shortened = param_value + param_shortened

        return param_shortened

    def shorten_name(full_name):
        split = full_name.split('#')
        alg_name = split[0]
        params_split = split[1:]

        alg_short = alg_name[:3].capitalize()
        alg_short += get_last_two_upper_case_letters_except_first(alg_name)

        param_name = ''.join([shorten_params(param) for param in params_split])

        return alg_short + param_name

    return ''.join([shorten_name(s_sp) for s_sp in s.split('-')])


def is_string_float_percentage(string_to_check: str):
    if string_to_check[0:2] != '0.':
        return False

    return string_to_check[2:].isdigit()


def float_percentage_string_to_full_percentage(float_percentage_string):
    return str(round(float(float_percentage_string) * 100, 10))
