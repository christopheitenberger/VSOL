
import numpy as np
import pandas as pd

from .constants import *


def print_and_save_if_file_present_multi_column(df_to_format, table_folder, file_name=None, styler=None):
    latex_and_string_print = create_multi_column_table_latex_and_string(df_to_format, styler)
    return write_to_file(file_name, table_folder, latex_and_string_print)


def create_multi_column_table_latex_and_string(df_to_format, styler=None):
    if styler is None:
        styler = df_to_format.style

    is_grouped_comparison_table = isinstance(df_to_format.columns, pd.MultiIndex)

    hide_names = True
    if hide_names:
        styler = styler.hide(axis=0, names=True)

    styler = styler.format_index(escape="latex", axis=1).format_index(escape="latex", axis=0)

    # styler.applymap_index(lambda v: "rotatebox:{45}--rwrap--latex;", level=0, axis=1) # rotate colum text

    if is_grouped_comparison_table:
        styler = style_set_comparison_table_columns_bold(styler, df_to_format)
        wrap_column_names_to_multicolumn_for_to_latex(df_to_format)
    else:
        styler = style_set_values_bold_by_function(styler, np.nanmin)

    show_index = True
    if not show_index:
        styler = styler.hide(axis=0)

    styler = styler.format(precision=3)
    latex_formatted = styler.to_latex(hrules=True, convert_css=True,
                                      multicol_align='p{1.7cm}')
    print_formatted = df_to_format.to_string()

    return [latex_formatted, print_formatted, styler]


def wrap_column_names_to_multicolumn_for_to_latex(df_to_format):
    wrap_multicolumn_warp = lambda x: f'\multicolumn{{1}}{{p{{2.5cm}}}}{{{x}}}' \
        if x is not None and 'multicolumn' not in x else x
    df_to_format.columns.names = list(map(wrap_multicolumn_warp, df_to_format.columns.names))


def style_set_comparison_table_columns_bold(styler, df_to_format):
    columns = df_to_format.columns.get_level_values(0)

    mark_max = [COMPRESSION_RATIO_COL, MEAN_ACCURACY_DIFF_COL]
    mark_max_in_df = [mark_max_col_name for mark_max_col_name in mark_max if mark_max_col_name in columns]
    styler = style_set_values_bold_by_function(styler, np.max, axis=0, subset=mark_max_in_df)

    mark_min = [COMPRESSION_TIME_COL, COMPRESSION_TIME_MAX_COL, DECOMPRESSION_TIME_COL, DECOMPRESSION_TIME_MAX_COL]
    mark_min_in_df = [mark_min_col_name for mark_min_col_name in mark_min if mark_min_col_name in columns]
    styler = style_set_values_bold_by_function(styler, np.min, axis=0, subset=mark_min_in_df)

    return styler


def style_set_values_bold_by_function(styler, selection_func, axis=None, subset=None):
    def highlight_value_by_func(s, props=''):
        if isinstance(s, pd.Series):
            s_v = [value for key, value in s.items() if BASELINE_PREFIX not in key]
        else:
            s_v = s
        sel_value = selection_func(s_v)
        return np.where(s == sel_value, props, '')

    styler = styler.apply(highlight_value_by_func, props='font-weight: bold;', axis=axis, subset=subset)
    return styler


def write_to_file(file_name, table_folder, latex_and_string_print):
    latex_formatted = latex_and_string_print[0]
    formatted_df = latex_and_string_print[2]
    is_file_name_given = file_name is not None
    is_full_path_given = table_folder is not None and is_file_name_given

    print('\n')
    if is_file_name_given:
        print(f'---- # {file_name}\n')

    if is_full_path_given:
        path_for_table_to_save = table_folder + file_name + '.tex'
        with open(path_for_table_to_save, 'w') as file:
            file.write(latex_formatted)

    return formatted_df
