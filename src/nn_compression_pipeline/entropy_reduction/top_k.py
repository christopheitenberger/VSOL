from nn_compression_pipeline import RunParamKeys
from src.nn_compression_pipeline import CompressWithMeasurementsWithLayerInformation
import numpy as np
import math


class TopK(CompressWithMeasurementsWithLayerInformation):
    """
    Tried optimizations:
    * Removing small values of the selected top k values had mostly a negative effect while having almost no memory effect
    * Setting maximum percentage of total layer to not use all numbers due to min_per_layer_of_max_layer has almost no effect
    * disconnect min_per_layer_of_max_layer and min_loss_adaptive => worse performance
    * using weights of prev run + last run diff, no real difference
    """
    lossy = True
    trigger_callback_if_lossy = False

    # testing = True

    def __init__(self, percentage, min_per_layer_of_max_layer=0.0001, loss_adaptive_percentage_linear_or_convex=False):
        super().__init__()
        self.percentage = percentage
        self.min_per_layer_of_max_layer = min_per_layer_of_max_layer
        self.loss_adaptive_percentage_linear_or_convex = \
            loss_adaptive_percentage_linear_or_convex

        self.debug_print_if_testing = True

        type_of_adapt_perc = type(self.loss_adaptive_percentage_linear_or_convex)
        self.loss_adaptive_percentage_is_convex = type_of_adapt_perc in [int, float]
        if self.loss_adaptive_percentage_is_convex:
            bend = math.fabs(self.loss_adaptive_percentage_linear_or_convex)
            # from solve for x: (1/(y(0+x)))-x = 1
            self.move_curve_to_x_y_1 = (1 / 2) * ((1 / math.sqrt(bend / (bend + 4))) - 1)

        self.debug_print = self.debug_print_if_testing and self.testing
        self.number_per_layer = None
        self.min_per_layer_number = None

    def convex_f(self, x):
        bend = math.fabs(self.loss_adaptive_percentage_linear_or_convex)
        return (1 / (bend * (x + self.move_curve_to_x_y_1))) - self.move_curve_to_x_y_1

    def calculate_and_set_values_used_per_layer(self):
        self.min_per_layer_number = self.calc_min_per_layer_number()

        selected_size_and_total_size = list(
            map(lambda layer_shapes: self.get_number_used_values_for_layer(self.percentage, layer_shapes),
                self.model_shapes))
        self.number_per_layer, _ = list(zip(*selected_size_and_total_size))

        if self.debug_print:
            p_list = list(map(lambda x: f'{x[0] / x[1]:.3f} ({x[0]}/{x[1]}', selected_size_and_total_size))
            print(f'TopK, numbers per layer {self.percentage} '
                  f'with min of {self.min_per_layer_of_max_layer}(abs:{self.min_per_layer_number}), '
                  f'values per layer:{p_list}')

    def calc_min_per_layer_number(self):
        max_layer_size = max(map(lambda layer_shapes: math.prod(layer_shapes), self.model_shapes))
        return math.floor(max_layer_size * self.min_per_layer_of_max_layer)

    def get_number_used_values_for_layer(self, percentage, layer_shapes, number_used_instead_of_layer_shapes=None):
        size_layer = math.prod(layer_shapes)
        value_percentage_to_use = number_used_instead_of_layer_shapes if number_used_instead_of_layer_shapes is not None else size_layer
        v = math.ceil(percentage * value_percentage_to_use)
        v = max(v, self.min_per_layer_number)
        v = min(size_layer, v)
        return v, size_layer

    def reset_other(self):
        if self.shape_changed_in_this_run:
            self.calculate_and_set_values_used_per_layer()

    def compress(self, value_in, run_params):
        percentage_through_loss = self.get_used_percentage_though_loss_from_param_if_present_and_set(run_params)
        value_out = list(map(lambda x: self.compress_layer(*x, percentage_through_loss),
                             zip(value_in, self.last_lossy_layer_saved.weights_from_last_run_compress,
                                 self.number_per_layer)))
        return self.compress_operations_on_in_and_out_values(value_in, value_out, run_params)

    def get_used_percentage_though_loss_from_param_if_present_and_set(self, run_params):
        if self.loss_adaptive_percentage_linear_or_convex and RunParamKeys.LOSS in run_params:
            return self.get_used_percentage_through_loss(run_params)
        else:
            return None

    def compress_layer(self, layer_in, layer_last_run, number_of_top_values_to_select, percentage_through_loss=None):
        diff_between_layers = layer_last_run - layer_in

        if self.debug_print:
            self.print_stats_of_np(diff_between_layers)

        if percentage_through_loss:
            number_of_top_values_to_select, _ = self.get_number_used_values_for_layer(percentage_through_loss,
                                                                                      diff_between_layers.shape)
        top_k_indexes = self.get_indicex_of_top_k_values(diff_between_layers, number_of_top_values_to_select)

        return self.select_values_from_layer_in_for_top_k_indexes(layer_last_run, layer_in, top_k_indexes)

    def get_used_percentage_through_loss(self, run_params):
        loss = run_params[RunParamKeys.LOSS]
        loss_normalized = self.normalize_loss_for_given_bounds(loss)
        perc_to_use_from_range = self.get_percentage_to_use_for_loss_from_selected_function(loss_normalized)
        percentage_through_loss = \
            ((self.percentage - self.min_per_layer_of_max_layer) * perc_to_use_from_range) \
            + self.min_per_layer_of_max_layer
        if self.debug_print:
            print(f'loss: {loss}, l_norm: {loss_normalized}, '
                  f'perc_to_use_from_range: {perc_to_use_from_range}, final percentage:{percentage_through_loss}')
        return percentage_through_loss

    def get_percentage_to_use_for_loss_from_selected_function(self, loss_normalized):
        if self.loss_adaptive_percentage_is_convex:
            if self.loss_adaptive_percentage_linear_or_convex < 0:
                return 1 - self.convex_f(loss_normalized)
            else:
                return self.convex_f(1 - loss_normalized)
        else:
            return loss_normalized

    def normalize_loss_for_given_bounds(self, loss):
        loss_lower_limit = 0
        loss_upper_limit = 0.5
        diff_upper_lower_bound = loss_upper_limit - loss_lower_limit
        loss_cutoff_by_limits = max(loss, loss_lower_limit)
        loss_cutoff_by_limits = min(loss_cutoff_by_limits, loss_upper_limit)
        loss_minus_lower_limit = loss_cutoff_by_limits - loss_lower_limit
        loss_normalized = loss_minus_lower_limit * (1 / diff_upper_lower_bound)
        assert 0 <= loss_normalized <= 1
        return loss_normalized

    def select_values_from_layer_in_for_top_k_indexes(self, layer_last_run, layer_in, top_k_selected_indexes):
        layer_out = layer_last_run.copy()
        layer_out[top_k_selected_indexes] = layer_in[top_k_selected_indexes]
        return layer_out

    def get_indicex_of_top_k_values(self, layer, k):
        # source for upcoming line: https://github.com/epfml/sparsifiedSGD/blob/master/memory.py#L43
        top_k_indexes = np.argpartition(np.abs(layer.ravel()), -k)[-k:]
        assert len(top_k_indexes) == k
        return np.unravel_index(top_k_indexes, layer.shape)

    def print_stats_of_np(self, weights):
        total = math.prod(weights.shape)
        numb_negative = (np.sum(weights > 0))
        numb_null = np.sum(weights == 0)
        numb_rest = total - numb_negative - numb_null
        print(
            f'pos: {numb_rest / total * 100:.2f}%/{numb_rest}/{total}, '
            f'null: {numb_null / total * 100:.2f}%/{numb_null}/{total}, '
            f'neg: {numb_negative / total * 100:.2f}%/{numb_negative}/{total}, '
        )

    def algorithm_name(self, dic_params=None):
        # old names used to use stable saves
        params_for_loss_based_if_used = {}
        if self.loss_adaptive_percentage_linear_or_convex:
            params_for_loss_based_if_used = {
                'laploc': self.loss_adaptive_percentage_linear_or_convex,
            }

        regular_values = {
            'perc': self.percentage,
        }
        if self.min_per_layer_of_max_layer > 0:
            regular_values['mpl'] = self.min_per_layer_of_max_layer
        return super().algorithm_name(regular_values | params_for_loss_based_if_used)
