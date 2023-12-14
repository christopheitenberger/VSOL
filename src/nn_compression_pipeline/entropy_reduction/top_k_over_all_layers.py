import numpy as np
import math

from np_utils import get_split_upper_limit_for_shapes, split_and_reshape
from np_utils.np_operation_wrapper import flatten_and_concatenate
from .top_k import TopK
from .. import RunParamKeys


class TopKOverAllLayers(TopK):
    lossy = True
    trigger_callback_if_lossy = False

    # testing = True

    def __init__(self, percentage, min_per_layer_of_max_layer=0.0001, loss_adaptive_percentage_linear_or_convex=False):
        super().__init__(percentage, min_per_layer_of_max_layer, loss_adaptive_percentage_linear_or_convex)

        self.min_per_layer_used = type(min_per_layer_of_max_layer) == float and min_per_layer_of_max_layer > 0
        self.increasing_total_shape_prod_for_split = None
        self.number_per_layer = None

    def get_number_used_values_for_layer(self, percentage, layer_shapes, number_used_instead_of_layer_shapes=None):
        size_layer = math.prod(layer_shapes)
        value_percentage_to_use = number_used_instead_of_layer_shapes if number_used_instead_of_layer_shapes is not None else size_layer
        v = math.ceil(percentage * value_percentage_to_use)
        v = min(size_layer, v)
        return v, size_layer

    def get_used_percentage_though_loss_from_param_if_present_or_regular_percentage(self, run_params):
        if self.loss_adaptive_percentage_linear_or_convex and RunParamKeys.LOSS in run_params:
            return self.get_used_percentage_through_loss(run_params)
        else:
            return self.percentage

    def compress(self, value_in, run_params):
        percentage_to_use = self.get_used_percentage_though_loss_from_param_if_present_or_regular_percentage(run_params)
        all_layers = flatten_and_concatenate(value_in)
        last_run = flatten_and_concatenate(self.last_lossy_layer_saved.weights_from_last_run_compress)
        size = math.ceil(percentage_to_use * self.total_size)

        if self.min_per_layer_used:
            all_layers_masked_out = self.compress_layer_with_min_per_layer(all_layers, last_run, size)
        else:
            all_layers_masked_out = self.compress_layer(all_layers, last_run, size)
        value_out = self.split_and_reshape(all_layers_masked_out)

        return self.compress_operations_on_in_and_out_values(value_in, value_out, run_params)

    def split_and_reshape(self, layers_concatenated):
        return split_and_reshape(layers_concatenated, self.increasing_total_shape_prod_for_split, self.model_shapes,
                                 self.testing)

    def calculate_and_set_values_used_per_layer(self):
        return None

    def compress_layer_with_min_per_layer(self, layer_in, layer_last_run, number_of_top_values_to_select):
        diff_between_layers = layer_last_run - layer_in

        if self.testing:
            self.print_stats_of_np(diff_between_layers)

        top_k_indexes = self.get_indicex_of_top_k_values(diff_between_layers, number_of_top_values_to_select)[0]

        def get_min_top_k_indexes_of_layer(start, end):
            layer_section = diff_between_layers[start:end]
            min_values_to_select = min(self.min_per_layer_number, len(layer_section))
            return self.get_indicex_of_top_k_values(layer_section, min_values_to_select)[0] + start

        start_end_per_layer = zip(self.increasing_total_shape_prod_for_split,
                                  self.increasing_total_shape_prod_for_split[1:] + [-1])
        top_k_for_min_per_layer = [get_min_top_k_indexes_of_layer(s, e) for s, e in start_end_per_layer]
        top_k_for_min_per_layer = np.concatenate(top_k_for_min_per_layer)

        top_k_all = np.unique(np.concatenate([top_k_for_min_per_layer, top_k_indexes]))

        if self.testing:
            print(f'top_k sel: {len(top_k_indexes)} (+{len(top_k_all) - len(top_k_indexes)}) new from min layer')

        return self.select_values_from_layer_in_for_top_k_indexes(layer_last_run, layer_in, top_k_all)

    def reset_other(self):
        super().reset_other()

        self.increasing_total_shape_prod_for_split = get_split_upper_limit_for_shapes(self.model_shapes)
        if self.min_per_layer_used:
            self.min_per_layer_number = self.calc_min_per_layer_number()
