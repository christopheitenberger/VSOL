import sparse as sp
import numpy as np

from nn_compression_pipeline import RunParamKeys
from np_utils import diff_flat_with_remaining_start, split_and_reshape, get_split_upper_limit_for_shapes
from src.nn_compression_pipeline import SplitFloatAndStack, CompressWithMeasurementsWithLayerInformation


class GCXS(CompressWithMeasurementsWithLayerInformation):
    """
    gcxs better on sparse while float split better on save
    """

    # testing = True

    def __init__(self, split_weight_floats=False, concatenate_all_layers=False, split_indices_ints=False,
                 diff_indices=False):
        super().__init__()
        self.split_weight_floats = split_weight_floats
        self.concatenate_all_layers = concatenate_all_layers
        self.split_indices_ints = split_indices_ints
        self.diff_indices = diff_indices

        self.increasing_total_shape_prod_for_split = None
        float_split_required = self.split_weight_floats or self.split_indices_ints
        self.fs = SplitFloatAndStack() if float_split_required else None

    def is_full_diff(self, run_params):
        return RunParamKeys.FILE_NAME_MIDDLE_OVERRIDE in run_params \
            and 'full_' in run_params[RunParamKeys.FILE_NAME_MIDDLE_OVERRIDE]

    def apply_to_each(self, values, func_to_apply):
        return [func_to_apply(v) for v in values]

    def compress(self, value_in: [np.ndarray], run_params):
        if self.is_full_diff(run_params):
            return self.compress_full_diff(value_in)

        if self.concatenate_all_layers:
            return self.compress_layer(self.flatten_and_concatenate(value_in))

        return self.apply_to_each(value_in, self.compress_layer)

    def compress_full_diff(self, value_in):
        if not self.split_weight_floats:
            return value_in

        if self.concatenate_all_layers:
            f = self.flatten_and_concatenate(value_in)

            return self.fs.compress_layer(f, f.shape)
        else:
            return self.fs.compress(value_in, None)

    def compress_layer(self, last):
        gcxs_obj = sp.GCXS(last)

        if self.split_weight_floats:
            gcxs_obj.data = self.fs.compress_layer(gcxs_obj.data, gcxs_obj.data.shape)

        self.compress_indices_if_set(gcxs_obj)
        return gcxs_obj

    def compress_indices_if_set(self, gcxs_obj):
        if not self.split_indices_ints and not self.diff_indices:
            return gcxs_obj

        indices = gcxs_obj.indices
        if self.diff_indices:
            indices = diff_flat_with_remaining_start(indices)

        if self.split_indices_ints:
            indices = self.fs.compress_layer(indices, indices.shape, 8)

        gcxs_obj.indices = indices
        return gcxs_obj

    def decompress(self, value_in, run_params):
        if self.is_full_diff(run_params):
            return self.decompress_full_diff(value_in)

        if self.concatenate_all_layers:
            return self.split_and_reshape(self.decompress_layer(value_in))

        return self.apply_to_each(value_in, self.decompress_layer)

    def decompress_full_diff(self, value_in):
        if not self.split_weight_floats:
            return value_in

        if self.concatenate_all_layers:
            de = self.fs.decompress_layer(value_in, (value_in.shape[1],))
            return self.split_and_reshape(de)
        else:
            return self.fs.decompress(value_in, None)

    def decompress_layer(self, gcxs_obj):
        if self.split_weight_floats:
            gcxs_obj.data = self.fs.decompress_layer(gcxs_obj.data, (gcxs_obj.data.shape[1],))

        self.decompress_indices_if_set(gcxs_obj)
        return gcxs_obj.todense()

    def decompress_indices_if_set(self, gcxs_obj):
        if not self.split_indices_ints and not self.diff_indices:
            return gcxs_obj

        indices = gcxs_obj.indices
        if self.split_indices_ints:
            indices = self.fs.decompress_layer(indices, (indices.shape[1],), 8, np.int64)

        if self.diff_indices:
            indices = np.cumsum(indices)

        gcxs_obj.indices = indices
        return gcxs_obj

    def algorithm_name(self, dic_params=None):
        return super().algorithm_name({
            'swf': self.split_weight_floats,
            'cal': self.concatenate_all_layers,
            'sii': self.split_indices_ints,
            'di': self.diff_indices,
        })

    def flatten_and_concatenate(self, value_in):
        return np.concatenate([np.ravel(v) for v in value_in])

    def split_and_reshape(self, layers_concatenated):
        return split_and_reshape(layers_concatenated, self.increasing_total_shape_prod_for_split, self.model_shapes,
                                 self.testing)

    def reset_other(self):
        super().reset_other()

        if self.fs:
            self.fs.model_shapes = self.model_shapes
        if self.concatenate_all_layers:
            self.increasing_total_shape_prod_for_split = get_split_upper_limit_for_shapes(self.model_shapes)
