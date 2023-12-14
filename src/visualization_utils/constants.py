# final column names
MEAN_ACCURACY_DIFF_COL = 'Mean Diff to Lossless Last Accuracy'
DECOMPRESSION_TIME_MAX_COL = 'Decompression Time Max'
DECOMPRESSION_TIME_COL = 'Decompression Time (sec.)'
COMPRESSION_TIME_MAX_COL = 'Compression Time Max (sec.)'
COMPRESSION_TIME_COL = 'Compression Time (sec.)'
COMPRESSION_RATIO_COL = 'Compression Ratio (sec.)'

# measurement names

DISK_USAGE_MEASUREMENT = 'disk_usage'
ACC_NEXT_MEAN_MEASUREMENT = 'acc_next-mean'
ACC_LAST_MEAN_MEASUREMENT = 'acc_last-mean'
COMPRESSION_TIME_MEAN_MEASUREMENT = 'compression_time_mean'
DECOMPRESSION_TIME_MEAN_MEASUREMENT = 'decompression_time_mean'
DECOMPRESSION_TIME_MAX_MEASUREMENT = 'decompression_time_max'
COMPRESSION_TIME_MAX_MEASUREMENT = 'compression_time_max'

# column names
NN_TYPE_COL = 'nn_type'
ALGORITHM_NAME_COL = 'algorithm_name'
RUN_NAME_COL = 'run_name'

# NN Names
NN_CONV = 'Conv'
NN_LSTM = 'LSTM'

# processing names
NN_TYPE_TO_REPLACE = {'TwoByTwoConvLayeredNN': NN_CONV, 'TwoBidirectionalLSTM': NN_LSTM,
                      'TwoBidirectionalLTSM': NN_LSTM}
BASELINE_PREFIX = 'BL-'
POTENTIAL_BASELINE_ALG_NAMES_KEYS_TO_REPLACE = {
    'PickleToFile': BASELINE_PREFIX + 'NoCompression',
    'PickleDump-ZSTDWithMeasurement#cpr_1-PickleToFile': BASELINE_PREFIX + 'GeneralCompr',
    'Combine#alg_float_byte_wise_xor-SplitFloatAndStackByByteSegments-PickleDump-ZSTDWithMeasurement#cpr_1-PickleToFile-DifResetSaver#fni_28': BASELINE_PREFIX + 'LosslessCompr',
}

# table grouping regex
TABLE_GROUP_REG_SAME_K = r'.*[kl]([\d\.]+)p.*'

# algorithm shortened names
GCXS_ALL_SETTINGS_TRUE = 'GcxxsTsTcTsTd'
SPLIT_FLOAT_AND_STACK_BY_BYTE_SEGMENTS = 'Splbs'
