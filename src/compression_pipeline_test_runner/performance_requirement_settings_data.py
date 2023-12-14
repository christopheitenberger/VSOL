from dataclasses import dataclass
from enum import Enum


class LossSetting(Enum):
    LOSSY = 'LOSSY'
    LOSSLESS = 'LOSSLESS'


class CompressionSpeed(Enum):
    SLOW = 'SLOW'
    FAST = 'FAST'


class AccuracyLimitPercent(Enum):
    ZERO = 'ZERO'
    ZERO_POINT_ONE = 'ZERO_POINT_ONE '
    ONE = 'ONE'


@dataclass
class PerformanceRequirementSettingsData:
    """
    The performance requirement settings for the
    OnlineLearningVersionSaver class.
    To try out the online version saver, no configuration is required since
    all parameters have a default and preserve the trained values.

    Changing the settings leads to a higher compression ratio
    while effecting runtime and accuracy.

    The table below shows which settings possibly lead to which compression ratios.
    The values can differ since they result from testing
    two different neural networks.
    The values result from a very low max_decompression_time_in_s
    and can be significantly higher with a higher value.

    Compression Ratios for settings
    -------------------------------

    lossy=LOSSLESS:
    -------
    1.6-4.2
    -------

    lossy=LOSSY:
    --------------------------------------------------------------
    AccuracyLimitPercent\CompressionSpeed |    FAST   |   SLOW
    --------------------------------------------------------------
    ZERO                                  | 12.5-30.0 | 14.3-46.2
    ZERO_POINT_ONE                        | 18.0-46.4 | 21.8-82.9
    ONE                                   | 21.7-62.6 | 52.2-129.7

    Attributes
    ----------
    max_decompression_time_in_s: float
        The maximum acceptable decompression time.
        Effects after how many saves a reset point is generated.
        A higher value leads to a higher compression ratio, has a high impact.
        A large neural network requires a higher setting to achieve the same
        compression ratio since compressing and decompressing requires
        more time with an increasing size.
    lossy: LossSetting
        If the compression pipline should be lossless or lossy.
        LOSSLESS saves the weights after each online batch as is.
        LOSSY can alter weights before they are saved and rolled out
        or even skip the rollout of a set of new weights.
        The runtime of LOSSLESS is close to the training time of the executed batch.
        LOSSY leads to a higher compression ratio. See table above for details.
        The attributes compression_speed and accuracy_limit_percent
        are only relevant if lossy is set to LOSSY.
    compression_speed: CompressionSpeed
        If the mean compression speed should be FAST or SLOW.
        Only relevant if lossy=LOSSY.
        SLOW leads to a higher compression ratio. See table above for details.
        The runtime of FAST is close to the training time of the executed batch.
        The runtime of SLOW is close to 3-5x the training time of
        the executed batch.
    accuracy_limit_percent: AccuracyLimitPercent
        The acceptable average accuracy drop compared to a lossless run.
        Only relevant if lossy=LOSSY.
        A higher value leads to a higher compression ratio.
        See table above for details.
    """
    max_decompression_time_in_s: float = 10
    lossy: LossSetting = LossSetting.LOSSLESS
    compression_speed: CompressionSpeed = CompressionSpeed.FAST
    accuracy_limit_percent: AccuracyLimitPercent = AccuracyLimitPercent.ZERO
