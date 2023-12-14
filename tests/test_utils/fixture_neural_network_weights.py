import numpy as np


def weights_fixture(seed=123):
    rng = np.random.default_rng(seed=seed)
    return [rng.random((5, 5), np.float32) for _ in range(3)]


def deep_copy_weights(weights_to_copy):
    return [r.copy() for r in weights_to_copy]
