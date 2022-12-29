import numpy as np


def split_train_test(
        inputs: np.ndarray,
        targets: np.ndarray,
        num_eval_samples: int or float
) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    assert inputs.shape[0] == targets.shape[0], \
        'AssertionError: Mismatch between input and target shapes: ' \
        f'Inputs: {inputs.shape[0]}, Targets: {targets.shape[0]}'

    assert (isinstance(num_eval_samples, int) and num_eval_samples > 0) or \
           (isinstance(num_eval_samples, float) and 0 < num_eval_samples < 1.0), \
        'AssertionError: num_eval_samples should be an integer greater than zero or a float value between (0, 1.0), '

    if isinstance(num_eval_samples, float):
        num_eval_samples = int(num_eval_samples*inputs.shape[0])

    assert 0 < num_eval_samples < inputs.shape[0], \
        f'AssertionError: num_eval_samples should be greater than 0 and less than input samples, got {num_eval_samples}'

    n_train_samples = inputs.shape[0] - num_eval_samples
    x_train = inputs[: n_train_samples]
    y_train = targets[: n_train_samples]
    x_test = inputs[n_train_samples:]
    y_test = targets[n_train_samples:]
    return x_train, y_train, x_test, y_test


def construct_timeframes(
        samples: np.ndarray,
        timeframe_len: int,
        target_horizon_len: int,
) -> np.ndarray:
    assert timeframe_len > 1, \
        f'AssertionError: timeframe_len is expected to be greater than 1, got {timeframe_len}'

    assert timeframe_len + target_horizon_len <= samples.shape[0], \
        'AssertionError: Cannot build inputs, because samples too few samples are provided'

    return np.float64([
        samples[i: i + timeframe_len] for i in range(samples.shape[0] - target_horizon_len - timeframe_len + 1)
    ])
