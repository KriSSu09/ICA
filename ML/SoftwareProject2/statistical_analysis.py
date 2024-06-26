import numpy as np
from scipy import stats


def mean_confidence_interval(data, confidence=0.95):
    """
    Calculate the mean and the confidence interval of the given data.

    Returns:
    tuple: mean, lower bound of the confidence interval, upper bound of the confidence interval
    """
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin_of_error = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, mean - margin_of_error, mean + margin_of_error
