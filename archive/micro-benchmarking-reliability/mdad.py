import math
import numpy as np
from collections import defaultdict

def bootstrap_mean_ci(data, B=1000, confidence_level=0.95):
    """
    Calculates the bootstrap mean and 95% confidence interval.

    Args:
    data: A list or numpy array of the sample data.
    B: The number of bootstrap samples to generate.
    confidence_level: The desired confidence level (default is 0.95).

    Returns:
    A tuple containing:
        - The bootstrap mean.
        - The lower and upper bounds of the confidence interval.
    """

    # 1. Calculate the sample mean and standard deviation
    sample_mean = np.mean(data)
    
    # 2. Generate bootstrap samples and calculate the mean for each
    bootstrap_means = []
    for _ in range(B):
        # Generate a bootstrap sample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        # Calculate the mean of the bootstrap sample
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)

    # 3. Determine the confidence interval
    lower_bound = np.quantile(bootstrap_means, (1 - confidence_level) / 2)
    upper_bound = np.quantile(bootstrap_means, 1 - (1 - confidence_level) / 2)

    return sample_mean, lower_bound, upper_bound

def calculate_mdad(acc_diffs, correct_ranks, mdad_threshold=0.8, resolution=0.5):
    round_multiplier = 1/resolution * 100
    # first bin the data
    data = zip(acc_diffs, correct_ranks)
    bins = sorted(list(set(acc_diffs)))
    bins = [math.floor(abs(x) * round_multiplier) / round_multiplier for x in bins]
    bin_to_data = defaultdict(list)
    for x, y in data:
        bin_to_data[x].append(y)
    first_m_over_threshold = 0.5
    first_l_over_threshold = 0.5
    first_u_over_threshold = 0.5
    for x in bins:
        # print(x, len(bin_to_data))
        m, l, u = bootstrap_mean_ci(bin_to_data[x])
        if m >= mdad_threshold and first_m_over_threshold == 0.5:
            first_m_over_threshold = x
        if l >= mdad_threshold and first_l_over_threshold == 0.5:
            first_l_over_threshold = x
        if u >= mdad_threshold and first_u_over_threshold == 0.5:
            first_u_over_threshold = x

    # Note:
    # lower bound on correctness = upper bound on MDAD
    # upper bound on correctness = lower bound on MDAD

    # take the max error bars on either side
    max_diff = max(first_m_over_threshold - first_u_over_threshold,
                   first_l_over_threshold - first_m_over_threshold)
    lower_bound_mdad = first_m_over_threshold - max_diff
    upper_bound_mdad = first_m_over_threshold + max_diff

    return first_m_over_threshold, lower_bound_mdad, upper_bound_mdad