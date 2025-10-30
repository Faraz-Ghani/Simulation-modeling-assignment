import numpy as np
from scipy import stats

def parse_input_numbers(input_text):
    """
    Takes raw text input and pulls out all the numbers from it.
    Works even if numbers are separated by spaces, commas, or new lines.
    """
    numbers = []
    lines = input_text.strip().split('\n')  # split by lines in case input has multiple rows

    for line in lines:
        # replace commas with spaces to handle both "1,2,3" and "1 2 3"
        parts = line.replace(',', ' ').split()
        for part in parts:
            try:
                # try converting each piece into a float
                numbers.append(float(part))
            except ValueError:
                # skip anything that isn't a number
                continue
    return numbers


def autocorrelation_test(numbers, i, m, alpha=0.05):
    """
    Performs a simple autocorrelation test on a sequence of numbers.
    
    H0: numbers are independent (no autocorrelation)
    H1: numbers are dependent (autocorrelation exists)
    """
    N = len(numbers)  # total number of values

    # make sure we have enough numbers for the lag and starting index
    if N < (m + i):
        return {
            'error': f'Not enough data for i={i}, m={m}. Need at least {i+m} numbers, got {N}.',
            'positions': [],
            'subsequence': []
        }

    numbers = np.array(numbers)
    mean_u = np.mean(numbers)  # mean of all numbers

    # M = how many pairs we can form after applying lag
    M = N - m

    # numerator: sum of (x_i - mean) * (x_(i+m) - mean)
    numerator = np.sum((numbers[:M] - mean_u) * (numbers[m:] - mean_u))

    # denominator: sum of (x_i - mean)^2 for all i
    denominator = np.sum((numbers - mean_u) ** 2)

    # serial correlation coefficient (r_m)
    r_m = numerator / denominator

    # test statistic under H0 ~ N(0,1)
    Z0 = r_m * np.sqrt(N)

    # two-tailed critical value for the given alpha
    Z_critical = stats.norm.ppf(1 - alpha / 2)

    # if |Z0| > Z_critical, we reject H0
    reject_H0 = abs(Z0) > Z_critical

    # readable result
    conclusion = (
        'Numbers are DEPENDENT (autocorrelation present)'
        if reject_H0 else
        'Numbers are INDEPENDENT (no autocorrelation)'
    )

    # optional – just to show which positions were used
    positions = list(range(1, N + 1))
    subsequence = numbers.tolist()

    return {
        'N': N,
        'i': i,
        'm': m,
        'M': M,
        'alpha': alpha,
        'mean': mean_u,
        'r_m': r_m,
        'Z0': Z0,
        'Z_critical': Z_critical,
        'reject_H0': reject_H0,
        'conclusion': conclusion,
        'positions': positions,
        'subsequence': subsequence,
        'error': None
    }
