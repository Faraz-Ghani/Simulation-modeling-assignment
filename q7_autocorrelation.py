import numpy as np
from scipy import stats

def parse_input_numbers(input_text):
    """Parse numbers from text input."""
    numbers = []
    lines = input_text.strip().split('\n')
    for line in lines:
        parts = line.replace(',', ' ').split()
        for part in parts:
            try:
                numbers.append(float(part))
            except ValueError:
                continue
    return numbers


def autocorrelation_test(numbers, i, m, alpha=0.05):
    """
    Parametric test for autocorrelation using the serial correlation coefficient (r_m).
    
    Hypotheses:
    H0: ρ_i,m = 0  → numbers are independent
    H1: ρ_i,m ≠ 0  → numbers are dependent
    
    Parameters:
    - numbers: list of random numbers
    - i: starting position (1-indexed)
    - m: lag between numbers
    - alpha: significance level
    
    Returns:
    - Dictionary of test results
    """
    N = len(numbers)
    if N < (m + i):
        return {
            'error': f'Not enough data for i={i}, m={m}. Need at least {i+m} numbers, got {N}.',
            'positions': [],
            'subsequence': []
        }
    
    numbers = np.array(numbers)
    mean_u = np.mean(numbers)
    
    # M = N - m
    M = N - m
    
    # Compute autocorrelation coefficient
    numerator = np.sum((numbers[:M] - mean_u) * (numbers[m:] - mean_u))
    denominator = np.sum((numbers - mean_u) ** 2)
    r_m = numerator / denominator
    
    # Test statistic under H0
    Z0 = r_m * np.sqrt(N)
    Z_critical = stats.norm.ppf(1 - alpha / 2)
    
    reject_H0 = abs(Z0) > Z_critical
    conclusion = (
        'Numbers are DEPENDENT (autocorrelation present)'
        if reject_H0 else
        'Numbers are INDEPENDENT (no autocorrelation)'
    )
    
    # Positions used for demonstration (optional visualization)
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
