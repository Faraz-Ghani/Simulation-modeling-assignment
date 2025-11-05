import numpy as np
from scipy import stats

def parse_input_numbers(input_text):
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
    H0: numbers are independent (no autocorrelation)
    H1: numbers are dependent (autocorrelation exists)
    """
    N = len(numbers)  # total number of values
    numbers_array = np.array(numbers)

    # The minimum required is i + (M+1)m <= N.
    if N < i or N < (i + m):
        return {
            'error': f'Not enough data. Need starting index i={i} and lag m={m}, but got N={N}.',
            'positions': [],
            'subsequence': [],
            'r_m': 0, 'Z0': 0, 'Z_critical': 0, 'reject_H0': False, 'conclusion': ''
        }

    # M = Number of pairs
    M = int(np.floor((N - i) / m)) - 1
    
    if M < 0:
        return {
            'error': f'Not enough pairs can be formed. M = {M}. Check i={i} and m={m} against N={N}.',
            'positions': [],
            'subsequence': [],
            'r_m': 0, 'Z0': 0, 'Z_critical': 0, 'reject_H0': False, 'conclusion': ''
        }
    
    # extract the subsequences for R_{i+km} and R_{i+(k+1)m}
    R_km = numbers_array[ (i-1) + np.arange(M + 1) * m ]
    
    R_k_plus_1_m = numbers_array[ (i-1) + np.arange(1, M + 2) * m ]

    sum_R_R = np.sum(R_km * R_k_plus_1_m)
    rho_hat_im = (1 / (M + 1)) * sum_R_R - 0.25

    sigma_rho_hat_im = np.sqrt(13 * M + 7) / (12 * (M + 1))

    # Z0 = rho_hat_im / sigma_rho_hat_im
    Z0 = rho_hat_im / sigma_rho_hat_im

    mean_u = np.mean(numbers) 

    # serial correlation coefficient (r_m)
    r_m = rho_hat_im 

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
    subsequence = numbers_array.tolist()

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