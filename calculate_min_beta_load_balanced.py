import numpy as np

# --- fixed parameters ---
c       = 8192
gamma   = 1 / 8
alpha_2 = 1 / 8

# ------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------
def kullback_leibler_divergence(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def entropy(p):
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def find_min_beta(k, beta_step=0.01, target=-0.012):
    """Return the smallest β₂ that drives max f(|G|) ≤ target for a given k."""
    beta = beta_step
    while True:
        h = int(np.ceil(c * gamma / (alpha_2 * beta * c)))       # ⌈h⌉
        min_G = int(np.floor(k / h) + 1)
        max_G = int(alpha_2 * beta * c)

        # guard against impossible ranges
        if min_G > max_G:
            beta += beta_step
            continue

        G_values = np.arange(min_G, max_G + 1)
        f_vals = []

        for G in G_values:
            p = G / (beta * c)
            q = (h * G / c) ** k
            D = kullback_leibler_divergence(p, q)
            H = entropy(h * G / c)
            f_vals.append(-D * beta + H)

        if max(f_vals) <= target:
            return beta, h
        beta += beta_step

# ------------------------------------------------------------------
# main loop – put the k’s you want to test here
# ------------------------------------------------------------------
k_values = [5, 6, 7, 8, 12, 16, 20, 24]      # <-- edit this list as you like
summary  = []

for k in k_values:
    beta, h = find_min_beta(k)
    summary.append((k, beta, beta * c, h))

# ------------------------------------------------------------------
# pretty-print a table
# ------------------------------------------------------------------
header = f"{'k':>6} | {'β₂ (min)':>9} | {'β₂·c':>8} | {'h':>4}"
print(header)
print("-" * len(header))
for k, beta, beta_c, h in summary:
    print(f"{k:6d} | {beta:9.4f} | {beta_c:8.1f} | {h:4d}")
