import numpy as np

# --- fixed parameters ---
c       = 8192
gamma   = 1 / 8

# ------------------------------------------------------------------
# helper functions
# ------------------------------------------------------------------
def kullback_leibler_divergence(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

def entropy(p):
    return -p * np.log(p) - (1 - p) * np.log(1 - p)

def find_min_m(k, m_step=1, target=-98):
    """Return the smallest β₂ that drives max f(|G|) ≤ target for a given k."""
    m = m_step
    while True:
        h = int(np.ceil(c * gamma / (alpha_2 * m)))       # ⌈h⌉
        min_G = int(np.floor(k / h) + 1)
        max_G = int(alpha_2 * m)

        # guard against impossible ranges
        if min_G > max_G:
            m += m_step
            continue

        G_values = np.arange(min_G, max_G + 1)
        f_vals = []

        for G in G_values:
            p = G / m
            q = (h * G / c) ** k
            D = kullback_leibler_divergence(p, q)
            H = entropy(h * G / c)
            f_vals.append(H * c - D * m)

        if max(f_vals) <= target:
            return m, h
        m += m_step

# ------------------------------------------------------------------
# main loop – put the k’s you want to test here
# ------------------------------------------------------------------
k_values = [5, 6, 7, 8, 12, 16, 20, 24]      # <-- edit this list as you like
alpha_2_values = [1/8]   # <-- edit this list as you like
summary  = []

for alpha_2 in alpha_2_values:
    for k in k_values:
        m, h = find_min_m(k)
        summary.append((alpha_2, k, m, h))

# ------------------------------------------------------------------
# pretty-print a table
# ------------------------------------------------------------------
header = f"{'alpha_2':>9} | {'k':>6} | {'m (min)':>9} | {'h':>4}"
print(header)
print("-" * len(header))
for alpha_2, k, m, h in summary:
    print(f"{alpha_2:9.4f} | {k:6d} | {m:8d} | {h:4d}")
