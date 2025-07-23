import numpy as np
import pandas as pd
import math

## This is used to generate the table of the minimal beta_1 value for the data recovery in DAS

# Given values
gamma = 1/8
c = 8192

# Function to calculate Kullback-Leibler Divergence
def kullback_leibler_divergence(alpha, gamma_k):
    return alpha * np.log(alpha / gamma_k) + (1 - alpha) * np.log((1 - alpha) / (1 - gamma_k))

# Function to calculate entropy H(gamma)
def entropy(gamma):
    return -gamma * np.log(gamma) - (1 - gamma) * np.log(1 - gamma)

def round_up_third_decimal(x):
    return math.ceil(x * 1000) / 1000

# Given values for k
k_values = [8]
alpha_values = [1/8, 1/6, 1/4, 1/3]

# Calculate results for each k
results = []

for alpha in alpha_values:
    for k in k_values:
        gamma_k = gamma ** k
        D_value = kullback_leibler_divergence(alpha, gamma_k)
        H_value = entropy(gamma)
        
        # Solve for beta_1 from the inequality
        # beta_1 >= (H_value + 0.011) / D_value
        beta_min = (H_value + 0.011) / D_value
        results.append([alpha, k, round_up_third_decimal(beta_min), math.ceil(beta_min*c)])

# Create DataFrame
df = pd.DataFrame(results, columns=["Alpha", "k", "Minimal Beta", "beta * c"])

print(df)