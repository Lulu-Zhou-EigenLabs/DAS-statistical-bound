import numpy as np
import pandas as pd
import math

## This is used to generate the table of the minimal m value for the data recovery in DAS

# Given values
gamma = 1/8
c = 8192

# Function to calculate Kullback-Leibler Divergence
def kullback_leibler_divergence(alpha, gamma_k):
    return alpha * np.log(alpha / gamma_k) + (1 - alpha) * np.log((1 - alpha) / (1 - gamma_k))

# Function to calculate entropy H(gamma)
def entropy(gamma):
    return -gamma * np.log(gamma) - (1 - gamma) * np.log(1 - gamma)

# Given values for k
k_values = [4, 5, 6, 7, 8, 12, 16, 20, 24]
alpha_values = [1/8]

# Calculate results for each k
results = []

for alpha in alpha_values:
    for k in k_values:
        gamma_k = gamma ** k
        D_value = kullback_leibler_divergence(alpha, gamma_k)
        H_value = entropy(gamma)
        
        # Solve for m from the inequality
        # m >= (c * H_value + 88.73) / D_value
        m_min = (c * H_value + 88.73) / D_value
        results.append([alpha, k, math.ceil(m_min)])

# Create DataFrame
df = pd.DataFrame(results, columns=["Alpha", "k", "min m"])

print(df)
