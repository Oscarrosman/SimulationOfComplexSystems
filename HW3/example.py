import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def erdos_renyi_random_graph(n, p):
    adjacency_matrix = np.random.rand(n, n) < p
    np.fill_diagonal(adjacency_matrix, 0)
    return np.sum(adjacency_matrix, axis=1)

def plot_degree_histogram(n, p, num_samples):
    degrees = []
    for _ in range(num_samples):
        degree_sequence = erdos_renyi_random_graph(n, p)
        degrees.extend(degree_sequence)

    plt.hist(degrees, bins=np.arange(0, n + 2) - 0.5, density=True, alpha=0.7)
    plt.xlabel('Degree (k)')
    plt.ylabel('Probability Density')
    plt.title(f'Degree Histogram for Erdős–Rényi Random Graph (n={n}, p={p})')
    plt.show()

# Example usage
n_value = 20
p_value_example = 0.2
num_samples_example = 1000
plot_degree_histogram(n_value, p_value_example, num_samples_example)
