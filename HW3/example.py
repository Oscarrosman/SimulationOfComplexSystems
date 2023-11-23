import numpy as np

def erdos_renyi_adjacency_matrix(n, p):
    """
    Generate an Erdős–Rényi random graph adjacency matrix.
    """
    return np.random.choice([0, 1], size=(n, n), p=[1-p, p])

def compute_path_length_matrix(A):
    """
    Compute the path length matrix L for an Erdős–Rényi random graph.
    """
    n = A.shape[0]
    L = np.full((n, n), -1)  # Initialize L with -1

    t = 1
    while np.any(L == -1):
        At = np.linalg.matrix_power(A, t)
        for i in range(n):
            for j in range(n):
                if A[i, j] != 0 and L[i, j] == L[j, i] == -1:
                    L[i, j] = L[j, i] = t
        t += 1

    return L

# Example usage:
n = 5  # Number of nodes
p = 0.3  # Probability of edge existence

# Generate Erdős–Rényi random graph adjacency matrix
A = erdos_renyi_adjacency_matrix(n, p)

# Compute path length matrix
L = compute_path_length_matrix(A)

print("Erdős–Rényi Random Graph Adjacency Matrix:")
print(A)
print("\nPath Length Matrix:")
print(L)
