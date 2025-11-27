# ==================================
# spectral_coloring.py
# ==================================

import numpy as np
from time import time
from core_utils import (
    refine_by_greedy_within_clusters,
    coloring_is_valid,
    relabel_coloring_sequential,
)


def _adjacency_matrix(graph):
    n = graph.number_of_nodes()
    A = np.zeros((n, n), dtype=float)
    for u, v in graph.edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def _normalized_laplacian(A: np.ndarray):
    d = A.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    I = np.eye(A.shape[0])
    L_sym = I - D_inv_sqrt @ (A + np.eye(A.shape[0])) @ D_inv_sqrt  # add self-loops for stability
    return L_sym


def _kmeans_numpy(X: np.ndarray, k: int, rng, n_init: int = 10, max_iter: int = 100):
    n, d = X.shape
    best_inertia = np.inf
    best_labels = None
    for _ in range(n_init):
        centers = np.empty((k, d))
        # kmeans++ Initialization
        idx0 = rng.integers(0, n)
        centers[0] = X[idx0]
        closest_dist_sq = np.full(n, np.inf)
        for ci in range(1, k):
            dist_sq = np.sum((X[:, None, :] - centers[None, :ci, :]) ** 2, axis=2).min(axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
            probs = closest_dist_sq / closest_dist_sq.sum()
            next_idx = rng.choice(n, p=probs)
            centers[ci] = X[next_idx]

        labels = np.zeros(n, dtype=int)
        for _it in range(max_iter):
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dists.argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                centers[j] = X[mask].mean(axis=0) if np.any(mask) else X[rng.integers(0, n)]
        inertia = 0.0
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                inertia += ((X[mask] - centers[j]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels


def spectral_coloring(noncomm_graph, upper_bound_k: int, rng):
    """Spectral clustering on normalized Laplacian + greedy refinement."""
    start = time()
    n = noncomm_graph.number_of_nodes()
    if n == 0:
        return {}, 0, 0.0
    if upper_bound_k <= 1:
        coloring = {i: 0 for i in noncomm_graph.nodes}
        return coloring, 1, 0.0

    A = _adjacency_matrix(noncomm_graph)
    L = _normalized_laplacian(A)
    evals, evecs = np.linalg.eigh(L)

    best_coloring, best_k = None, None
    for k in range(2, max(2, upper_bound_k) + 1):
        X = evecs[:, 1:k+1]  # skip trivial eigenvector
        labels = _kmeans_numpy(X, k=k, rng=rng, n_init=5, max_iter=100)
        prelim = {i: int(labels[i]) for i in range(n)}
        refined = refine_by_greedy_within_clusters(prelim, noncomm_graph)
        if coloring_is_valid(refined, noncomm_graph):
            best_coloring = refined
            best_k = len(set(refined.values()))
            break

    if best_coloring is None:
        # Fallback: DSATUR greedy
        from networkx.algorithms.coloring import greedy_color
        refined = greedy_color(noncomm_graph, strategy="saturation_largest_first")
        best_coloring = refined
        best_k = len(set(refined.values()))

    elapsed = time() - start
    return relabel_coloring_sequential(best_coloring), best_k, elapsed