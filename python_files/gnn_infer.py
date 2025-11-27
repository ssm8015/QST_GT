# ==================================
# gnn_infer.py
# ==================================

import numpy as np
from time import time
from core_utils import (
    refine_by_greedy_within_clusters, coloring_is_valid, relabel_coloring_sequential,
)
from gnn_model import TORCH_AVAILABLE, load_model, build_inputs_from_graph


def _kmeans_numpy(X: np.ndarray, k: int, rng, n_init: int = 10, max_iter: int = 100):
    n, d = X.shape
    best_inertia = np.inf
    best_labels = None
    for _ in range(n_init):
        centers = np.empty((k, d))
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


def color_with_trained_gnn(noncomm_graph, model_path: str, upper_bound_k: int, rng, device='cpu'):
    """Load a pretrained GNN and color a (possibly larger) graph without additional training.

    Steps: forward pass ; embeddings ; k-means (k ≤ upper_bound_k) ; greedy refinement.
    Returns (coloring_dict, num_colors, elapsed_sec).
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available: cannot run GNN inference.")

    start_total = time()
    model, cfg = load_model(model_path, device=device)

    # Build inputs for this new graph (size can differ from training graphs)
    t_X, t_Ahat = build_inputs_from_graph(noncomm_graph, device=device)

    # Forward to get embeddings
    import torch
    with torch.no_grad():
        Z = model(t_X, t_Ahat).cpu().numpy()

    n = noncomm_graph.number_of_nodes()
    if n == 0:
        return {}, 0, 0.0

    best_coloring = None
    best_k = None
    for k in range(2, max(2, upper_bound_k) + 1):
        labels = _kmeans_numpy(Z, k=k, rng=rng, n_init=5, max_iter=100)
        prelim = {i: int(labels[i]) for i in range(n)}
        refined = refine_by_greedy_within_clusters(prelim, noncomm_graph)
        if coloring_is_valid(refined, noncomm_graph):
            best_coloring = refined
            best_k = len(set(refined.values()))
            break

    if best_coloring is None:
        # Fallback: single-color or greedy (should be rare after refinement)
        from networkx.algorithms.coloring import greedy_color
        refined = greedy_color(noncomm_graph, strategy="saturation_largest_first")
        best_coloring = refined
        best_k = len(set(refined.values()))

    elapsed = time() - start_total
    return relabel_coloring_sequential(best_coloring), best_k, elapsed