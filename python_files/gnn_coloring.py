# gnn_coloring.py
# ==================================
# Per-instance unsupervised GNN (Graph Autoencoder) that trains on the given graph
# and returns a valid coloring via embeddings → k-means → greedy refinement.

from time import time
import numpy as np
import networkx as nx

from core_utils import (
    refine_by_greedy_within_clusters,
    coloring_is_valid,
    relabel_coloring_sequential,
)
from spectral_coloring import spectral_coloring  # fallback if torch not available
from gnn_model import GAE, build_inputs_from_graph, TORCH_AVAILABLE

# Public API
__all__ = ["gnn_coloring", "_kmeans_numpy", "TORCH_AVAILABLE"]


# ---- helpers ---- #

def _adjacency_matrix(graph: nx.Graph) -> np.ndarray:
    n = graph.number_of_nodes()
    A = np.zeros((n, n), dtype=float)
    for u, v in graph.edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def _kmeans_numpy(X: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 10, max_iter: int = 100) -> np.ndarray:
    """Lightweight k-means with k-means++-style init; returns labels (n,)."""
    n, d = X.shape
    best_inertia = np.inf
    best_labels = None
    for _ in range(max(1, n_init)):
        centers = np.empty((k, d))
        # first center
        idx0 = int(rng.integers(0, n))
        centers[0] = X[idx0]
        # remaining centers via approximate k-means++
        closest_dist_sq = np.full(n, np.inf)
        for ci in range(1, k):
            dist_sq = np.sum((X[:, None, :] - centers[None, :ci, :]) ** 2, axis=2).min(axis=1)
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)
            denom = float(closest_dist_sq.sum()) or 1.0
            probs = closest_dist_sq / denom
            next_idx = int(rng.choice(n, p=probs))
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
                if np.any(mask):
                    centers[j] = X[mask].mean(axis=0)
                else:
                    centers[j] = X[int(rng.integers(0, n))]
        # inertia
        inertia = 0.0
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                inertia += ((X[mask] - centers[j]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels


def _sample_neg_edges(n: int, A: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    neg = []
    tries = 0
    limit = max(10 * num_samples, 100)
    while len(neg) < num_samples and tries < limit:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            tries += 1
            continue
        u, v = (i, j) if i < j else (j, i)
        if A[u, v] == 0:
            neg.append((u, v))
        tries += 1
    if len(neg) < num_samples:
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 0 and len(neg) < num_samples:
                    neg.append((u, v))
    return np.array(neg, dtype=int)


# ---- main API ---- #

def gnn_coloring(noncomm_graph: nx.Graph, upper_bound_k: int, rng: np.random.Generator,
                 epochs: int = 200, hidden_dim: int = 32, emb_dim: int = 16, lr: float = 1e-2,
                 verbose: bool = False):
    """Train a small Graph Autoencoder on *this* graph and color it.

    Returns (coloring_dict, num_colors, elapsed_sec, used_gnn: bool).
    If PyTorch is not available, falls back to spectral_coloring and returns used_gnn=False.
    """
    n = noncomm_graph.number_of_nodes()
    start_total = time()

    if n == 0:
        return {}, 0, 0.0, False

    if upper_bound_k <= 1:
        coloring = {i: 0 for i in noncomm_graph.nodes}
        return coloring, 1, 0.0, False

    if not TORCH_AVAILABLE:
        col, k, t_spec = spectral_coloring(noncomm_graph, upper_bound_k, rng)
        return col, k, t_spec, False

    # Build inputs
    import torch
    device = torch.device('cpu')
    t_X, t_Ahat = build_inputs_from_graph(noncomm_graph, device=device)

    # Positive/negative pairs for link prediction
    A = _adjacency_matrix(noncomm_graph)
    pos_pairs = np.array([(u, v) if u < v else (v, u) for u, v in noncomm_graph.edges], dtype=int)
    num_pos = len(pos_pairs)
    neg_pairs = _sample_neg_edges(n, A, num_pos, rng)

    t_pos = torch.from_numpy(pos_pairs).long().to(device)
    t_neg = torch.from_numpy(neg_pairs).long().to(device)

    # Model & training
    torch.manual_seed(0)
    model = GAE(in_dim=t_X.shape[1], hidden_dim=hidden_dim, emb_dim=emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        Z = model(t_X, t_Ahat)
        pos_scores = (Z[t_pos[:, 0]] * Z[t_pos[:, 1]]).sum(dim=1)
        neg_scores = (Z[t_neg[:, 0]] * Z[t_neg[:, 1]]).sum(dim=1)
        loss = bce(pos_scores, torch.ones_like(pos_scores)) + bce(neg_scores, torch.zeros_like(neg_scores))
        loss.backward()
        opt.step()
        if verbose and (epoch + 1) % 50 == 0:
            print(f"[GNN] epoch {epoch+1:03d} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        Z = model(t_X, t_Ahat).cpu().numpy()

    # Cluster embeddings and refine
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
        # Greedy fallback (should be rare after refinement)
        from networkx.algorithms.coloring import greedy_color
        refined = greedy_color(noncomm_graph, strategy="saturation_largest_first")
        best_coloring = refined
        best_k = len(set(refined.values()))

    elapsed = time() - start_total
    return relabel_coloring_sequential(best_coloring), best_k, elapsed, True