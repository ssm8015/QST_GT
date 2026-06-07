import numpy as np
import networkx as nx
from itertools import combinations
from collections import defaultdict
from time import time

# ILP
from pulp import (
    LpProblem, LpVariable, lpSum, LpMinimize, LpBinary,
    LpStatus, value, PULP_CBC_CMD
)

# Heuristic coloring
from networkx.algorithms.coloring import greedy_color

# Torch optional
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

def generate_su_d_basis(d: int):
    """Generate a Hermitian, traceless basis for su(d) (Pauli-like generalization).
    Returns a list of dxd complex numpy arrays. Includes symmetric, antisymmetric, and diagonal gens.
    """
    basis = []
    for i in range(d):
        for j in range(i + 1, d):
            mat = np.zeros((d, d), dtype=complex)
            mat[i, j] = 1
            mat[j, i] = 1
            basis.append(mat)

            mat = np.zeros((d, d), dtype=complex)
            mat[i, j] = -1j
            mat[j, i] = 1j
            basis.append(mat)

    for i in range(1, d):
        diag = np.zeros((d, d), dtype=complex)
        for j in range(i):
            diag[j, j] = 1
        diag[i, i] = -i
        diag /= np.sqrt(i * (i + 1))
        basis.append(diag)

    return basis


def tensor_product_operators(single_site_ops, N: int):
    from itertools import product
    ops = []
    labels = []
    basis_size = len(single_site_ops)
    for idxs in product(range(basis_size), repeat=N):
        label = "-".join(map(str, idxs))
        op = single_site_ops[idxs[0]]
        for i in idxs[1:]:
            op = np.kron(op, single_site_ops[i])
        ops.append(op)
        labels.append(label)
    return ops, labels


def generate_non_commutativity_graph(ops):
    G = nx.Graph()
    G.add_nodes_from(range(len(ops)))
    for i, j in combinations(range(len(ops)), 2):
        if not np.allclose(ops[i] @ ops[j], ops[j] @ ops[i]):
            G.add_edge(i, j)
    return G


def generate_commutativity_graph(ops):
    G = nx.Graph()
    G.add_nodes_from(range(len(ops)))
    for i, j in combinations(range(len(ops)), 2):
        if np.allclose(ops[i] @ ops[j], ops[j] @ ops[i]):
            G.add_edge(i, j)
    return G


# ---- coloring helpers ---- #
def coloring_is_valid(coloring: dict, G: nx.Graph) -> bool:
    for u, v in G.edges():
        if coloring.get(u) == coloring.get(v):
            return False
    return True


def relabel_coloring_sequential(coloring: dict) -> dict:
    """Map color labels to 0..K-1 consistently."""
    uniq = sorted(set(coloring.values()))
    remap = {c: i for i, c in enumerate(uniq)}
    return {n: remap[c] for n, c in coloring.items()}


def refine_by_greedy_within_clusters(prelim: dict, G: nx.Graph) -> dict:
    """Within each preliminary cluster label, run greedy coloring on the induced subgraph
    and offset color ids to keep them disjoint; then merge.
    """
    out = {}
    offset = 0
    for label in sorted(set(prelim.values())):
        nodes = [n for n, lab in prelim.items() if lab == label]
        H = G.subgraph(nodes).copy()
        local = greedy_color(H, strategy="saturation_largest_first")
        # shift
        for n in nodes:
            out[n] = local[n] + offset
        offset = max(out.values()) + 1 if out else 0
    return out

def dsatur_color(G: nx.Graph):
    coloring = greedy_color(G, strategy="saturation_largest_first")
    k = len(set(coloring.values())) if coloring else 0
    return coloring, k


def rlf_color(G: nx.Graph):
    coloring = greedy_color(G, strategy="largest_first")
    k = len(set(coloring.values())) if coloring else 0
    return coloring, k

def _adjacency_matrix(G: nx.Graph) -> np.ndarray:
    n = G.number_of_nodes()
    A = np.zeros((n, n), dtype=float)
    for u, v in G.edges():
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def _kmeans_numpy(X: np.ndarray, k: int, rng: np.random.Generator, n_init: int = 10, max_iter: int = 100) -> np.ndarray:
    n, d = X.shape
    best_inertia = np.inf
    best_labels = None
    for _ in range(max(1, n_init)):
        centers = np.empty((k, d))
        # first center
        idx0 = int(rng.integers(0, n))
        centers[0] = X[idx0]
        # rest via approximate k-means++
        closest = np.full(n, np.inf)
        for ci in range(1, k):
            dist_sq = ((X[:, None, :] - centers[None, :ci, :]) ** 2).sum(axis=2).min(axis=1)
            closest = np.minimum(closest, dist_sq)
            denom = float(closest.sum()) or 1.0
            probs = closest / denom
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


def spectral_coloring(G: nx.Graph, upper_bound_k: int, rng: np.random.Generator):
    """Use Laplacian eigen-embeddings + k-means to propose clusters; then refine.

    Tries k = 2..upper_bound_k and returns the first valid coloring. If none valid, falls back to greedy.

    Returns (coloring_dict, num_colors, elapsed_sec).
    """
    start = time()
    n = G.number_of_nodes()
    if n == 0:
        return {}, 0, 0.0
    if upper_bound_k <= 1:
        col = {i: 0 for i in G.nodes()}
        return col, 1, 0.0

    A = _adjacency_matrix(G)
    D = np.diag(A.sum(axis=1))
    L = D - A
    # compute a bunch of eigenvectors of L (smallest eigenvalues)
    # to be safe, ask for up to min(upper_bound_k, n) eigenvectors
    m = int(min(max(2, upper_bound_k), n))
    w, V = np.linalg.eigh(L)  # full since graphs are small/moderate
    # take the m smallest eigenvectors (skip the first all-ones if present)
    X = V[:, :m]

    best = None
    best_k = None
    for k in range(2, max(2, upper_bound_k) + 1):
        labels = _kmeans_numpy(X, k=k, rng=rng, n_init=5, max_iter=100)
        prelim = {i: int(labels[i]) for i in range(n)}
        refined = refine_by_greedy_within_clusters(prelim, G)
        if coloring_is_valid(refined, G):
            best = refined
            best_k = len(set(refined.values()))
            break

    if best is None:
        # fallback: greedy DSATUR
        prelim = greedy_color(G, strategy="saturation_largest_first")
        best = prelim
        best_k = len(set(prelim.values()))
    return relabel_coloring_sequential(best), best_k, time() - start

def solve_ilp_clique_cover(comm_graph: nx.Graph):
    cliques = list(nx.find_cliques(comm_graph))
    prob = LpProblem("MinCliqueCover", LpMinimize)
    vars = [LpVariable(f"c{i}", cat=LpBinary) for i in range(len(cliques))]
    prob += lpSum(vars)
    for v in comm_graph.nodes:
        prob += lpSum(vars[i] for i, clique in enumerate(cliques) if v in clique) >= 1
    start = time()
    prob.solve(PULP_CBC_CMD(msg=0))
    elapsed = time() - start
    selected_cliques = [cliques[i] for i in range(len(cliques)) if value(vars[i]) > 0.5]
    node_to_color = {}
    for color, clique in enumerate(selected_cliques):
        for node in clique:
            if node not in node_to_color:
                node_to_color[node] = color
    return node_to_color, elapsed, LpStatus[prob.status]

# ---- GNN model and IO ---- #
def normalized_adj_with_selfloops(A: np.ndarray):
    A_sl = A + np.eye(A.shape[0])
    d = A_sl.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ A_sl @ D_inv_sqrt


def make_node_features(A: np.ndarray):
    deg = A.sum(axis=1, keepdims=True)
    ones = np.ones_like(deg)
    X = np.hstack([deg / max(1.0, float(deg.max() or 1.0)), ones])
    return X.astype(np.float32)


class GCNLayer(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_dim, out_dim):
        if not TORCH_AVAILABLE: return
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X, A_hat):
        return A_hat @ self.lin(X)


class GAE(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, in_dim=2, hidden_dim=32, emb_dim=16):
        if not TORCH_AVAILABLE: return
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, emb_dim)

    def forward(self, X, A_hat):
        Z = self.gcn1(X, A_hat)
        Z = F.relu(Z)
        Z = self.gcn2(Z, A_hat)
        return Z


def build_inputs_from_graph(G: nx.Graph, device='cpu'):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for GNN training/inference.")
    A = _adjacency_matrix(G)
    A_hat = normalized_adj_with_selfloops(A).astype(np.float32)
    X = make_node_features(A)
    t_X = torch.from_numpy(X).to(device)
    t_Ahat = torch.from_numpy(A_hat).to(device)
    return t_X, t_Ahat


def save_model(model, path: str, in_dim: int, hidden_dim: int, emb_dim: int):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available; cannot save model.")
    payload = {
        'state_dict': model.state_dict(),
        'config': {'in_dim': in_dim, 'hidden_dim': hidden_dim, 'emb_dim': emb_dim},
    }
    torch.save(payload, path)


def load_model(path: str, device='cpu'):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available; cannot load model.")
    payload = torch.load(path, map_location=device)
    cfg = payload['config']
    model = GAE(in_dim=cfg['in_dim'], hidden_dim=cfg['hidden_dim'], emb_dim=cfg['emb_dim'])
    model.load_state_dict(payload['state_dict'])
    model.to(device)
    model.eval()
    return model, cfg

def _sample_neg_edges(n: int, A: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    neg = []
    tries = 0
    limit = max(10 * num_samples, 100)
    while len(neg) < num_samples and tries < limit:
        i = int(rng.integers(0, n)); j = int(rng.integers(0, n))
        if i == j: 
            tries += 1; 
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


def gnn_coloring(noncomm_graph: nx.Graph, upper_bound_k: int, rng: np.random.Generator,
                 epochs: int = 200, hidden_dim: int = 32, emb_dim: int = 16, lr: float = 1e-2,
                 verbose: bool = False):
    """Per-instance unsupervised GAE: trains on the given graph, outputs a coloring.

    Returns (coloring_dict, num_colors, elapsed_sec, used_gnn: bool).
    """
    start_total = time()
    n = noncomm_graph.number_of_nodes()
    if n == 0:
        return {}, 0, 0.0, False
    if upper_bound_k <= 1:
        return {i: 0 for i in noncomm_graph.nodes()}, 1, 0.0, False
    if not TORCH_AVAILABLE:
        col, k, t = spectral_coloring(noncomm_graph, upper_bound_k, rng)
        return col, k, t, False

    device = torch.device('cpu')
    t_X, t_Ahat = build_inputs_from_graph(noncomm_graph, device=device)

    A = _adjacency_matrix(noncomm_graph)
    pos_pairs = np.array([(u, v) if u < v else (v, u) for u, v in noncomm_graph.edges()], dtype=int)
    num_pos = len(pos_pairs)
    neg_pairs = _sample_neg_edges(n, A, num_pos, rng)

    t_pos = torch.from_numpy(pos_pairs).long().to(device)
    t_neg = torch.from_numpy(neg_pairs).long().to(device)

    torch.manual_seed(0)
    model = GAE(in_dim=t_X.shape[1], hidden_dim=hidden_dim, emb_dim=emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        Z = model(t_X, t_Ahat)
        pos_scores = (Z[t_pos[:, 0]] * Z[t_pos[:, 1]]).sum(dim=1)
        neg_scores = (Z[t_neg[:, 0]] * Z[t_neg[:, 1]]).sum(dim=1)
        loss = bce(pos_scores, torch.ones_like(pos_scores)) + bce(neg_scores, torch.zeros_like(neg_scores))
        loss.backward()
        opt.step()
        if verbose and ((ep + 1) % 50 == 0 or ep == 0):
            print(f"[GNN] epoch {ep+1:03d} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        Z = model(t_X, t_Ahat).cpu().numpy()

    best, best_k = None, None
    for k in range(2, max(2, upper_bound_k) + 1):
        labels = _kmeans_numpy(Z, k=k, rng=rng, n_init=5, max_iter=100)
        prelim = {i: int(labels[i]) for i in range(n)}
        refined = refine_by_greedy_within_clusters(prelim, noncomm_graph)
        if coloring_is_valid(refined, noncomm_graph):
            best = refined
            best_k = len(set(refined.values()))
            break
    if best is None:
        prelim = greedy_color(noncomm_graph, strategy="saturation_largest_first")
        best = prelim
        best_k = len(set(prelim.values()))
    return relabel_coloring_sequential(best), best_k, time() - start_total, True

def _graph_from_random_ops(d, N, M, seed):
    rng = np.random.default_rng(seed)
    identity = np.eye(d, dtype=complex)
    su = generate_su_d_basis(d)
    single_site_ops = [identity] + su
    ops, _ = tensor_product_operators(single_site_ops, N)
    dim = d ** N
    I = np.eye(dim)
    ops = [op for op in ops if not np.allclose(op, I)]
    idx = rng.choice(len(ops), size=M, replace=False)
    ops_sel = [ops[i] for i in idx]
    G = generate_non_commutativity_graph(ops_sel)
    return G


def train_gnn_small(out_path: str, graphs: int = 50, d: int = 3, N: int = 1, M: int = 12,
                    seed0: int = 0, epochs: int = 300, hidden: int = 64, emb: int = 32, lr: float = 1e-2,
                    pairs_per_graph: int = 512):
    """Train a GAE on many small random graphs and save weights for reuse."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available: cannot train GNN.")
    device = torch.device('cpu')
    in_dim = 2
    model = GAE(in_dim=in_dim, hidden_dim=hidden, emb_dim=emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(seed0)
    train_graphs = [_graph_from_random_ops(d, N, M, seed0 + i) for i in range(graphs)]
    print(f"Training on {len(train_graphs)} graphs, epochs={epochs}, pairs/graph={pairs_per_graph}")
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for G in train_graphs:
            t_X, t_Ahat = build_inputs_from_graph(G, device=device)
            A = _adjacency_matrix(G)
            # pos/neg sampling
            pos = np.array([(u, v) if u < v else (v, u) for u, v in G.edges()], dtype=int)
            neg = _sample_neg_edges(G.number_of_nodes(), A, max(1, min(pairs_per_graph, len(pos))), rng)
            if len(pos) == 0 or len(neg) == 0:
                continue
            if len(pos) > pairs_per_graph:
                pos = pos[:pairs_per_graph]
            t_pos = torch.from_numpy(pos).long().to(device)
            t_neg = torch.from_numpy(neg).long().to(device)

            opt.zero_grad()
            Z = model(t_X, t_Ahat)
            pos_scores = (Z[t_pos[:, 0]] * Z[t_pos[:, 1]]).sum(dim=1)
            neg_scores = (Z[t_neg[:, 0]] * Z[t_neg[:, 1]]).sum(dim=1)
            loss = bce(pos_scores, torch.ones_like(pos_scores)) + bce(neg_scores, torch.zeros_like(neg_scores))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if ep % 20 == 0 or ep == 1:
            print(f"Epoch {ep:04d}/{epochs} | loss={total_loss:.4f}")
    save_model(model, out_path, in_dim=in_dim, hidden_dim=hidden, emb_dim=emb)
    print(f"Saved trained GNN to: {out_path}")


def color_with_trained_gnn(noncomm_graph: nx.Graph, model_path: str, upper_bound_k: int, rng, device='cpu'):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available: cannot run GNN inference.")
    model, cfg = load_model(model_path, device=device)
    t_X, t_Ahat = build_inputs_from_graph(noncomm_graph, device=device)
    with torch.no_grad():
        Z = model(t_X, t_Ahat).cpu().numpy()
    n = noncomm_graph.number_of_nodes()
    if n == 0:
        return {}, 0, 0.0
    best, best_k = None, None
    for k in range(2, max(2, upper_bound_k) + 1):
        labels = _kmeans_numpy(Z, k=k, rng=rng, n_init=5, max_iter=100)
        prelim = {i: int(labels[i]) for i in range(n)}
        refined = refine_by_greedy_within_clusters(prelim, noncomm_graph)
        if coloring_is_valid(refined, noncomm_graph):
            best = refined
            best_k = len(set(refined.values()))
            break
    if best is None:
        prelim = greedy_color(noncomm_graph, strategy="saturation_largest_first")
        best = prelim
        best_k = len(set(prelim.values()))
    return relabel_coloring_sequential(best), best_k, 0.0

def _single_site_ops(d: int):
    I = np.eye(d, dtype=complex)
    return [I] + generate_su_d_basis(d)


def _build_local_sets(nq: int, nt: int):
    qubit_ops = _single_site_ops(2)
    qutrit_ops = _single_site_ops(3)
    local_sets = [qubit_ops for _ in range(nq)] + [qutrit_ops for _ in range(nt)]
    return local_sets


def _tensor_from_indices(local_sets, idxs):
    op = local_sets[0][idxs[0]]
    for s, i in zip(local_sets[1:], idxs[1:]):
        op = np.kron(op, s[i])
    return op


def _sample_unique_ops(local_sets, M: int, rng: np.random.Generator):
    arities = [len(s) for s in local_sets]
    seen = set()
    ops, labels = [], []
    max_trials = M * 50
    trials = 0
    while len(ops) < M and trials < max_trials:
        idxs = tuple(int(rng.integers(0, a)) for a in arities)
        if all(i == 0 for i in idxs):  # exclude global identity
            trials += 1; continue
        if idxs in seen:
            trials += 1; continue
        seen.add(idxs)
        op = _tensor_from_indices(local_sets, idxs)
        ops.append(op)
        nq = sum(1 for s in local_sets if s[0].shape[0] == 2)
        parts = []
        for p, i in enumerate(idxs[:nq]):
            parts.append(f"q{p}:{i}")
        for p, i in enumerate(idxs[nq:]):
            parts.append(f"t{p}:{i}")
        labels.append("|".join(parts))
        trials += 1
    if len(ops) < M:
        raise RuntimeError(f"Could only sample {len(ops)} unique operators (requested M={M}). Try reducing M.")
    return ops, labels

def _parse_methods(methods):
    if methods is None: return {'dsatur','rlf','spectral','gnn','ilp'}
    if isinstance(methods, str):
        tokens = [t.strip().lower() for t in methods.split(',') if t.strip()]
    else:
        tokens = [str(t).lower().strip() for t in methods]
    valid = {'dsatur','rlf','spectral','gnn','ilp','all'}
    for t in tokens:
        if t not in valid:
            raise ValueError(f"Unknown method '{t}'. Valid: dsatur, rlf, spectral, gnn, ilp, all")
    if not tokens or 'all' in tokens:
        return {'dsatur','rlf','spectral','gnn','ilp'}
    return set(tokens)


def run_benchmark(d=3, N=1, M=8, seed=2, methods='all',
                  gnn_epochs=150, gnn_hidden=32, gnn_emb=16, gnn_lr=1e-2,
                  gnn_model_path=None, verbose=True):
    rng = np.random.default_rng(seed)
    selected = _parse_methods(methods)

    need_noncomm = bool(selected & {'dsatur','rlf','spectral','gnn'})
    need_comm = ('ilp' in selected)

    identity = np.eye(d, dtype=complex)
    su_d_basis = generate_su_d_basis(d)
    single_site_ops = [identity] + su_d_basis
    ops, labels = tensor_product_operators(single_site_ops, N)

    dim = d ** N
    I_global = np.eye(dim)
    ops_filtered = [op for op in ops if not np.allclose(op, I_global)]

    if M > len(ops_filtered):
        raise ValueError(f"Cannot select M={M}; only {len(ops_filtered)} available after filtering.")

    idx = rng.choice(len(ops_filtered), size=M, replace=False)
    ops_sel = [ops_filtered[i] for i in idx]

    graphs_info = {'comm_edges':0,'noncomm_edges':0,'comm_build_time':0.0,'noncomm_build_time':0.0}
    noncomm_graph = comm_graph = None

    if need_noncomm:
        start = time(); noncomm_graph = generate_non_commutativity_graph(ops_sel)
        graphs_info['noncomm_build_time'] = time() - start
        graphs_info['noncomm_edges'] = noncomm_graph.number_of_edges()
    if need_comm:
        start = time(); comm_graph = generate_commutativity_graph(ops_sel)
        graphs_info['comm_build_time'] = time() - start
        graphs_info['comm_edges'] = comm_graph.number_of_edges()

    if verbose:
        print(f"Operators: M={M}, d={d}, N={N}")
        if need_noncomm:
            print(f"Non-comm edges={graphs_info['noncomm_edges']}  build={graphs_info['noncomm_build_time']:.6f}s")
        if need_comm:
            print(f"Comm     edges={graphs_info['comm_edges']}      build={graphs_info['comm_build_time']:.6f}s")

    methods_out = {}
    ds_k = None

    if 'dsatur' in selected and need_noncomm:
        t0 = time(); ds_col, ds_k = dsatur_color(noncomm_graph)
        methods_out['DSATUR'] = {'colors': ds_k, 'time': time()-t0}
    if 'rlf' in selected and need_noncomm:
        t0 = time(); rlf_col, rlf_k = rlf_color(noncomm_graph)
        methods_out['RLF'] = {'colors': rlf_k, 'time': time()-t0}
    if 'spectral' in selected and need_noncomm:
        base_ub = max(2, noncomm_graph.number_of_nodes())
        ub = ds_k if (ds_k is not None and ds_k > 0) else base_ub
        spec_col, spec_k, spec_t = spectral_coloring(noncomm_graph, upper_bound_k=ub, rng=rng)
        methods_out['Spectral'] = {'colors': spec_k, 'time': spec_t}
    if 'gnn' in selected and need_noncomm:
        base_ub = max(2, noncomm_graph.number_of_nodes())
        ub = ds_k if (ds_k is not None and ds_k > 0) else base_ub
        if gnn_model_path:
            try:
                gnn_col, gnn_k, gnn_t = color_with_trained_gnn(noncomm_graph, model_path=gnn_model_path, upper_bound_k=ub, rng=rng)
                methods_out['GNN'] = {'colors': gnn_k, 'time': gnn_t, 'backend': 'pretrained'}
            except Exception as e:
                print(f"[WARN] pretrained GNN failed ({e}); falling back to per-instance training.")
                gnn_col, gnn_k, gnn_t, used = gnn_coloring(noncomm_graph, upper_bound_k=ub, rng=rng,
                                                          epochs=gnn_epochs, hidden_dim=gnn_hidden,
                                                          emb_dim=gnn_emb, lr=gnn_lr, verbose=False)
                methods_out['GNN'] = {'colors': gnn_k, 'time': gnn_t, 'backend': 'torch' if used else 'spectral_fallback'}
        else:
            gnn_col, gnn_k, gnn_t, used = gnn_coloring(noncomm_graph, upper_bound_k=ub, rng=rng,
                                                       epochs=gnn_epochs, hidden_dim=gnn_hidden,
                                                       emb_dim=gnn_emb, lr=gnn_lr, verbose=False)
            methods_out['GNN'] = {'colors': gnn_k, 'time': gnn_t, 'backend': 'torch' if used else 'spectral_fallback'}
    if 'ilp' in selected and need_comm:
        ilp_col, ilp_t, ilp_status = solve_ilp_clique_cover(comm_graph)
        ilp_k = len(set(ilp_col.values()))
        methods_out['ILP'] = {'colors': ilp_k, 'time': ilp_t, 'status': ilp_status}

    if verbose:
        print("\n=== Results ===")
        for name in ['DSATUR','RLF','Spectral','GNN','ILP']:
            if name in methods_out:
                line = f"{name:8s} colors={methods_out[name].get('colors')} time={methods_out[name].get('time'):.4f}s"
                if name == 'GNN' and 'backend' in methods_out[name]:
                    line += f" backend={methods_out[name]['backend']}"
                if name == 'ILP' and 'status' in methods_out[name]:
                    line += f" status={methods_out[name]['status']}"
                print(line)

    return {
        'meta': {'d': d, 'N': N, 'M': M, 'seed': seed, 'methods': sorted(list(selected))},
        'graphs': graphs_info,
        'methods': methods_out,
    }


def hetero_run_benchmark(nq=1, nt=1, M=20, seed=42, methods='all',
                         gnn_epochs=150, gnn_hidden=32, gnn_emb=16, gnn_lr=1e-2,
                         gnn_model_path=None, verbose=True):
    rng = np.random.default_rng(seed)
    selected = _parse_methods(methods)
    need_noncomm = bool(selected & {'dsatur','rlf','spectral','gnn'})
    need_comm = ('ilp' in selected)

    local_sets = _build_local_sets(nq, nt)
    ops_sel, labels_sel = _sample_unique_ops(local_sets, M, rng)

    graphs_info = {'comm_edges':0, 'noncomm_edges':0, 'comm_build_time':0.0, 'noncomm_build_time':0.0,
                   'n_qubits': nq, 'n_qutrits': nt, 'hilbert_dim': int((2**nq)*(3**nt))}
    noncomm_graph = comm_graph = None
    if need_noncomm:
        start = time(); noncomm_graph = generate_non_commutativity_graph(ops_sel)
        graphs_info['noncomm_build_time'] = time() - start
        graphs_info['noncomm_edges'] = noncomm_graph.number_of_edges()
    if need_comm:
        start = time(); comm_graph = generate_commutativity_graph(ops_sel)
        graphs_info['comm_build_time'] = time() - start
        graphs_info['comm_edges'] = comm_graph.number_of_edges()

    if verbose:
        print(f"Hetero system: nq={nq}, nt={nt}, dim={graphs_info['hilbert_dim']}, M={M}")
        if need_noncomm:
            print(f"Non-comm edges={graphs_info['noncomm_edges']}  build={graphs_info['noncomm_build_time']:.6f}s")
        if need_comm:
            print(f"Comm     edges={graphs_info['comm_edges']}      build={graphs_info['comm_build_time']:.6f}s")

    methods_out = {}
    ds_k = None
    if 'dsatur' in selected and need_noncomm:
        t0 = time(); ds_col, ds_k = dsatur_color(noncomm_graph)
        methods_out['DSATUR'] = {'colors': ds_k, 'time': time()-t0}
    if 'rlf' in selected and need_noncomm:
        t0 = time(); rlf_col, rlf_k = rlf_color(noncomm_graph)
        methods_out['RLF'] = {'colors': rlf_k, 'time': time()-t0}
    if 'spectral' in selected and need_noncomm:
        base_ub = max(2, noncomm_graph.number_of_nodes())
        ub = ds_k if (ds_k is not None and ds_k > 0) else base_ub
        spec_col, spec_k, spec_t = spectral_coloring(noncomm_graph, upper_bound_k=ub, rng=rng)
        methods_out['Spectral'] = {'colors': spec_k, 'time': spec_t}
    if 'gnn' in selected and need_noncomm:
        base_ub = max(2, noncomm_graph.number_of_nodes())
        ub = ds_k if (ds_k is not None and ds_k > 0) else base_ub
        if gnn_model_path:
            try:
                gnn_col, gnn_k, gnn_t = color_with_trained_gnn(noncomm_graph, model_path=gnn_model_path, upper_bound_k=ub, rng=rng)
                methods_out['GNN'] = {'colors': gnn_k, 'time': gnn_t, 'backend': 'pretrained'}
            except Exception as e:
                print(f"[WARN] pretrained GNN failed ({e}); falling back to per-instance training.")
                gnn_col, gnn_k, gnn_t, used = gnn_coloring(noncomm_graph, upper_bound_k=ub, rng=rng,
                                                          epochs=gnn_epochs, hidden_dim=gnn_hidden,
                                                          emb_dim=gnn_emb, lr=gnn_lr, verbose=False)
                methods_out['GNN'] = {'colors': gnn_k, 'time': gnn_t, 'backend': 'torch' if used else 'spectral_fallback'}
        else:
            gnn_col, gnn_k, gnn_t, used = gnn_coloring(noncomm_graph, upper_bound_k=ub, rng=rng,
                                                       epochs=gnn_epochs, hidden_dim=gnn_hidden,
                                                       emb_dim=gnn_emb, lr=gnn_lr, verbose=False)
            methods_out['GNN'] = {'colors': gnn_k, 'time': gnn_t, 'backend': 'torch' if used else 'spectral_fallback'}
    if 'ilp' in selected and need_comm:
        ilp_col, ilp_t, ilp_status = solve_ilp_clique_cover(comm_graph)
        ilp_k = len(set(ilp_col.values()))
        methods_out['ILP'] = {'colors': ilp_k, 'time': ilp_t, 'status': ilp_status}

    if verbose:
        print("\n=== Results (hetero) ===")
        for name in ['DSATUR','RLF','Spectral','GNN','ILP']:
            if name in methods_out:
                line = f"{name:8s} colors={methods_out[name].get('colors')} time={methods_out[name].get('time'):.4f}s"
                if name == 'GNN' and 'backend' in methods_out[name]:
                    line += f" backend={methods_out[name]['backend']}"
                if name == 'ILP' and 'status' in methods_out[name]:
                    line += f" status={methods_out[name]['status']}"
                print(line)

    return {
        'meta': {'nq': nq, 'nt': nt, 'M': M, 'seed': seed, 'methods': sorted(list(selected))},
        'graphs': graphs_info,
        'methods': methods_out,
    }