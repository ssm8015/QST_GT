# ==================================
# gnn_model.py
# ==================================

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ---- GNN architecture ---- #
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X, A_hat):
        return A_hat @ self.lin(X)


class GAE(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=32, emb_dim=16):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, emb_dim)

    def forward(self, X, A_hat):
        Z = self.gcn1(X, A_hat)
        Z = F.relu(Z)
        Z = self.gcn2(Z, A_hat)
        return Z


# ---- Graph → tensors utilities ---- #
def adjacency_matrix(graph):
    n = graph.number_of_nodes()
    A = np.zeros((n, n), dtype=float)
    for u, v in graph.edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


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
    X = np.hstack([deg / np.maximum(1.0, deg.max()), ones])
    return X.astype(np.float32)


def build_inputs_from_graph(graph, device='cpu'):
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for GNN training/inference.")
    A = adjacency_matrix(graph)
    A_hat = normalized_adj_with_selfloops(A).astype(np.float32)
    X = make_node_features(A)
    t_X = torch.from_numpy(X).to(device)
    t_Ahat = torch.from_numpy(A_hat).to(device)
    return t_X, t_Ahat


# ---- Save/Load helpers ---- #

def save_model(model, path: str, in_dim: int, hidden_dim: int, emb_dim: int):
    payload = {
        'state_dict': model.state_dict(),
        'config': {'in_dim': in_dim, 'hidden_dim': hidden_dim, 'emb_dim': emb_dim},
    }
    import torch
    torch.save(payload, path)


def load_model(path: str, device='cpu'):
    import torch
    payload = torch.load(path, map_location=device)
    cfg = payload['config']
    model = GAE(in_dim=cfg['in_dim'], hidden_dim=cfg['hidden_dim'], emb_dim=cfg['emb_dim'])
    model.load_state_dict(payload['state_dict'])
    model.to(device)
    model.eval()
    return model, cfg