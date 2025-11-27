# ==================================
# gnn_train.py
# ==================================

import argparse
import numpy as np
from time import time
from core_utils import (
    generate_su_d_basis,
    tensor_product_operators,
    generate_non_commutativity_graph,
)
from gnn_model import (
    TORCH_AVAILABLE, GAE, build_inputs_from_graph, adjacency_matrix,
    save_model,
)

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn


def _sample_pos_neg_pairs(A: np.ndarray, rng: np.random.Generator, max_pairs: int = None):
    # Positive edges
    pos = [(u, v) if u < v else (v, u) for u in range(A.shape[0]) for v in range(u + 1, A.shape[1]) if A[u, v] == 1]
    # Negative edges
    neg = [(u, v) for u in range(A.shape[0]) for v in range(u + 1, A.shape[1]) if A[u, v] == 0]
    if max_pairs is not None:
        rng.shuffle(pos)
        rng.shuffle(neg)
        pos = pos[:max_pairs]
        neg = neg[:max_pairs]
    return np.array(pos, dtype=int), np.array(neg, dtype=int)


def _graph_from_random_ops(d, N, M, seed):
    rng = np.random.default_rng(seed)
    identity = np.eye(d, dtype=complex)
    su = generate_su_d_basis(d)
    single_site_ops = [identity] + su
    ops, _ = tensor_product_operators(single_site_ops, N)
    # remove global identity
    dim = d ** N
    I = np.eye(dim)
    ops = [op for op in ops if not np.allclose(op, I)]
    idx = rng.choice(len(ops), size=M, replace=False)
    ops_sel = [ops[i] for i in idx]
    G = generate_non_commutativity_graph(ops_sel)
    return G


def main():
    p = argparse.ArgumentParser(description="Train a graph autoencoder (GNN) once on small graphs; reuse weights later.")
    p.add_argument('--out', type=str, required=True, help='Path to save trained model (e.g., gnn_small.pt).')
    p.add_argument('--graphs', type=int, default=50, help='Number of training graphs to generate.')
    p.add_argument('--d', type=int, default=3)
    p.add_argument('--N', type=int, default=1)
    p.add_argument('--M', type=int, default=12, help='Operators per training graph (small).')
    p.add_argument('--seed0', type=int, default=0, help='Base seed; training uses seed0..seed0+graphs-1.')
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--emb', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--pairs-per-graph', type=int, default=512, help='Max pos/neg pairs per graph per epoch.')

    args = p.parse_args()

    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available: cannot train GNN.")

    device = torch.device('cpu')
    in_dim = 2  # from make_node_features
    model = GAE(in_dim=in_dim, hidden_dim=args.hidden, emb_dim=args.emb).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(args.seed0)

    # Pre-generate training graphs
    train_graphs = [_graph_from_random_ops(args.d, args.N, args.M, seed=args.seed0 + i) for i in range(args.graphs)]

    print(f"Training on {len(train_graphs)} graphs, epochs={args.epochs}, pairs/graph={args.pairs_per_graph}")
    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for G in train_graphs:
            # Build inputs
            t_X, t_Ahat = build_inputs_from_graph(G, device=device)
            # Sample pairs for this graph
            A = adjacency_matrix(G)
            pos, neg = _sample_pos_neg_pairs(A, rng, max_pairs=args.pairs_per_graph)
            if len(pos) == 0 or len(neg) == 0:
                continue
            t_pos = torch.from_numpy(pos).long().to(device)
            t_neg = torch.from_numpy(neg).long().to(device)

            # Forward & loss
            opt.zero_grad()
            Z = model(t_X, t_Ahat)
            pos_scores = (Z[t_pos[:, 0]] * Z[t_pos[:, 1]]).sum(dim=1)
            neg_scores = (Z[t_neg[:, 0]] * Z[t_neg[:, 1]]).sum(dim=1)
            loss = bce(pos_scores, torch.ones_like(pos_scores)) + bce(neg_scores, torch.zeros_like(neg_scores))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if ep % 20 == 0 or ep == 1:
            print(f"Epoch {ep:04d}/{args.epochs} | loss={total_loss:.4f}")

    # Save trained model
    save_model(model, args.out, in_dim=in_dim, hidden_dim=args.hidden, emb_dim=args.emb)
    print(f"Saved trained GNN to: {args.out}")


if __name__ == '__main__':
    main()