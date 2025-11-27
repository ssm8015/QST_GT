# =========================
# core_utils.py
# =========================

import numpy as np
import networkx as nx
from itertools import combinations, product
from collections import defaultdict

# ---------- Operator generation ----------

def generate_su_d_basis(d: int):
    """Generate a standard Hermitian traceless basis for su(d)."""
    basis = []
    # Off-diagonal (symmetric & antisymmetric)
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

    # Diagonal (Cartan) generators
    for i in range(1, d):
        diag = np.zeros((d, d), dtype=complex)
        for j in range(i):
            diag[j, j] = 1
        diag[i, i] = -i
        diag /= np.sqrt(i * (i + 1))
        basis.append(diag)
    return basis


def tensor_product_operators(single_site_ops, N: int):
    """Build the N-fold tensor product basis from given single-site operators.
    Returns (ops, labels). The identity should be included in single_site_ops."""
    ops, labels = [], []
    basis_size = len(single_site_ops)
    for idxs in product(range(basis_size), repeat=N):
        label = "-".join(map(str, idxs))
        op = single_site_ops[idxs[0]]
        for i in idxs[1:]:
            op = np.kron(op, single_site_ops[i])
        ops.append(op)
        labels.append(label)
    return ops, labels

# ---------- Graph construction ----------

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

# ---------- Coloring helpers shared by Spectral/GNN ----------

def coloring_is_valid(coloring: dict, graph: nx.Graph) -> bool:
    for u, v in graph.edges:
        if coloring.get(u) == coloring.get(v):
            return False
    return True


def relabel_coloring_sequential(coloring: dict):
    mapping, next_c = {}, 0
    new_col = {}
    for n in sorted(coloring.keys()):
        c = coloring[n]
        if c not in mapping:
            mapping[c] = next_c
            next_c += 1
        new_col[n] = mapping[c]
    return new_col


def refine_by_greedy_within_clusters(clusters: dict, graph: nx.Graph) -> dict:
    """Greedily color each cluster's induced subgraph and offset color ids."""
    from networkx.algorithms.coloring import greedy_color
    cluster_to_nodes = defaultdict(list)
    for n, c in clusters.items():
        cluster_to_nodes[c].append(n)

    final_coloring = {}
    color_offset = 0
    for c_id in sorted(cluster_to_nodes.keys()):
        nodes = cluster_to_nodes[c_id]
        subg = graph.subgraph(nodes)
        if subg.number_of_edges() == 0:
            for n in nodes:
                final_coloring[n] = color_offset
            color_offset += 1
        else:
            sub_col = greedy_color(subg, strategy="saturation_largest_first")
            k_sub = len(set(sub_col.values()))
            for n in nodes:
                final_coloring[n] = sub_col[n] + color_offset
            color_offset += k_sub
    return final_coloring
