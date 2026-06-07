#!/usr/bin/env python3
"""
Quantum Tomography Design Script with Combined Heuristic + ILP Partitioning

This script automates the design of quantum tomography measurement settings.
It partitions non-identity Pauli operator strings into commuting sets using several
methods:
    - DSATUR graph coloring (fast heuristic)
    - Neural network–based method (fast heuristic)
    - ILP-based method (optimal but slow for large systems)
    - Spectral clustering method
    - Recursive Largest First (RLF) graph coloring
    - Combined heuristic + ILP (balances speed and optimality)

Usage (command-line):
    python tomography_design.py <number_of_qubits> [--nn] [--ilp] [--combined] [--spectral] [--rlf]

Example:
    python tomography_design.py 2 --combined

Dependencies:
    numpy, networkx, qutip, pulp, torch, scikit-learn
"""

import itertools
import time
import random
import sys
import numpy as np
import networkx as nx

from qutip import Qobj, sigmax, sigmay, sigmaz, qeye, tensor

# ILP optimization
import pulp

# Neural network dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F

# Spectral clustering
from sklearn.cluster import KMeans

#############################################
# Helper Functions for Pauli Operators
#############################################

def generate_pauli_strings(n_qubits):
    """
    Generate all non-identity Pauli operator strings for a given number of qubits.
    Each operator is represented as a tuple of integers:
        0 -> I (Identity)
        1 -> X
        2 -> Y
        3 -> Z
    Returns:
        ops: List of tuples representing the operator.
        labels: List of corresponding string labels.
    """
    pauli_map = ['I', 'X', 'Y', 'Z']
    single_qubit_ops = [0, 1, 2, 3]
    ops, labels = [], []
    for combo in itertools.product(single_qubit_ops, repeat=n_qubits):
        if any(p != 0 for p in combo):
            ops.append(combo)
            labels.append("".join(pauli_map[p] for p in combo))
    return ops, labels

def commute_pauli_strings(p1, p2):
    """
    Check if two Pauli operator strings commute.
    They commute if the number of positions where both operators are non-identity
    and differ is even.
    """
    anticommute_count = sum(
        1 for a, b in zip(p1, p2) if a != 0 and b != 0 and a != b
    )
    return anticommute_count % 2 == 0

def build_commutation_graph(ops):
    """
    Build a graph where each node represents a Pauli operator string.
    An edge is added between two nodes if their operators do NOT commute.
    """
    G = nx.Graph()
    N = len(ops)
    G.add_nodes_from(range(N))
    print("Building commutation graph...")
    for i in range(N):
        for j in range(i + 1, N):
            if not commute_pauli_strings(ops[i], ops[j]):
                G.add_edge(i, j)
    print("Commutation graph built.")
    return G

def Qobj_pauli_from_tuple(pauli_tuple):
    """
    Convert a tuple representation of a Pauli operator into a Qobj operator.
    """
    pauli_map = [qeye(2), sigmax(), sigmay(), sigmaz()]
    ops = [pauli_map[p] for p in pauli_tuple]
    return tensor(ops)

#############################################
# Graph Coloring Heuristics: DSATUR and RLF
#############################################

def dsatur_coloring(G, random_seed=None):
    """
    DSATUR graph coloring heuristic.
    Returns a dictionary mapping node -> color.
    """
    if random_seed is not None:
        random.seed(random_seed)

    coloring = {}
    saturation = {node: 0 for node in G.nodes()}
    degrees = {node: G.degree(node) for node in G.nodes()}
    # Start with the highest-degree node.
    nodes_sorted = sorted(G.nodes(), key=lambda x: (-degrees[x], x))
    current = nodes_sorted[0]
    coloring[current] = 0

    # Update saturation for neighbors.
    for neighbor in G.neighbors(current):
        saturation[neighbor] = 1

    uncolored = set(G.nodes()) - {current}
    while uncolored:
        # Choose node with highest saturation.
        max_sat = max(saturation[node] for node in uncolored)
        candidates = [node for node in uncolored if saturation[node] == max_sat]
        node = random.choice(candidates) if len(candidates) > 1 else candidates[0]
        # Assign the smallest available color.
        neighbor_colors = {coloring.get(neigh) for neigh in G.neighbors(node) if neigh in coloring}
        color = 0
        while color in neighbor_colors:
            color += 1
        coloring[node] = color
        # Update saturation for uncolored neighbors.
        for neighbor in G.neighbors(node):
            if neighbor in uncolored:
                neighbor_used_colors = {coloring.get(n) for n in G.neighbors(neighbor) if n in coloring}
                saturation[neighbor] = len(neighbor_used_colors)
        uncolored.remove(node)
    return coloring

def find_best_dsatur_coloring(G, attempts=10, fixed_seed=None):
    """
    Run DSATUR multiple times and return the coloring with the fewest colors.
    """
    best_coloring, best_num = None, float('inf')
    for attempt in range(attempts):
        seed = fixed_seed + attempt if fixed_seed is not None else random.randint(0, 1000000)
        coloring = dsatur_coloring(G, random_seed=seed)
        num_colors = max(coloring.values()) + 1
        if num_colors < best_num:
            best_num = num_colors
            best_coloring = coloring
    return best_coloring

def find_dsatur_commuting_partition(G, attempts=10, fixed_seed=None):
    """
    Partition nodes into commuting sets using DSATUR coloring.
    Returns a dict mapping color index to list of operator indices.
    """
    coloring = find_best_dsatur_coloring(G, attempts=attempts, fixed_seed=fixed_seed)
    partition = {}
    for node, color in coloring.items():
        partition.setdefault(color, []).append(node)
    return partition

def rlf_coloring(G, random_seed=None):
    """
    Recursive Largest First (RLF) graph coloring algorithm.
    Returns a dictionary mapping each node to a color.
    """
    if random_seed is not None:
        random.seed(random_seed)
    coloring = {}
    color = 0
    uncolored = set(G.nodes())
    while uncolored:
        S = set(uncolored)  # Candidate vertices
        independent_set = set()  # Current color class
        working_set = set(S)
        while True:
            if not independent_set:
                # Pick vertex with maximum degree in S.
                subG = G.subgraph(S)
                degrees = {u: subG.degree(u) for u in S}
                max_deg = max(degrees.values())
                candidates = [u for u, d in degrees.items() if d == max_deg]
                v = random.choice(candidates)
            else:
                # Choose vertex in working_set not adjacent to any in independent_set.
                T = {u for u in working_set if all(not G.has_edge(u, w) for w in independent_set)}
                if not T:
                    break
                subG = G.subgraph(S)
                degrees = {u: subG.degree(u) for u in T}
                max_deg = max(degrees.values())
                candidates = [u for u, d in degrees.items() if d == max_deg]
                v = random.choice(candidates)
            independent_set.add(v)
            S.remove(v)
            working_set.discard(v)
            # Remove neighbors of v from working_set.
            working_set.difference_update(set(G.neighbors(v)))
        # Assign the current color to all vertices in the independent set.
        for v in independent_set:
            coloring[v] = color
        color += 1
        uncolored.difference_update(independent_set)
    return coloring

def find_rlf_commuting_partition(G, attempts=10, fixed_seed=None):
    """
    Run RLF coloring several times and return the partition corresponding to
    the best (fewest colors) coloring found.
    Returns a dict mapping color index to list of operator indices.
    """
    best_coloring, best_num = None, float('inf')
    for attempt in range(attempts):
        seed = fixed_seed + attempt if fixed_seed is not None else random.randint(0, 1000000)
        coloring = rlf_coloring(G, random_seed=seed)
        num_colors = max(coloring.values()) + 1
        if num_colors < best_num:
            best_num = num_colors
            best_coloring = coloring
    partition = {}
    for node, col in best_coloring.items():
        partition.setdefault(col, []).append(node)
    return partition

#############################################
# ILP-Based Partitioning Functions
#############################################

def optimal_partition_ilp(ops, G):
    """
    Full ILP formulation for optimal partitioning.
    May be slow for larger problems.
    Returns a list of groups (each group is a list of operator indices).
    """
    N = len(ops)
    M = N  # Worst-case: each operator in its own setting.
    prob = pulp.LpProblem("OptimalMeasurementSettings", pulp.LpMinimize)

    # Decision variables: x[i, j] indicates operator i is assigned to setting j.
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(N) for j in range(M)), cat="Binary")
    # y[j] indicates that setting j is used.
    y = pulp.LpVariable.dicts("y", (j for j in range(M)), cat="Binary")

    # Each operator must be assigned exactly one setting.
    for i in range(N):
        prob += pulp.lpSum(x[i, j] for j in range(M)) == 1

    # If an operator is assigned to setting j, then setting j is used.
    for i in range(N):
        for j in range(M):
            prob += x[i, j] <= y[j]

    # Noncommuting operators cannot share a setting.
    for j in range(M):
        for i in range(N):
            for k in range(i + 1, N):
                if not commute_pauli_strings(ops[i], ops[k]):
                    prob += x[i, j] + x[k, j] <= 1

    # Objective: minimize the number of settings used.
    prob += pulp.lpSum(y[j] for j in range(M))
    prob.solve()
    print("Full ILP Status:", pulp.LpStatus[prob.status])

    # Extract partition.
    partition = []
    for j in range(M):
        group = [i for i in range(N)
                 if pulp.value(x[i, j]) is not None and pulp.value(x[i, j]) > 0.5]
        if group:
            partition.append(group)
    return partition

def ilp_feasible_coloring(ops, G, K):
    """
    ILP feasibility check for K-coloring.
    Returns a tuple (feasible, partition), where partition maps color index to operator indices.
    """
    N = len(ops)
    prob = pulp.LpProblem("FeasibleColoring", pulp.LpMinimize)

    # Binary variables: x[i, c] = 1 if operator i gets color c.
    x = pulp.LpVariable.dicts("x", ((i, c) for i in range(N) for c in range(K)), cat="Binary")

    # Each operator must be assigned exactly one color.
    for i in range(N):
        prob += pulp.lpSum(x[i, c] for c in range(K)) == 1

    # Noncommuting operators cannot share the same color.
    for c in range(K):
        for i, j in G.edges():
            prob += x[i, c] + x[j, c] <= 1

    prob.setObjective(pulp.lpSum([]))  # Trivial objective.
    result = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    feasible = (pulp.LpStatus[prob.status] == "Optimal")
    partition = {}
    if feasible:
        for c in range(K):
            partition[c] = [i for i in range(N)
                            if pulp.value(x[i, c]) is not None and pulp.value(x[i, c]) > 0.5]
    return feasible, partition

def combined_optimal_partition(ops, G, heuristic_method="dsatur", fixed_seed=250):
    """
    Combined heuristic + ILP partitioning.
    Uses a heuristic (default DSATUR) to obtain an upper bound on the number of settings,
    then performs binary search over K (number of settings) using ILP feasibility.
    Returns a list of groups (each group is a list of operator indices).
    """
    n_qubits = len(ops[0]) if ops else 1
    expected_settings = (4**n_qubits - 1) // (2**n_qubits - 1)

    if heuristic_method == "nn":
        heuristic_partition = neural_network_coloring(G, ops, n_qubits,
                                                      num_epochs=100, lr=0.01, lambda_reg=0.1)
        upper_bound = len(heuristic_partition)
    else:
        dsatur_partition = find_dsatur_commuting_partition(G, attempts=10, fixed_seed=fixed_seed)
        heuristic_partition = list(dsatur_partition.values())
        upper_bound = len(heuristic_partition)

    lower_bound = max(largest_clique_size(G), expected_settings)
    print(f"Combined ILP: theoretical lower bound = {expected_settings}, "
          f"largest clique = {largest_clique_size(G)}, "
          f"using lower bound = {lower_bound}, and upper bound = {upper_bound}")

    best_K = upper_bound
    best_partition = heuristic_partition

    # Binary search for smallest feasible K.
    left, right = lower_bound, upper_bound
    while left <= right:
        mid = (left + right) // 2
        print(f"Trying K = {mid} colors...")
        feasible, partition = ilp_feasible_coloring(ops, G, mid)
        if feasible:
            best_K = mid
            best_partition = [grp for grp in partition.values() if grp]
            right = mid - 1
            print(f"Feasible with {mid} colors.")
        else:
            left = mid + 1
            print(f"Not feasible with {mid} colors.")
    print(f"Optimal number of settings found: {best_K}")
    return best_partition

def largest_clique_size(G):
    """
    Return the size of the largest clique in the graph.
    """
    cliques = list(nx.find_cliques(G))
    return max((len(clique) for clique in cliques), default=1)

#############################################
# Neural Network (GNN) for Graph Coloring
#############################################

def pauli_to_feature(pauli_tuple):
    """
    Convert a Pauli operator tuple to a one-hot encoded feature vector.
    """
    feature = []
    for p in pauli_tuple:
        one_hot = [0, 0, 0, 0]
        one_hot[p] = 1
        feature.extend(one_hot)
    return feature

def generate_features(ops):
    """
    Generate a feature matrix (each row is the one-hot encoding of an operator).
    """
    features = [pauli_to_feature(op) for op in ops]
    return np.array(features, dtype=np.float32)

class GCNLayer(nn.Module):
    """
    A simple Graph Convolutional Layer.
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A_hat):
        support = self.linear(X)
        out = torch.matmul(A_hat, support)
        return F.relu(out)

class GNNColoringModel(nn.Module):
    """
    Two-layer GNN that outputs logits for a fixed number of colors.
    """
    def __init__(self, in_features, hidden_features, num_colors):
        super(GNNColoringModel, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_features)
        self.gcn2 = GCNLayer(hidden_features, hidden_features)
        self.fc = nn.Linear(hidden_features, num_colors)

    def forward(self, X, A_hat):
        x = self.gcn1(X, A_hat)
        x = self.gcn2(x, A_hat)
        logits = self.fc(x)
        return logits

def refine_nn_partition(ops, partition):
    """
    Refine a partition so that each group contains only mutually commuting operators.
    Accepts partition as a dict or list of lists.
    """
    groups = list(partition.values()) if isinstance(partition, dict) else partition
    changed = True
    while changed:
        changed = False
        new_groups = []
        for group in groups:
            valid_group, leftovers = [], []
            for node in group:
                if all(commute_pauli_strings(ops[node], ops[other]) for other in valid_group):
                    valid_group.append(node)
                else:
                    leftovers.append(node)
            new_groups.append(valid_group)
            for node in leftovers:
                assigned = False
                for grp in new_groups:
                    if all(commute_pauli_strings(ops[node], ops[other]) for other in grp):
                        grp.append(node)
                        assigned = True
                        break
                if not assigned:
                    new_groups.append([node])
                    changed = True
        groups = new_groups
    return groups

def neural_network_coloring(G, ops, n_qubits, num_epochs=200, lr=0.01,
                            num_colors=None, max_graph_size=500, lambda_reg=0.1):
    """
    GNN-based graph coloring. If the graph is large, it partitions it into chunks.
    Returns a partition (list of groups of operator indices) refined so that each group is commuting.
    """
    N = len(ops)
    # If graph too large, process in chunks.
    if N > max_graph_size:
        print(f"Graph has {N} nodes (>{max_graph_size}). Partitioning into chunks...")
        node_list = list(G.nodes())
        chunks = [node_list[i:i+max_graph_size] for i in range(0, N, max_graph_size)]
        partitions = []
        for chunk in chunks:
            subG = G.subgraph(chunk).copy()
            # FIX: Re-label subgraph nodes to consecutive integers.
            subG = nx.convert_node_labels_to_integers(subG)
            sub_ops = [ops[i] for i in chunk]
            part_chunk = neural_network_coloring(subG, sub_ops, n_qubits,
                                                 num_epochs=num_epochs, lr=lr,
                                                 num_colors=num_colors, max_graph_size=max_graph_size,
                                                 lambda_reg=lambda_reg)
            # Map back to original indices.
            for group in part_chunk:
                original_group = [chunk[i] for i in group]
                partitions.append(original_group)
        return partitions

    # Prepare features and adjacency.
    features = generate_features(ops)  # Shape: (N, n_qubits * 4)
    N, in_features = features.shape
    if num_colors is None:
        expected_settings = (4**n_qubits - 1) // (2**n_qubits - 1)
        num_colors = expected_settings * 2

    # Build normalized adjacency.
    A = nx.adjacency_matrix(G).todense()
    A = np.array(A, dtype=np.float32)
    A += np.eye(N, dtype=np.float32)
    D = np.sum(A, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt

    # Convert to torch tensors.
    X = torch.tensor(features)
    A_hat_t = torch.tensor(A_hat)
    hidden_features = 32
    model = GNNColoringModel(in_features, hidden_features, num_colors)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    edges = list(G.edges())

    # Train the model.
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X, A_hat_t)
        P = F.softmax(logits, dim=1)
        # Conflict loss: penalize adjacent nodes sharing probability mass.
        conflict_loss = sum(torch.sum(P[i] * P[j]) for i, j in edges) / len(edges)
        # Regularization to encourage balanced color usage.
        usage = P.mean(dim=0)
        usage_entropy = -torch.sum(usage * torch.log(usage + 1e-10))
        loss = conflict_loss + lambda_reg * usage_entropy
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"[NN Coloring] Epoch {epoch}, Conflict Loss: {conflict_loss.item():.4f}, "
                  f"Usage Entropy: {usage_entropy.item():.4f}, Total Loss: {loss.item():.4f}")

    # Get color assignments.
    model.eval()
    with torch.no_grad():
        logits = model(X, A_hat_t)
        predicted_colors = torch.argmax(logits, dim=1).numpy().tolist()
    partition = {}
    for node, color in enumerate(predicted_colors):
        partition.setdefault(color, []).append(node)
    refined_partition = refine_nn_partition(ops, partition)
    return refined_partition

#############################################
# Spectral Clustering for Graph Coloring
#############################################

def spectral_coloring(G, ops, n_qubits, num_clusters=None):
    """
    Spectral clustering–based coloring.
    Steps:
      1. Compute the complement graph (nodes connected if they commute).
      2. Form the normalized Laplacian.
      3. Compute eigenvectors corresponding to smallest eigenvalues.
      4. Cluster rows of eigenvector matrix using KMeans.
      5. Refine clusters so that each group is mutually commuting.
    Returns a refined partition (list of groups of operator indices).
    """
    # Complement graph: connection indicates commutation.
    G_compl = nx.complement(G)
    A = nx.adjacency_matrix(G_compl).todense()
    A = np.array(A, dtype=np.float32)
    D = np.diag(np.array(A.sum(axis=1)).flatten())
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    # Determine number of clusters.
    if num_clusters is None:
        expected_settings = (4**n_qubits - 1) // (2**n_qubits - 1)
        dsatur_col = find_best_dsatur_coloring(G, attempts=10)
        dsatur_num = max(dsatur_col.values()) + 1
        num_clusters = min(expected_settings, dsatur_num)
        num_clusters = max(num_clusters, 1)

    eigenvals, eigenvecs = np.linalg.eigh(L)
    X = eigenvecs[:, :num_clusters]
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(X)

    partition = {}
    for idx, label in enumerate(labels):
        partition.setdefault(label, []).append(idx)

    refined_partition = refine_nn_partition(ops, partition)
    return refined_partition

#############################################
# Simultaneous Diagonalization Helpers
#############################################

def unique_eigvals(eigvals, tol=1e-12):
    """
    Group eigenvalues that are identical within a tolerance.
    Returns a list of groups.
    """
    sorted_vals = np.sort(eigvals)
    groups = []
    used = np.zeros(len(sorted_vals), dtype=bool)
    for i, val in enumerate(sorted_vals):
        if used[i]:
            continue
        group = [val]
        used[i] = True
        for j in range(i + 1, len(sorted_vals)):
            if not used[j] and abs(sorted_vals[j] - val) < tol:
                used[j] = True
                group.append(sorted_vals[j])
        groups.append(group)
    return groups

def refine_basis(op, basis_vecs):
    """
    Diagonalize operator op within a given subspace defined by basis_vecs.
    Returns eigenvalues and the refined basis.
    """
    proj_op = basis_vecs.conj().T @ op.full() @ basis_vecs
    w, U = np.linalg.eigh(proj_op)
    new_basis = basis_vecs @ U
    return w, new_basis

def simultaneous_eig_general(operators):
    """
    Simultaneously diagonalize a set of commuting operators.
    Returns a matrix whose columns form the final basis.
    """
    if not operators:
        raise ValueError("No operators provided for simultaneous diagonalization.")
    eigvals, eigstates = operators[0].eigenstates()
    eigvals = np.array(eigvals)
    V = np.column_stack([st.full().flatten() for st in eigstates])
    Q, _ = np.linalg.qr(V)
    final_basis = Q
    final_eigvals = eigvals
    for op in operators[1:]:
        new_vectors = []
        new_eigvals = []
        for group in unique_eigvals(final_eigvals):
            indices = [i for i, val in enumerate(final_eigvals)
                       if any(abs(val - x) < 1e-12 for x in group)]
            subspace = final_basis[:, indices]
            if subspace.shape[1] > 1:
                w_sub, refined = refine_basis(op, subspace)
                new_vectors.append(refined)
                new_eigvals.extend(w_sub)
            else:
                new_vectors.append(subspace)
                new_eigvals.append(final_eigvals[indices[0]])
        final_basis = np.hstack(new_vectors)
        Q_final, _ = np.linalg.qr(final_basis)
        final_basis = Q_final
        final_eigvals = np.array(new_eigvals)
    return final_basis

def verify_commuting_sets(ops, commuting_sets):
    """
    Verify that all operators within each set commute.
    Returns True if valid; prints an error message and returns False otherwise.
    """
    for s in commuting_sets:
        for i, j in itertools.combinations(s, 2):
            if not commute_pauli_strings(ops[i], ops[j]):
                op_i = "".join("IXYZ"[p] for p in ops[i])
                op_j = "".join("IXYZ"[p] for p in ops[j])
                print(f"Operators {i} ({op_i}) and {j} ({op_j}) do not commute!")
                return False
    return True

def optimize_commuting_sets(ops, commuting_sets):
    """
    Greedily merge compatible sets to further reduce the number of measurement settings.
    Returns the optimized list of commuting sets.
    """
    sorted_sets = sorted(commuting_sets, key=lambda s: len(s))
    optimized_sets = sorted_sets.copy()
    for small_set in sorted_sets:
        if not small_set:
            continue
        for op in small_set.copy():
            for target_set in optimized_sets:
                if target_set is small_set:
                    continue
                if all(commute_pauli_strings(ops[op], ops[j]) for j in target_set):
                    target_set.append(op)
                    small_set.remove(op)
                    break
    return [s for s in optimized_sets if s]

#############################################
# Main Tomography Design Function
#############################################

def run_tomography_design(n_qubits, dsatur_attempts=25, fixed_seed=250,
                          use_nn=False, use_ilp=False, use_combined=False,
                          use_spectral=False, use_rlf=False):
    """
    Execute the tomography design process.
    Chooses partitioning method based on flags:
      - use_combined: combined heuristic + ILP approach.
      - use_ilp: full ILP method.
      - use_nn: neural network method.
      - use_spectral: spectral clustering method.
      - use_rlf: Recursive Largest First method.
      - Otherwise, defaults to DSATUR.
    Returns:
      optimized_commuting_sets: List of commuting sets (each is a list of operator indices).
      A: Sensory matrix constructed from measurement bases.
    """
    start_time = time.time()
    random.seed(fixed_seed)
    np.random.seed(fixed_seed)

    # Generate Pauli operators.
    ops, labels = generate_pauli_strings(n_qubits)
    N = len(ops)
    expected_settings = (4**n_qubits - 1) // (2**n_qubits - 1)
    print(f"\nNumber of non-identity Pauli operators for {n_qubits} qubits: {N}")
    print(f"Expected optimal number of measurement settings: {expected_settings}")

    # Build the commutation graph.
    G = build_commutation_graph(ops)

    # Select partitioning method.
    if use_combined:
        print("\nApplying combined heuristic + ILP partitioning...")
        initial_partition = combined_optimal_partition(ops, G, heuristic_method="dsatur", fixed_seed=fixed_seed)
    elif use_ilp:
        print("\nApplying full ILP-based optimal partitioning...")
        initial_partition = optimal_partition_ilp(ops, G)
    elif use_nn:
        print("\nApplying Neural Network based coloring...")
        initial_partition = neural_network_coloring(G, ops, n_qubits, max_graph_size=500, lambda_reg=0.1)
        # Ensure partition is a list of groups.
        if isinstance(initial_partition, dict) or (initial_partition and isinstance(initial_partition[0], int)):
            initial_partition = [initial_partition]
    elif use_spectral:
        print("\nApplying Spectral Clustering based coloring...")
        initial_partition = spectral_coloring(G, ops, n_qubits)
    elif use_rlf:
        print("\nApplying Recursive Largest First (RLF) graph coloring...")
        rlf_color_map = find_rlf_commuting_partition(G, attempts=dsatur_attempts, fixed_seed=fixed_seed)
        initial_partition = list(rlf_color_map.values())
    else:
        print("\nApplying DSATUR graph coloring...")
        dsatur_color_map = find_dsatur_commuting_partition(G, attempts=dsatur_attempts, fixed_seed=fixed_seed)
        initial_partition = list(dsatur_color_map.values())

    current_partition = initial_partition

    # Ensure all operators are assigned.
    all_measured = set(itertools.chain.from_iterable(current_partition))
    missing_ops = set(range(N)) - all_measured
    if missing_ops:
        print(f"\nWarning: {len(missing_ops)} operators missing from initial settings.")
        for op_idx in list(missing_ops):
            for group in current_partition:
                if all(commute_pauli_strings(ops[op_idx], ops[j]) for j in group):
                    group.append(op_idx)
                    break
            else:
                # If still unassigned, create a new group.
                current_partition.append([op_idx])
        print("All missing operators have been assigned.")
    else:
        print("\nAll operators are included in the initial measurement settings.")

    print(f"\nTotal measurement settings after initial assignment: {len(current_partition)}")
    print("Optimizing commuting sets by merging compatible sets...")
    optimized_commuting_sets = optimize_commuting_sets(ops, current_partition)
    print(f"Number of measurement settings after optimization: {len(optimized_commuting_sets)}")

    # Verify commuting property.
    print("Verifying commutativity of final commuting sets...")
    valid = verify_commuting_sets(ops, optimized_commuting_sets)
    print("Commutativity Verification:", "Passed" if valid else "Failed")

    # Construct the sensory matrix A.
    print("\nConstructing unitary matrices and sensory matrix A...")
    A_rows = []
    for idx, group in enumerate(optimized_commuting_sets):
        set_ops = [Qobj_pauli_from_tuple(ops[i]) for i in group]
        final_basis = simultaneous_eig_general(set_ops)
        U = Qobj(final_basis, dims=[[2] * n_qubits, [2] * n_qubits])
        # Check unitarity.
        check_identity = (U.dag() * U - qeye([2] * n_qubits)).norm() < 1e-12
        print(f"Measurement Setting {idx + 1} Unitarity Check:", "Passed" if check_identity else "Failed")
        print("-" * 50)
        # Construct rows from projector vectors.
        for i in range(final_basis.shape[1]):
            psi = final_basis[:, i]
            P = np.outer(psi, psi.conj())
            A_rows.append(P.flatten(order='F').conj())
    A = np.array(A_rows, dtype=complex)
    print("\nSensory matrix A dimensions:", A.shape)

    # Compute SVD of A.
    print("Computing Singular Value Decomposition (SVD) for A...")
    try:
        U_svd, s_vals, Vh_svd = np.linalg.svd(A, full_matrices=False)
        tol = max(A.shape) * np.amax(s_vals) * np.finfo(s_vals.dtype).eps
        rank_A = np.sum(s_vals > tol)
        cond_A = s_vals[0] / s_vals[-1] if s_vals[-1] > tol else np.inf
        print("SVD computed successfully.")
    except np.linalg.LinAlgError as e:
        print("SVD computation failed:", e)
        rank_A = "Undefined"
        cond_A = "Undefined"
    print("Rank of A:", rank_A)
    print("Condition number of A:", cond_A)

    total_time = time.time() - start_time
    print(f"\nTotal computation time: {total_time:.2f} seconds")
    return optimized_commuting_sets, A

#############################################
# Main Entry Point
#############################################

def main_entry():
    """
    Entry point for the tomography design script.
    Accepts command-line arguments to select the partitioning method.
    Flags:
      --nn       : Use Neural Network method.
      --ilp      : Use full ILP method.
      --combined : Use combined heuristic + ILP method.
      --spectral : Use Spectral Clustering method.
      --rlf      : Use Recursive Largest First method.
    """
    # Interactive mode (e.g., in a shell) or command-line mode.
    if hasattr(sys, 'ps1'):
        try:
            n_qubits = int(input("Enter the number of qubits: "))
            if n_qubits < 1:
                raise ValueError
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
            return
        method = input("Choose method - (d)SATUR, (n) Neural Network, (i) ILP, (c) Combined, (s) Spectral, or (r) RLF: ").strip().lower()
        use_nn = (method == 'n')
        use_ilp = (method == 'i')
        use_combined = (method == 'c')
        use_spectral = (method == 's')
        use_rlf = (method == 'r')
    else:
        if len(sys.argv) < 2:
            print("Usage: python tomography_design.py <number_of_qubits> [--nn] [--ilp] [--combined] [--spectral] [--rlf]")
            sys.exit(1)
        try:
            n_qubits = int(sys.argv[1])
            if n_qubits < 1:
                raise ValueError
        except ValueError:
            print("Invalid input. Provide a positive integer for the number of qubits.")
            sys.exit(1)
        use_nn = '--nn' in sys.argv[2:]
        use_ilp = '--ilp' in sys.argv[2:]
        use_combined = '--combined' in sys.argv[2:]
        use_spectral = '--spectral' in sys.argv[2:]
        use_rlf = '--rlf' in sys.argv[2:]

    optimized_commuting_sets, A = run_tomography_design(
        n_qubits,
        use_nn=use_nn,
        use_ilp=use_ilp,
        use_combined=use_combined,
        use_spectral=use_spectral,
        use_rlf=use_rlf
    )

    print("\n=== Final Results ===")
    print("Optimized Commuting Sets:")
    for idx, group in enumerate(optimized_commuting_sets, start=1):
        print(f"  Set {idx}: {group}")
    print("\nSensory Matrix A shape:", A.shape)
    # Optionally, print the sensory matrix:
    # print("Sensory Matrix A:\n", A)

if __name__ == "__main__":
    main_entry()
