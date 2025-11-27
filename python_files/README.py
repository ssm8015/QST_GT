#!/usr/bin/env python3
"""
README.py — Quantum Operator Partitioning & Benchmarking Framework
==================================================================

Run this file to print a full README with a workflow pipeline diagram and
commands for every method.

    python README.py
    python README.py --section quickstart
    python README.py --list-sections

This repository implements a modular framework to:
  • Generate operator sets for homogeneous (qudits) or heterogeneous (qubits ⊗ qutrits) systems
  • Build commutativity / non-commutativity graphs
  • Partition operators via:
        - Heuristics: DSATUR, RLF
        - Spectral clustering (+ greedy refinement)
        - GNN (per-instance unsupervised AE OR pretrained inference)
        - ILP (Minimum Clique Cover on commutativity graph)
  • Benchmark methods across seeds and export results (CSV/JSON)

Folder map (files in this repo)
-------------------------------
- core_utils.py
    • generate_su_d_basis(d)
    • tensor_product_operators(single_site_ops, N)
    • generate_non_commutativity_graph(ops)
    • generate_commutativity_graph(ops)
    • coloring helpers:
        - coloring_is_valid
        - relabel_coloring_sequential
        - refine_by_greedy_within_clusters

- dsatur_rlf.py
    • dsatur_color(G_noncomm)
    • rlf_color(G_noncomm)

- spectral_coloring.py
    • spectral_coloring(G_noncomm, upper_bound_k, rng)

- ilp_mcc.py
    • solve_ilp_clique_cover(G_comm)  # Pulp + CBC

- gnn_model.py
    • GAE (GCN-based Autoencoder)
    • build_inputs_from_graph, save_model, load_model

- gnn_train.py
    • Train once on small graphs → save weights (e.g., gnn_small.pt)

- gnn_infer.py
    • color_with_trained_gnn(G_noncomm, model_path, upper_bound_k, rng)

- benchmark.py
    • Homogeneous systems (qudits): runs DSATUR, RLF, Spectral, GNN, ILP

- hetero_benchmark.py
    • Heterogeneous systems (n_q qubits ⊗ n_t qutrits): same method suite

NOTE: benchmark.py and hetero_benchmark.py refer to gnn_coloring.py for
      per-instance unsupervised training fallback. If not,
      either provide gnn_coloring.py OR always pass --gnn-model to use a
      pretrained GNN.

Dependencies
------------
    pip install numpy networkx pulp
    pip install torch

CBC (default Pulp solver) ships with PuLP. For large ILP instances consider installing a commercial solver and adapting ilp_mcc.py.

ASCII Pipeline Diagram
----------------------
The end-to-end flow (homogeneous & heterogeneous share the same backbone):

    ┌────────────────────┐
    │  System Spec       │  (d,N)   OR   (n_q, n_t)
    └─────────┬──────────┘
              │
              v
    ┌────────────────────┐
    │ Local Ops per site │  identity + su(d) basis
    └─────────┬──────────┘
              │  tensor products (exclude global I)
              v
    ┌────────────────────┐
    │  Operator Set (M)  │  sampled unique operators
    └─────────┬──────────┘
              │
              │   pairwise (anti)commutation checks
              v
    ┌──────────────────────────────────────────────┐
    │ Graphs:                                      │
    │  • Non-commutativity graph  (edges = don't commute) │
    │  •    Commutativity graph   (edges = do    commute) │
    └─────────┬────────────────────────────┬───────┘
              │                            │
    ┌─────────v─────────┐        ┌─────────v─────────┐
    │  Heuristics/GNN/  │        │        ILP         │
    │  Spectral (color) │        │  (min clique cover)│
    └─────────┬─────────┘        └─────────┬─────────┘
              │                             │
              └─────── merge results & timings ───────► Benchmark summary/CSV/JSON

Quickstart
----------
1) Train a small reusable GNN (recommended once):
    python gnn_train.py --out gnn_small.pt --graphs 100 --d 3 --N 1 --M 12 --epochs 300

2) Run homogeneous benchmark with pretrained GNN:
    python benchmark.py --d 3 --N 2 --M 30 --seed 7 --gnn-model gnn_small.pt

3) Run heterogeneous benchmark with pretrained GNN:
    python hetero_benchmark.py --nq 2 --nt 1 --M 40 --seed 7 --gnn-model gnn_small.pt

4) Export CSV across multiple seeds:
    python hetero_benchmark.py --nq 2 --nt 2 --M 25 --seed 10 --seeds 5 --out hetero_results.csv

Interpreting Outputs
--------------------
Both benchmark scripts print per-method results and write per-run rows to CSV/JSON.

Printed example Output:
    === Coloring Results ===
    DSATUR:   colors=5 time=0.0023 s
    RLF:      colors=6 time=0.0019 s
    Spectral: colors=5 time=0.0047 s
    GNN:      colors=5 time=0.0031 s backend=pretrained
    ILP:      colors=5 time=0.0451 s status=Optimal

CSV/JSON per-run fields (superset; heterogeneous includes n_qubits/n_qutrits):
    method, d, N, M, seed, backend, status,
    comm_edges, noncomm_edges, comm_build_time, noncomm_build_time

Color count = number of commuting groups (measurement settings).
Lower is better (fewer settings).

Method-by-Method: How to Run & Save Outputs
-------------------------------------------

A) Homogeneous (qudits: dimension d, sites N)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1) DSATUR / RLF / Spectral:
    python benchmark.py --methods dsatur,RLF,Spectral --d 3 --N 1 --M 20 --seed 10 --no-verbose
(They always run; --no-verbose just does not display prints.)

2) With pretrained GNN (fast inference, recommended):
    python benchmark.py --methods gnn --d 3 --N 2 --M 30 --seed 7 --gnn-model gnn_small.pt

3) Per-instance GNN (unsupervised AE) if we don't have a model:
    python benchmark.py --methods gnn --d 3 --N 1 --M 20 --seed 10 --gnn-epochs 200

4) Include ILP and export CSV:
    python benchmark.py --methods ilp --d 5 --N 1 --M 25 --seed 1 --out runs.csv --format csv

B) Heterogeneous (n_q qubits ⊗ n_t qutrits)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1) With pretrained GNN:
    python hetero_benchmark.py --methods gnn --nq 2 --nt 1 --M 40 --seed 7 --gnn-model gnn_small.pt

2) Per-instance GNN:
    python hetero_benchmark.py --methods gnn --nq 1 --nt 2 --M 30 --seed 3 --gnn-epochs 200

3) Multi-seed + CSV:
    python hetero_benchmark.py --methods all --nq 2 --nt 2 --M 25 --seed 10 --seeds 5 --out hetero_results.csv

Training a Reusable GNN Once (Small Graph Regime)
-------------------------------------------------
The intent is to train on small graphs (cheap) and reuse embeddings on larger graphs.

Command:
    python gnn_train.py --methods gnn --out gnn_small.pt --graphs 100 --d 3 --N 1 --M 12 --epochs 300

Key args:
    --graphs    Number of random training graphs
    --d, --N    Local dimension and sites used to synthesize training graphs
    --M         Operators per training graph (keep small, around 16)
    --hidden    GAE hidden size (default 64)
    --emb       Embedding dimension (default 32)
    --lr        Learning rate (default 1e-2)

Then use the saved model:
    python benchmark.py --methods gnn --d 3 --N 2 --M 30 --seed 7 --gnn-model gnn_small.pt
    python hetero_benchmark.py --methods gnn --nq 2 --nt 1 --M 40 --seed 7 --gnn-model gnn_small.pt

Spectral Coloring Details
-------------------------
- Builds normalized Laplacian with self-loops for stability
- Eigen-embedding (skip trivial eigenvector); k-means for k ≤ DSATUR upper bound
- Greedy refinement within clusters for valid colorings

ILP (Minimum Clique Cover)
--------------------------
- Runs on the commutativity graph
- Formulated as set cover over maximal cliques (NetworkX find_cliques)
- Solved with PuLP's CBC by default; reports solver status
- Optimal but can be slow; expect scaling limits as M grows

GNN Notes
---------
Two modes:
  1) Pretrained inference (recommended):
      - Fast, no training during benchmarking
      - Robust for bigger graphs than seen at train time
  2) Per-instance unsupervised AE (requires gnn_coloring.py):
      - Trains a small autoencoder directly on the instance
      - Falls back to spectral if torch unavailable

Tips & Performance
------------------
- Always exclude the global identity before building graphs.
- Use DSATUR's color count as an upper bound for spectral/GNN k-means.
- For larger instances, prefer pretrained GNN or spectral; ILP can become the bottleneck.
- Fix seeds for reproducibility: --seed, --seeds, or --seed-list.
- Use --out runs.csv to persist per-run results; combine across seeds and summarize.

Troubleshooting
---------------
• “Could only sample X unique operators”: reduce M or adjust seeds; heterogeneous sampling avoids duplicates and the global identity.  
• ILP very slow: lower N or M, or skip ILP for large runs; or plug a faster MILP solver into ilp_mcc.py.

======================================================================================================================================
"""

import argparse
import textwrap
import sys

SECTIONS = {
    "overview": ("Overview", 0, 110),
    "map": ("Folder map", 121, 521),
    "deps": ("Dependencies", 523, 647),
    "diagram": ("ASCII Pipeline Diagram", 649, 1264),
    "quickstart": ("Quickstart", 1266, 1621),
    "outputs": ("Interpreting Outputs", 1623, 2024),
    "homogeneous": ("Homogeneous usage", 2026, 2443),
    "heterogeneous": ("Heterogeneous usage", 2445, 2761),
    "train": ("Training a Reusable GNN", 2763, 3292),
    "spectral": ("Spectral Coloring Details", 3294, 3546),
    "ilp": ("ILP (Minimum Clique Cover)", 3548, 3814),
    "gnn": ("GNN Notes", 3816, 4100),
    "tips": ("Tips & Performance", 4102, 4572),
    "troubleshooting": ("Troubleshooting", 4574, 4981),
    "api": ("Minimal Programmatic API Example", 4983, 5618),
    "license": ("License", 5620, 5672),
    "maintainers": ("Maintainers", 5674, 99999),
}

def _get_doc():
    return __doc__.strip("\n")

def _print_section(name):
    doc = _get_doc()
    if name not in SECTIONS:
        raise SystemExit(f"Unknown section '{name}'. Use --list-sections to see all.")
    _, start, end = SECTIONS[name]
    print(doc.splitlines()[start:end])

def _print_all():
    print(_get_doc())

def _list_sections():
    width = 28
    print("Available sections:\n")
    for key, (title, _, _) in SECTIONS.items():
        print(f"  {key:<{width}} {title}")

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Print README sections for the Quantum Operator Partitioning & Benchmarking Framework.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--section", type=str, default=None,
                        help="Print only a specific section (use --list-sections to see keys).")
    parser.add_argument("--list-sections", action="store_true",
                        help="List section keys you can pass to --section.")
    args = parser.parse_args(argv)

    if args.list_sections:
        _list_sections()
        return 0

    if args.section:
        _print_section(args.section)
        return 0

    _print_all()
    return 0

if __name__ == "__main__":
    sys.exit(main())
