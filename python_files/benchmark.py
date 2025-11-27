# ==================================
# benchmark.py
# ==================================

import argparse
import csv
import json
import os
from time import time
import numpy as np

from core_utils import (
    generate_su_d_basis,
    tensor_product_operators,
    generate_non_commutativity_graph,
    generate_commutativity_graph,
)
from dsatur_rlf import dsatur_color, rlf_color
from spectral_coloring import spectral_coloring
# Per-instance unsupervised GNN (autoencoder) fallback
from gnn_coloring import gnn_coloring
# Pretrained inference-only GNN
from gnn_infer import color_with_trained_gnn
from ilp_mcc import solve_ilp_clique_cover


def run_benchmark(d=5, N=1, M=20, seed=42, gnn_epochs=200, gnn_hidden=32, gnn_emb=16, gnn_lr=1e-2, verbose=True):
    rng = np.random.default_rng(seed)

    # Methods to run
    selected = getattr(run_benchmark, '_methods_selection', None)
    if not selected or ('all' in selected):
        selected = {'dsatur', 'rlf', 'spectral', 'gnn', 'ilp'}

    need_noncomm = bool(selected & {'dsatur', 'rlf', 'spectral', 'gnn'})
    need_comm = ('ilp' in selected)

    # Include identity + SU(d) generators
    identity = np.eye(d, dtype=complex)
    su_d_basis = generate_su_d_basis(d)
    single_site_ops = [identity] + su_d_basis  # d^2 ops per site

    ops, labels = tensor_product_operators(single_site_ops, N)

    # Remove global identity only
    dim = d ** N
    I_global = np.eye(dim)
    ops_filtered = [op for op in ops if not np.allclose(op, I_global)]

    if verbose:
        print(f"Generating operators for {N}-qudit system (d = {d})...")
        print(f"Total number of SU({d})^{N} operators (excluding identity): {len(ops_filtered)}")

    if M > len(ops_filtered):
        raise ValueError(f"Cannot select {M} operators; only {len(ops_filtered)} available after filtering.")

    idx = rng.choice(len(ops_filtered), size=M, replace=False)
    ops_sel = [ops_filtered[i] for i in idx]

    graphs_info = {
        'comm_edges': 0,
        'noncomm_edges': 0,
        'comm_build_time': 0.0,
        'noncomm_build_time': 0.0,
    }

    # Build graphs conditionally
    noncomm_graph = None
    comm_graph = None

    if need_noncomm:
        start = time()
        noncomm_graph = generate_non_commutativity_graph(ops_sel)
        graphs_info['noncomm_build_time'] = time() - start
        graphs_info['noncomm_edges'] = noncomm_graph.number_of_edges()

    if need_comm:
        start = time()
        comm_graph = generate_commutativity_graph(ops_sel)
        graphs_info['comm_build_time'] = time() - start
        graphs_info['comm_edges'] = comm_graph.number_of_edges()

    if verbose:
        if need_comm:
            print(f"Edges in commutativity graph: {graphs_info['comm_edges']}")
            print(f"Time to build commutativity graph: {graphs_info['comm_build_time']:.6f} s")
        if need_noncomm:
            print(f"Edges in non-commutativity graph: {graphs_info['noncomm_edges']}")
            print(f"Time to build non-commutativity graph: {graphs_info['noncomm_build_time']:.6f} s")

    methods = {}

    ds_k = None

    # DSATUR
    if 'dsatur' in selected and need_noncomm:
        t0 = time()
        ds_col, ds_k = dsatur_color(noncomm_graph)
        methods['DSATUR'] = {'colors': ds_k, 'time': time() - t0}

    # RLF (largest_first)
    if 'rlf' in selected and need_noncomm:
        t0 = time()
        rlf_col, rlf_k = rlf_color(noncomm_graph)
        methods['RLF'] = {'colors': rlf_k, 'time': time() - t0}

    # Spectral (upper bound from DSATUR if available)
    if 'spectral' in selected and need_noncomm:
        base_ub = max(2, noncomm_graph.number_of_nodes())
        ub = ds_k if (ds_k is not None and ds_k > 0) else base_ub
        spec_col, spec_k, spec_t = spectral_coloring(noncomm_graph, upper_bound_k=ub, rng=rng)
        methods['Spectral'] = {'colors': spec_k, 'time': spec_t}

    # GNN
    if 'gnn' in selected and need_noncomm:
        base_ub = max(2, noncomm_graph.number_of_nodes())
        ub = ds_k if (ds_k is not None and ds_k > 0) else base_ub
        gnn_model_path = getattr(run_benchmark, '_gnn_model_path', None)
        if gnn_model_path:
            try:
                gnn_col, gnn_k, gnn_t = color_with_trained_gnn(noncomm_graph,
                                                              model_path=gnn_model_path,
                                                              upper_bound_k=ub, rng=rng)
                methods['GNN'] = {'colors': gnn_k, 'time': gnn_t, 'backend': 'pretrained'}
            except Exception as e:
                print(f"[WARN] GNN inference failed ({e}); falling back to per-instance training.")
                gnn_col, gnn_k, gnn_t, used_gnn = gnn_coloring(noncomm_graph, upper_bound_k=ub, rng=rng,
                                                               epochs=gnn_epochs, hidden_dim=gnn_hidden,
                                                               emb_dim=gnn_emb, lr=gnn_lr, verbose=False)
                methods['GNN'] = {'colors': gnn_k, 'time': gnn_t, 'backend': 'torch' if used_gnn else 'spectral_fallback'}
        else:
            gnn_col, gnn_k, gnn_t, used_gnn = gnn_coloring(noncomm_graph, upper_bound_k=ub, rng=rng,
                                                           epochs=gnn_epochs, hidden_dim=gnn_hidden,
                                                           emb_dim=gnn_emb, lr=gnn_lr, verbose=False)
            methods['GNN'] = {'colors': gnn_k, 'time': gnn_t, 'backend': 'torch' if used_gnn else 'spectral_fallback'}

    # ILP (min clique cover on commutativity graph)
    if 'ilp' in selected and need_comm:
        ilp_col, ilp_t, ilp_status = solve_ilp_clique_cover(comm_graph)
        ilp_k = len(set(ilp_col.values()))
        methods['ILP'] = {'colors': ilp_k, 'time': ilp_t, 'status': ilp_status}

    if verbose:
        print("=== Coloring Results ===")
        order = ['DSATUR', 'RLF', 'Spectral', 'GNN', 'ILP']
        for name in order:
            if name in methods:
                line = f"{name}: colors={methods[name].get('colors')} time={methods[name].get('time'):.4f} s"
                if name == 'GNN' and 'backend' in methods[name]:
                    line += f" backend={methods[name]['backend']}"
                if name == 'ILP' and 'status' in methods[name]:
                    line += f" status={methods[name]['status']}"
                print(line)

    result = {
        'meta': {'d': d, 'N': N, 'M': M, 'seed': seed, 'methods': sorted(list(selected))},
        'graphs': graphs_info,
        'methods': methods,
    }
    return result


def _rows_from_result(res):
    base = {**res['meta'], **res['graphs']}
    rows = []
    for m, info in res['methods'].items():
        row = {'method': m, **base}
        row['colors'] = info.get('colors')
        row['time'] = info.get('time')
        if m == 'ILP':
            row['status'] = info.get('status')
        if m == 'GNN':
            row['backend'] = info.get('backend')
        rows.append(row)
    return rows


def _write_out(rows, out_path, fmt='csv'):
    fmt = (fmt or 'csv').lower()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    if fmt == 'json':
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2)
        return out_path
    # CSV default
    keys = set()
    for r in rows:
        keys.update(r.keys())
    header = sorted(keys)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out_path


def _aggregate(rows):
    from collections import defaultdict
    acc = defaultdict(lambda: {'colors': [], 'time': []})
    for r in rows:
        acc[r['method']]['colors'].append(r['colors'])
        acc[r['method']]['time'].append(r['time'])
    summary = {}
    for m, vals in acc.items():
        c = np.array(vals['colors'], dtype=float)
        t = np.array(vals['time'], dtype=float)
        summary[m] = {
            'colors_mean': float(np.nanmean(c)) if len(c) else float('nan'),
            'colors_std': float(np.nanstd(c, ddof=0)) if len(c) else float('nan'),
            'time_mean_s': float(np.nanmean(t)) if len(t) else float('nan'),
            'time_std_s': float(np.nanstd(t, ddof=0)) if len(t) else float('nan'),
            'runs': int(len(c)),
        }
    return summary


def _print_summary(summary):
    print("=== Aggregate over seeds ===")
    print(f"{'Method':<10} {'colors(mean±std)':>20} {'time s (mean±std)':>22}  runs")
    for m, s in summary.items():
        print(f"{m:<10} {s['colors_mean']:.2f}±{s['colors_std']:.2f:>10} {s['time_mean_s']:.4f}±{s['time_std_s']:.4f:>12}  {s['runs']:>3}")


def _parse_methods_arg(methods_str: str):
    tokens = [t.strip().lower() for t in (methods_str or 'all').split(',') if t.strip()]
    valid = {'dsatur', 'rlf', 'spectral', 'gnn', 'ilp', 'all'}
    for t in tokens:
        if t not in valid:
            raise ValueError(f"Unknown method '{t}'. Valid: dsatur, rlf, spectral, gnn, ilp, all")
    if 'all' in tokens or not tokens:
        return {'dsatur', 'rlf', 'spectral', 'gnn', 'ilp'}
    return set(tokens)


def main():
    p = argparse.ArgumentParser(description="Benchmark coloring/partitioning methods on qudit operator graphs.")
    p.add_argument('--d', type=int, default=3, help='Local Hilbert space dimension (e.g., 2=qubit, 3=qutrit).')
    p.add_argument('--N', type=int, default=1, help='Number of particles (sites).')
    p.add_argument('--M', type=int, default=8, help='Number of randomly selected operators.')
    p.add_argument('--seed', type=int, default=2, help='Random seed (used if --seeds or --seed-list not provided).')
    p.add_argument('--seeds', type=int, default=None, help='Run multiple seeds: run seeds seed..seed+seeds-1')
    p.add_argument('--seed-list', type=str, default=None, help='Comma-separated list of seeds, e.g. 1,7,11')
    p.add_argument('--methods', type=str, default='all', help='Comma-separated subset of methods: dsatur,rlf,spectral,gnn,ilp,all')
    p.add_argument('--out', type=str, default=None, help='Path to save results (CSV by default).')
    p.add_argument('--format', type=str, default='csv', choices=['csv', 'json'], help='Output format for --out.')
    p.add_argument('--no-verbose', action='store_true', help='Suppress per-run prints.')

    # GNN params (used ONLY for per-instance training)
    p.add_argument('--gnn-epochs', type=int, default=150)
    p.add_argument('--gnn-hidden', type=int, default=32)
    p.add_argument('--gnn-emb', type=int, default=16)
    p.add_argument('--gnn-lr', type=float, default=1e-2)

    # Pretrained model path (if provided, we will use inference-only mode)
    p.add_argument('--gnn-model', type=str, default=None, help='Path to a pretrained GNN .pt file (from gnn_train.py).')

    args = p.parse_args()

    selected = _parse_methods_arg(args.methods)

    seeds = []
    if args.seed_list:
        seeds = [int(s.strip()) for s in args.seed_list.split(',') if s.strip()]
    elif args.seeds is not None and args.seeds > 0:
        seeds = list(range(args.seed, args.seed + args.seeds))
    else:
        seeds = [args.seed]

    # Stash model path & methods into the function so run_benchmark can see them without changing signature
    setattr(run_benchmark, '_gnn_model_path', args.gnn_model if args.gnn_model else None)
    setattr(run_benchmark, '_methods_selection', selected)

    all_rows = []
    for sd in seeds:
        res = run_benchmark(d=args.d, N=args.N, M=args.M, seed=sd,
                            gnn_epochs=args.gnn_epochs, gnn_hidden=args.gnn_hidden,
                            gnn_emb=args.gnn_emb, gnn_lr=args.gnn_lr,
                            verbose=not args.no_verbose)
        all_rows.extend(_rows_from_result(res))

    # Aggregate across seeds
    if len(seeds) > 1:
        summary = _aggregate(all_rows)
        _print_summary(summary)

    # Optional write-out
    if args.out:
        path = _write_out(all_rows, args.out, fmt=args.format)
        print(f"Saved per-run results to: {path}")


if __name__ == "__main__":
    main()

