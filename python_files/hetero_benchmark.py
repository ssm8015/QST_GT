# ==================================
# hetero_benchmark.py
# ==================================
# Partition M operators drawn from a heterogeneous system: n_q qubits (d=2) and n_t qutrits (d=3).
# Reuses the same protocol: DSATUR, RLF, Spectral, GNN (pretrained or per-instance), ILP.

import argparse
import csv
import json
import os
from time import time
import numpy as np

from core_utils import (
    generate_su_d_basis,
    generate_non_commutativity_graph,
    generate_commutativity_graph,
)
from dsatur_rlf import dsatur_color, rlf_color
from spectral_coloring import spectral_coloring
from gnn_coloring import gnn_coloring
from gnn_infer import color_with_trained_gnn
from ilp_mcc import solve_ilp_clique_cover


def _single_site_ops(d: int):
    I = np.eye(d, dtype=complex)
    return [I] + generate_su_d_basis(d)


def _build_local_sets(nq: int, nt: int):
    qubit_ops = _single_site_ops(2)   # 4 ops
    qutrit_ops = _single_site_ops(3)  # 9 ops
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
        if all(i == 0 for i in idxs):
            trials += 1
            continue
        if idxs in seen:
            trials += 1
            continue
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


def hetero_run_benchmark(nq=1, nt=1, M=20, seed=42, gnn_epochs=200, gnn_hidden=32, gnn_emb=16, gnn_lr=1e-2, verbose=True):
    rng = np.random.default_rng(seed)

    # Methods to run
    selected = getattr(hetero_run_benchmark, '_methods_selection', None)
    if not selected or ('all' in selected):
        selected = {'dsatur', 'rlf', 'spectral', 'gnn', 'ilp'}

    need_noncomm = bool(selected & {'dsatur', 'rlf', 'spectral', 'gnn'})
    need_comm = ('ilp' in selected)

    local_sets = _build_local_sets(nq, nt)

    ops_sel, labels_sel = _sample_unique_ops(local_sets, M, rng)

    graphs_info = {
        'comm_edges': 0,
        'noncomm_edges': 0,
        'comm_build_time': 0.0,
        'noncomm_build_time': 0.0,
        'n_qubits': nq,
        'n_qutrits': nt,
        'hilbert_dim': int((2 ** nq) * (3 ** nt)),
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
        print(f"Heterogeneous system: {nq} qubits ⊗ {nt} qutrits | dim={graphs_info['hilbert_dim']}")
        print(f"Sampled M={M} operators.")
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

    # Spectral
    if 'spectral' in selected and need_noncomm:
        base_ub = max(2, noncomm_graph.number_of_nodes())
        ub = ds_k if (ds_k is not None and ds_k > 0) else base_ub
        spec_col, spec_k, spec_t = spectral_coloring(noncomm_graph, upper_bound_k=ub, rng=rng)
        methods['Spectral'] = {'colors': spec_k, 'time': spec_t}

    # GNN (pretrained or per-instance AE)
    if 'gnn' in selected and need_noncomm:
        base_ub = max(2, noncomm_graph.number_of_nodes())
        ub = ds_k if (ds_k is not None and ds_k > 0) else base_ub
        gnn_model_path = getattr(hetero_run_benchmark, '_gnn_model_path', None)
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

    # ILP
    if 'ilp' in selected and need_comm:
        ilp_col, ilp_t, ilp_status = solve_ilp_clique_cover(comm_graph)
        ilp_k = len(set(ilp_col.values()))
        methods['ILP'] = {'colors': ilp_k, 'time': ilp_t, 'status': ilp_status}

    if verbose:
        print("=== Coloring Results (hetero) ===")
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
        'meta': {'nq': nq, 'nt': nt, 'M': M, 'seed': seed, 'methods': sorted(list(selected))},
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
    print("=== Aggregate over seeds (hetero) ===")
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
    p = argparse.ArgumentParser(description="Benchmark on heterogeneous n_q qubits ⊗ n_t qutrits.")
    p.add_argument('--nq', type=int, default=1, help='Number of qubit sites (d=2).')
    p.add_argument('--nt', type=int, default=1, help='Number of qutrit sites (d=3).')
    p.add_argument('--M', type=int, default=20, help='Number of sampled operators.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--seeds', type=int, default=None, help='Run multiple seeds: seed..seed+seeds-1')
    p.add_argument('--seed-list', type=str, default=None, help='Comma-separated list of seeds, e.g. 1,7,11')
    p.add_argument('--methods', type=str, default='all', help='Comma-separated subset of methods: dsatur,rlf,spectral,gnn,ilp,all')
    p.add_argument('--out', type=str, default=None, help='Path to save results (CSV by default).')
    p.add_argument('--format', type=str, default='csv', choices=['csv', 'json'])
    p.add_argument('--no-verbose', action='store_true')

    # GNN params (per-instance training only)
    p.add_argument('--gnn-epochs', type=int, default=150)
    p.add_argument('--gnn-hidden', type=int, default=32)
    p.add_argument('--gnn-emb', type=int, default=16)
    p.add_argument('--gnn-lr', type=float, default=1e-2)

    # Pretrained model for inference-only
    p.add_argument('--gnn-model', type=str, default=None, help='Path to a pretrained GNN .pt file (from gnn_train.py).')

    args = p.parse_args()

    selected = _parse_methods_arg(args.methods)

    if args.seed_list:
        seeds = [int(s.strip()) for s in args.seed_list.split(',') if s.strip()]
    elif args.seeds is not None and args.seeds > 0:
        seeds = list(range(args.seed, args.seed + args.seeds))
    else:
        seeds = [args.seed]

    # Make model path & methods visible to hetero_run_benchmark without changing signature
    setattr(hetero_run_benchmark, '_gnn_model_path', args.gnn_model if args.gnn_model else None)
    setattr(hetero_run_benchmark, '_methods_selection', selected)

    all_rows = []
    for sd in seeds:
        res = hetero_run_benchmark(nq=args.nq, nt=args.nt, M=args.M, seed=sd,
                                   gnn_epochs=args.gnn_epochs, gnn_hidden=args.gnn_hidden,
                                   gnn_emb=args.gnn_emb, gnn_lr=args.gnn_lr,
                                   verbose=not args.no_verbose)
        all_rows.extend(_rows_from_result(res))

    if len(seeds) > 1:
        summary = _aggregate(all_rows)
        _print_summary(summary)

    if args.out:
        path = _write_out(all_rows, args.out, fmt=args.format)
        print(f"Saved per-run results to: {path}")


if __name__ == '__main__':
    main()
