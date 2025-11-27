# =============================
# dsatur_rlf.py
# =============================

from networkx.algorithms.coloring import greedy_color


def dsatur_color(noncomm_graph):
    """DSATUR heuristic coloring on the non-commutativity graph."""
    col = greedy_color(noncomm_graph, strategy="saturation_largest_first")
    return col, len(set(col.values()))


def rlf_color(noncomm_graph):
    """RLF-like (via largest_first strategy) heuristic coloring."""
    col = greedy_color(noncomm_graph, strategy="largest_first")
    return col, len(set(col.values()))