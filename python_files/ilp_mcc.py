# =============================
# ilp_mcc.py
# =============================

from time import time
from pulp import (
    LpProblem, LpVariable, lpSum, LpMinimize, LpBinary,
    LpStatus, value, PULP_CBC_CMD
)


def solve_ilp_clique_cover(comm_graph):
    """Minimum clique cover via set cover on maximal cliques."""
    import networkx as nx
    cliques = list(nx.find_cliques(comm_graph))
    prob = LpProblem("MinCliqueCover", LpMinimize)
    vars_ = [LpVariable(f"c{i}", cat=LpBinary) for i in range(len(cliques))]
    prob += lpSum(vars_)

    for v in comm_graph.nodes:
        prob += lpSum(vars_[i] for i, clique in enumerate(cliques) if v in clique) >= 1

    start = time()
    prob.solve(PULP_CBC_CMD(msg=0))
    elapsed = time() - start
    selected_cliques = [cliques[i] for i in range(len(cliques)) if value(vars_[i]) > 0.5]

    node_to_color = {}
    for color, clique in enumerate(selected_cliques):
        for node in clique:
            if node not in node_to_color:
                node_to_color[node] = color
    status = LpStatus[prob.status]
    return node_to_color, elapsed, status