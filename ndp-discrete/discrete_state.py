"""Discrete node state for NDP: replaces the evolved latent embedding with a fixed,
human-readable vector of inputs that is recomputed from the graph at every growth cycle.

Latent NDP:   node state = learned embedding, propagated over the graph and transformed by an MLP.
Discrete NDP: node state = [role, bias, weights, neighbours' weights/bias, dev state, degree],
              read straight off the graph. Nothing is propagated, nothing is co-evolved.

The 3 rules read this vector:
    growth rule  features(node)                  -> divide if > 0
    bias rule    features(node)                  -> node bias        (uses NDP's "transform" slot)
    weight rule  [features(i), features(j)]      -> weight of edge (i, j)
"""
import numpy as np
import networkx as nx
import torch

FEATURES = (
    "node_id",            # Node ID / (n - 1): breaks the tie between nodes that would otherwise look identical
    "is_input",           # role of the node (input / hidden / output)
    "is_hidden",
    "is_output",
    "bias",               # Bias
    "mean_weight",        # Weight: mean weight of the node's edges
    "nbr_mean_bias",      # Neighbors (W/B): mean bias of neighbours
    "nbr_mean_weight",    #                  mean of neighbours' mean edge weight
    "age",                # Dev state: growth cycles since the node was born / total cycles
    "degree",             # Node degree / (n - 1)
)


def feature_size():
    return len(FEATURES)


def init_graph_attributes(G):
    """Every starting node: bias 0, born at cycle 0."""
    for n in G.nodes:
        G.nodes[n]["bias"] = 0.0
        G.nodes[n]["born"] = 0
    return G


def node_biases(G):
    """Bias vector in node order, or None for graphs grown without biases (latent NDP)."""
    if len(G) == 0 or "bias" not in G.nodes[next(iter(G.nodes))]:
        return None
    return np.array([G.nodes[n]["bias"] for n in G.nodes], dtype=float)


def node_features(G, config, cycle):
    U = G.to_undirected(as_view=True) if G.is_directed() else G
    nodes = list(G.nodes)
    embryo = config.get("initial_graph") == "embryo"
    inputs = set(config["input_nodes"]) if embryo else set()
    outputs = set(config["output_nodes"]) if embryo else set()
    n_cycles = max(1, config["number_of_growth_cycles"])

    nbrs = {n: [m for m in U.neighbors(n) if m != n] for n in nodes}
    mean_w = {n: np.mean([U[n][m]["weight"] for m in nbrs[n]]) if nbrs[n] else 0.0 for n in nodes}

    X = np.zeros((len(nodes), feature_size()))
    for k, n in enumerate(nodes):
        X[k] = (
            n / max(1, len(nodes) - 1),
            n in inputs,
            n not in inputs and n not in outputs,
            n in outputs,
            G.nodes[n]["bias"],
            mean_w[n],
            np.mean([G.nodes[m]["bias"] for m in nbrs[n]]) if nbrs[n] else 0.0,
            np.mean([mean_w[m] for m in nbrs[n]]) if nbrs[n] else 0.0,
            (cycle - G.nodes[n]["born"]) / n_cycles,
            len(nbrs[n]) / max(1, len(nodes) - 1),
        )
    return X


def divide(G, divide_mask, cycle):
    """Same rule as NDP's node_based_growth: the child connects to its parent and the parent's neighbours."""
    neighbors = [list(set(nx.all_neighbors(G, n)) | {n}) for n in range(len(G))]
    new_id = len(G)
    for parent in np.flatnonzero(divide_mask):
        G.add_node(new_id, bias=0.0, born=cycle + 1)
        for nb in neighbors[parent]:
            G.add_edge(nb, new_id, weight=1.0)
            if G.is_directed():
                G.add_edge(new_id, nb, weight=1.0)
        new_id += 1
    return G


def grow_discrete(G, growth_rule, bias_rule, weight_rule, config, on_step=None):
    """Grow G for number_of_growth_cycles. on_step(G, cycle) is called after every change (for animations)."""
    G = init_graph_attributes(G.copy())
    for cycle in range(config["number_of_growth_cycles"]):
        if on_step:
            on_step(G, cycle)

        X = node_features(G, config, cycle)
        divide_mask = (growth_rule(torch.tensor(X)).detach().numpy().reshape(-1) > 0)
        if config.get("initial_graph") == "embryo":
            divide_mask[config["input_nodes"] + config["output_nodes"]] = False
        G = divide(G, divide_mask, cycle)
        if on_step:
            on_step(G, cycle)

        X = node_features(G, config, cycle + 1)
        biases = bias_rule(torch.tensor(X)).detach().numpy().reshape(-1)
        for k, n in enumerate(G.nodes):
            G.nodes[n]["bias"] = float(biases[k])
        if G.number_of_edges():
            pairs = np.array([np.concatenate([X[i], X[j]]) for i, j in G.edges()])
            weights = weight_rule(torch.tensor(pairs)).detach().numpy().reshape(-1)
            for (i, j), w in zip(G.edges(), weights):
                G[i][j]["weight"] = float(w)
        if on_step:
            on_step(G, cycle)

        if config["prunning_phase"]:
            G.remove_edges_from([(a, b) for a, b, w in G.edges(data="weight") if abs(w) <= config["prunning_threshold"]])
            if on_step:
                on_step(G, cycle)
    return G
