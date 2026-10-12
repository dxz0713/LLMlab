"""Draw the Cora citation graph, colored by predicted class.

Reads outputs/test_predictions.npz (produced by train_cora.py) and the
cached dataset, keeps the largest connected component, and lays it out with
a spring embedding. Nodes are colored by their predicted class; nodes whose
prediction disagrees with the true label are highlighted with a black ring.

Usage:
    uv run python train_cora.py        # first, to produce predictions
    uv run python visualize_graph.py
"""

from __future__ import annotations

import os

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.data import CORA_CLASSES, load_cora


def main(predictions_path="outputs/test_predictions.npz", data_root="data",
         out_path="outputs/cora_graph.png", seed=0):
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(
            f"{predictions_path} not found -- run `uv run python train_cora.py` first."
        )

    with np.load(predictions_path) as z:
        y_true, y_pred, test_mask = z["y_true"], z["y_pred"], z["test_mask"]
    _, _, edges = load_cora(data_root)

    # largest connected component only, so the figure is not dominated by
    # dozens of tiny disconnected clusters
    G = nx.Graph()
    G.add_edges_from(edges.tolist())
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()
    print(f"largest connected component: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pos = nx.spring_layout(G, seed=seed, k=0.05, iterations=60)

    nodes = np.array(sorted(G.nodes()))
    colors = y_pred[nodes]
    wrong = (y_pred[nodes] != y_true[nodes]) & test_mask[nodes]

    matplotlib.use("Agg")
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.08, width=0.5)
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodes[~wrong].tolist(), node_color=colors[~wrong],
        cmap="tab10", vmin=0, vmax=9, node_size=18, linewidths=0,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=nodes[wrong].tolist(), node_color=colors[wrong],
        cmap="tab10", vmin=0, vmax=9, node_size=32, linewidths=1.2,
        edgecolors="black",
    )
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                   markerfacecolor=plt.get_cmap("tab10")(c), markeredgecolor="none",
                   label=name)
        for c, name in enumerate(CORA_CLASSES)
    ]
    handles.append(
        plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                   markerfacecolor="white", markeredgecolor="black",
                   label="misclassified (test)")
    )
    plt.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)
    plt.title("Cora citation network, colored by predicted class")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"graph figure saved to {out_path}")


if __name__ == "__main__":
    main()
