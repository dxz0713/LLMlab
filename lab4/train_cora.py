"""Train a from-scratch GNN on the Cora citation network (Task 5 run script).

Usage:
    uv run python train_cora.py                          # default: 2-layer GCN
    uv run python train_cora.py --layer-type sage        # GraphSAGE (mean)
    uv run python train_cora.py --norm row --epochs 300  # tweak hyperparameters

The script ends by:
  1. printing overall and per-class accuracy on the 1000-node test set;
  2. saving the training curves (loss / accuracy) to PNG files;
  3. saving test predictions to outputs/test_predictions.npz;
  4. (with --smoothness) printing the Dirichlet energy of propagated
     features for k = 0..8 hops, a direct view of over-smoothing.
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from src.data import CORA_CLASSES, load_cora, make_splits, row_normalize_features
from src.graph import (
    add_self_loops,
    dirichlet_energy,
    make_adjacency,
    propagate,
    row_normalize,
    symmetric_normalize,
)
from src.train import TrainConfig, evaluate, train


def parse_args():
    p = argparse.ArgumentParser(description="Train a GNN on Cora from scratch")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--layer-type", choices=["gcn", "sage"], default="gcn")
    p.add_argument("--aggregator", choices=["mean", "max"], default="mean",
                   help="SAGE aggregator (ignored for GCN)")
    p.add_argument("--norm", choices=["sym", "row", "none"], default="sym",
                   help="adjacency normalization for GCN layers")
    p.add_argument("--no-self-loops", action="store_true",
                   help="use A instead of A + I (GCN only)")
    p.add_argument("--identity-adj", action="store_true",
                   help="replace the graph by I: no message passing, i.e. a plain MLP baseline")
    p.add_argument("--labels-per-class", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--optimizer", choices=["sgd", "sgd_momentum", "adam"], default="adam")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-root", type=str, default="data")
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--smoothness", action="store_true",
                   help="print Dirichlet energy of A_norm^k X for k = 0..8 (over-smoothing)")
    return p.parse_args()


def build_adjacency(edges, num_nodes, args):
    """Prepare the (normalized) adjacency matrix according to the CLI flags."""
    if args.identity_adj:
        return np.eye(num_nodes)

    A = make_adjacency(num_nodes, edges)

    if args.layer_type == "sage":
        # SAGE keeps a dedicated self path via concat, so NO self loops here.
        return row_normalize(A) if args.aggregator == "mean" else A

    # GCN
    A_tilde = A if args.no_self_loops else add_self_loops(A)
    if args.norm == "sym":
        return symmetric_normalize(A_tilde)
    if args.norm == "row":
        return row_normalize(A_tilde)
    return A_tilde


def to_efficient_format(A_norm, args):
    """Convert the adjacency to scipy sparse where it pays off.

    Cora has 2708 nodes but only ~13k stored entries, so a sparse @ dense
    product is orders of magnitude faster than a dense GEMM. Because the
    layers in src/ are written purely with @ and .T, the very same code
    runs on csr matrices with no modification -- the matrix-formulation
    dividend.

    Exception: the max aggregator indexes rows of A directly, so SAGE-max
    keeps the dense matrix.
    """
    if args.identity_adj:
        return sp.eye(A_norm.shape[0], format="csr")
    if args.layer_type == "sage" and args.aggregator == "max":
        return A_norm
    return sp.csr_matrix(A_norm)


def _save_curves(history, out_dir):
    matplotlib.use("Agg")
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_loss"], marker="o", markersize=2, label="train loss")
    if "val_loss" in history:
        plt.plot(epochs, history["val_loss"], marker="s", markersize=2, label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"loss curve saved to {loss_path}")

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, history["train_acc"], marker="o", markersize=2, label="train acc")
    if "val_acc" in history:
        plt.plot(epochs, history["val_acc"], marker="s", markersize=2, label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    acc_path = os.path.join(out_dir, "accuracy_curve.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"accuracy curve saved to {acc_path}")


def _report_test(net, X, A_norm, y, test_mask, out_dir):
    test_loss, acc = evaluate(net, X, A_norm, y, test_mask)
    print("=" * 60)
    print(f"test loss = {test_loss:.4f}, test accuracy = {acc:.4f} ({acc * 100:.2f}%)")
    print("-" * 60)
    print("per-class accuracy:")
    preds = net.predict(X, A_norm)
    for c, name in enumerate(CORA_CLASSES):
        mask = test_mask & (y == c)
        c_acc = float((preds[mask] == c).mean())
        print(f"  {name:>22s}: {c_acc * 100:5.2f}%  ({mask.sum():4d} nodes)")
    print("=" * 60)

    np.savez(
        os.path.join(out_dir, "test_predictions.npz"),
        y_true=y,
        y_pred=preds,
        test_mask=test_mask,
        classes=np.array(CORA_CLASSES),
    )
    print(f"predictions saved to {os.path.join(out_dir, 'test_predictions.npz')}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X, y, edges = load_cora(args.data_root)
    num_nodes = X.shape[0]
    X = row_normalize_features(X)
    A_norm = to_efficient_format(build_adjacency(edges, num_nodes, args), args)

    train_mask, val_mask, test_mask = make_splits(
        num_nodes, y, labels_per_class=args.labels_per_class, seed=args.seed
    )
    print(f"nodes: {num_nodes}, edges: {edges.shape[0]}, features: {X.shape[1]}")
    print(f"labeled splits: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    print("class names:", CORA_CLASSES)

    cfg = TrainConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        layer_type=args.layer_type,
        aggregator=args.aggregator,
        epochs=args.epochs,
        lr=args.lr,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    print(f"config: {cfg}")

    start_time = time.time()
    net, history = train(X, A_norm, y, train_mask, cfg, val_mask=val_mask)
    total_time = time.time() - start_time
    print(f"Training {args.epochs} epochs completed, total time: {total_time:.2f}s")

    _save_curves(history, args.out_dir)
    _report_test(net, X, A_norm, y, test_mask, args.out_dir)

    if args.smoothness:
        print("Dirichlet energy of A_norm^k @ X (over-smoothing check):")
        A_raw = make_adjacency(num_nodes, edges)
        for k in range(9):
            print(f"  k={k}: {dirichlet_energy(propagate(A_norm, X, k), A_raw):.4f}")

    return net, history


if __name__ == "__main__":
    main()
