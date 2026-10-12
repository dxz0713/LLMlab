from __future__ import annotations

import os
import tarfile
import urllib.request

import numpy as np

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"

CORA_CLASSES = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]

NUM_FEATURES = 1433


def _parse_cora(content_path, cites_path):
    """Parse the raw Cora text files into (X, y, edges, num_nodes).

    cora.content: one paper per line, tab-separated:
        paper_id \t feature_1 ... feature_1433 \t subject_label
    cora.cites: one citation per line, tab-separated:
        cited_paper_id \t citing_paper_id

    Returns:
    - X: (N, 1433) float64 binary bag-of-words features
    - y: (N,) int64 class indices in [0, 7)
    - edges: (E, 2) int64 array of undirected edges (both directions removed,
      each unordered pair stored once)
    """
    paper_ids = []
    labels = []
    feats = []
    with open(content_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            paper_ids.append(parts[0])
            feats.append([float(v) for v in parts[1:-1]])
            labels.append(parts[-1])

    classes = sorted(set(labels))
    assert classes == CORA_CLASSES, f"unexpected class set: {classes}"
    class_to_idx = {c: i for i, c in enumerate(classes)}
    id_to_idx = {pid: i for i, pid in enumerate(paper_ids)}

    X = np.asarray(feats, dtype=np.float64)
    y = np.asarray([class_to_idx[c] for c in labels], dtype=np.int64)

    edge_set = set()
    with open(cites_path, "r", encoding="utf-8") as f:
        for line in f:
            a, b = line.strip().split("\t")[:2]
            i, j = id_to_idx[a], id_to_idx[b]
            if i != j:
                edge_set.add((min(i, j), max(i, j)))
    edges = np.asarray(sorted(edge_set), dtype=np.int64)
    return X, y, edges, len(paper_ids)


def _locate_extracted_dir(root):
    """Return (content_path, cites_path) under <root>/cora, or (None, None)."""
    candidates = [
        os.path.join(root, "cora"),
        os.path.join(root, "cora", "cora"),
    ]
    for d in candidates:
        content = os.path.join(d, "cora.content")
        cites = os.path.join(d, "cora.cites")
        if os.path.exists(content) and os.path.exists(cites):
            return content, cites
    return None, None


def load_cora(root="data"):
    """Load the Cora citation network into numpy arrays.

    On first use it looks for the extracted dataset under ``<root>/cora``;
    if missing, it downloads the archive and extracts it. The parsed arrays
    are then cached as a single ``cora.npz`` so subsequent runs start
    instantly.

    Returns:
    - X: (2708, 1433) float64 binary features in {0, 1}
    - y: (2708,) int64 labels in [0, 7)
    - edges: (E, 2) int64 undirected edges (E = 5429, each pair once)
    """
    os.makedirs(root, exist_ok=True)
    cache = os.path.join(root, "cora.npz")

    if os.path.exists(cache):
        with np.load(cache) as z:
            return z["X"], z["y"], z["edges"]

    content_path, cites_path = _locate_extracted_dir(root)

    if content_path is None:
        tgz_path = os.path.join(root, "cora.tgz")
        if not os.path.exists(tgz_path):
            print(f"downloading Cora from {CORA_URL} ...")
            urllib.request.urlretrieve(CORA_URL, tgz_path)
        extract_dir = os.path.join(root, "cora")
        print(f"extracting to {extract_dir} ...")
        with tarfile.open(tgz_path, "r:gz") as tf:
            tf.extractall(extract_dir)
        content_path, cites_path = _locate_extracted_dir(root)

    if content_path is None:
        raise FileNotFoundError(
            "Could not locate the Cora files. Expected them under "
            f"{os.path.join(root, 'cora', 'cora.content')}."
        )

    X, y, edges, num_nodes = _parse_cora(content_path, cites_path)
    assert X.shape == (num_nodes, NUM_FEATURES)

    np.savez_compressed(cache, X=X, y=y, edges=edges)
    return X, y, edges


def row_normalize_features(X):
    """Row-normalize node features so each node's feature vector sums to 1.

    Cora features are binary word indicators; dividing by the row sum turns
    them into a word-frequency distribution, which keeps nodes with many
    words from dominating the aggregation. Nodes with an all-zero feature
    vector are left as zero.
    """
    s = X.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return X / s


def make_splits(num_nodes, y, labels_per_class=20, num_val=500, num_test=1000, seed=0):
    """Construct train/val/test boolean masks for semi-supervised learning.

    The training mask contains exactly ``labels_per_class`` nodes per class
    (the classic "few labels per class" protocol for citation networks);
    ``num_val`` and ``num_test`` nodes are then sampled from the rest.
    Everything else is left unlabeled (usable neither for training nor for
    evaluation).

    Returns (train_mask, val_mask, test_mask), each a bool array of shape
    (num_nodes,).
    """
    rng = np.random.RandomState(seed)

    # Fix the val/test split first, so that experiments varying
    # labels_per_class stay directly comparable (same evaluation nodes).
    perm = rng.permutation(num_nodes)
    val_idx = perm[:num_val]
    test_idx = perm[num_val : num_val + num_test]

    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    reserved = val_mask | test_mask
    train_mask = np.zeros(num_nodes, dtype=bool)
    for c in np.unique(y):
        idx = np.where((y == c) & ~reserved)[0]
        chosen = rng.choice(idx, size=labels_per_class, replace=False)
        train_mask[chosen] = True
    return train_mask, val_mask, test_mask
