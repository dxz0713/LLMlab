from __future__ import annotations

import numpy as np

from src.gcn import graph_conv_backward, graph_conv_forward
from src.init import kaiming
from src.layers import relu_backward, relu_forward, softmax_loss
from src.sage import sage_conv_backward, sage_conv_forward

LAYER_TYPES = ("gcn", "sage")
AGGREGATORS = ("mean", "max")


def masked_softmax_loss(scores, y, mask):
    """Softmax cross-entropy over the masked (labeled) nodes only.

    In semi-supervised node classification the labels of most nodes are
    withheld during training. The loss must therefore average only over the
    nodes selected by ``mask``, and the gradient w.r.t. the scores of every
    other node must be exactly zero -- their scores still influence the
    loss indirectly, but only through the graph propagation, not directly.

    Inputs:
    - scores: (N, C)
    - y: (N,) integer labels in [0, C); entries outside the mask are ignored
    - mask: (N,) bool, True for labeled nodes

    Returns a tuple of:
    - loss: float, averaged over the masked nodes
    - dscores: (N, C), exactly zero on rows where mask is False
    """
    scores_m = scores[mask]
    y_m = y[mask]
    loss, dscores_m = softmax_loss(scores_m, y_m)
    dscores = np.zeros_like(scores)
    dscores[mask] = dscores_m
    return loss, dscores


def masked_accuracy(scores, y, mask):
    """Fraction of correctly classified nodes among the masked ones.

    Inputs:
    - scores: (N, C)
    - y: (N,) integer labels
    - mask: (N,) bool

    Returns a float in [0, 1].
    """
    if mask.sum() == 0:
        return 0.0
    pred = np.argmax(scores[mask], axis=1)
    return np.mean(pred == y[mask])


class GraphNet:
    """A multi-layer graph neural network for node classification.

    Architecture:

        [graph-conv - relu] x (num_layers - 1)  ->  graph-conv  ->  scores

    where "graph-conv" is either a GCN layer (Kipf & Welling 2017,
    out = A_norm X W + b) or a GraphSAGE layer (Hamilton et al. 2017,
    out = concat(X, AGG(X)) W + b with a mean or max aggregator), selected
    by ``layer_type``.

    The adjacency matrix is NOT a parameter: it is computed once from the
    graph (e.g. symmetric_normalize(A + I) for GCN, row_normalize(A) for
    SAGE-mean) and passed into every call. All layers see the same matrix,
    so with L layers each node's score depends on its L-hop neighbourhood.

    The constructor only allocates parameters. Params are stored as a plain
    dict of numpy arrays (W0/b0/.../W{L-1}/b{L-1}) so that any optimizer
    from optim.py can be applied directly:

        for name, p in net.params.items():
            net.params[name], _ = adam(p, grads[name], configs[name])
    """

    def __init__(self, num_features, num_classes, hidden_dim=16, num_layers=2,
                 layer_type="gcn", aggregator="mean", seed=0):
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if layer_type not in LAYER_TYPES:
            raise ValueError(f"layer_type must be one of {LAYER_TYPES}")
        if aggregator not in AGGREGATORS:
            raise ValueError(f"aggregator must be one of {AGGREGATORS}")
        rng = np.random.default_rng(seed)
        self.num_layers = num_layers
        self.layer_type = layer_type
        self.aggregator = aggregator
        self.params = {}
        self.grads = {}

        dims = [num_features] + [hidden_dim] * (num_layers - 1) + [num_classes]
        for i in range(num_layers):
            fan_in = dims[i] if layer_type == "gcn" else 2 * dims[i]
            self.params[f"W{i}"] = kaiming(fan_in, dims[i + 1], rng)
            self.params[f"b{i}"] = np.zeros(dims[i + 1])

    # ------------------------------------------------------------------ layers

    def _conv(self, X, A_norm, idx):
        """One graph convolution (GCN or SAGE, per self.layer_type)."""
        W, b = self.params[f"W{idx}"], self.params[f"b{idx}"]
        if self.layer_type == "gcn":
            return graph_conv_forward(A_norm, X, W, b)
        elif self.layer_type == "sage":
            return sage_conv_forward(A_norm, X, W, b, self.aggregator)

    def _conv_backward(self, dout, cache):
        if self.layer_type == "gcn":
            return graph_conv_backward(dout, cache)
        elif self.layer_type == "sage":
            return sage_conv_backward(dout, cache)

    def _conv_relu(self, X, A_norm, idx):
        """One graph-conv-relu block. Returns (out, (conv_cache, relu_cache))."""
        out, conv_cache = self._conv(X, A_norm, idx)
        out, relu_cache = relu_forward(out)
        return out, (conv_cache, relu_cache)

    def _conv_relu_backward(self, dout, caches):
        """Backward of one graph-conv-relu block. Returns (dX, dW, db)."""
        conv_cache, relu_cache = caches
        dconv = relu_backward(dout, relu_cache)
        return self._conv_backward(dconv, conv_cache)
    # ------------------------------------------------------------ fwd / bwd

    def loss(self, X, A_norm, y=None, mask=None):
        """Forward pass, and backward pass if labels are given.

        Inputs:
        - X: (N, num_features) node features
        - A_norm: (N, N) adjacency prepared for this layer type (see the
          class docstring), computed once up front
        - y: (N,) int labels; if None only scores are returned
        - mask: (N,) bool selecting the labeled nodes the loss averages over

        Returns:
        - scores (N, num_classes) if y is None
        - (loss, grads) otherwise, grads keyed like params
        """
        cache = []
        H = X
        for idx in range(self.num_layers - 1):
            H, c = self._conv_relu(H, A_norm, idx)
            cache.append(c)

        last_idx = self.num_layers - 1
        scores, last_cache = self._conv(H, A_norm, last_idx)

        if y is None:
            return scores

        loss, dscores = masked_softmax_loss(scores, y, mask)
        grads = self._backward(dscores, cache, last_cache)
        return loss, grads

    def _backward(self, dscores, caches, last_cache):
        """Backward pass through the entire network.

        Returns a dict mapping every parameter name to its gradient,
        keyed like self.params.
        """
        grads = {}
        dH, dW, db = self._conv_backward(dscores, last_cache)
        grads[f"W{self.num_layers - 1}"] = dW
        grads[f"b{self.num_layers - 1}"] = db

        for idx in reversed(range(self.num_layers - 1)):
            dH, dW, db = self._conv_relu_backward(dH, caches[idx])
            grads[f"W{idx}"] = dW
            grads[f"b{idx}"] = db

        return grads

    def predict(self, X, A_norm):
        """Predicted class label of every node, shape (N,)."""
        scores = self.loss(X, A_norm)
        return np.argmax(scores, axis=1)
