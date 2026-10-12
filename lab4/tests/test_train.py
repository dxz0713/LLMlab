import numpy as np

from src.graph import add_self_loops, make_adjacency, symmetric_normalize
from src.train import TrainConfig, evaluate, train


def _two_cliques(seed=0):
    """Two 10-node cliques with cluster-correlated but noisy features.

    Class c nodes have features drawn from N(mu_c, I) with mu_0 = -mu_1;
    the signal-to-noise ratio per feature is low, but aggregating over the
    clique averages the noise away -- graph structure is what makes the
    classes separable.
    """
    n_per = 10
    edges = [[i, j] for i in range(n_per) for j in range(i + 1, n_per)]
    edges += [[n_per + i, n_per + j] for i in range(n_per) for j in range(i + 1, n_per)]
    A = make_adjacency(2 * n_per, np.array(edges))

    rng = np.random.RandomState(seed)
    dim = 16
    mu = np.ones(dim) * 0.35
    X0 = rng.randn(n_per, dim) + mu
    X1 = rng.randn(n_per, dim) - mu
    X = np.vstack([X0, X1])
    y = np.array([0] * n_per + [1] * n_per)
    A_norm = symmetric_normalize(add_self_loops(A))
    return A_norm, X, y


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.num_layers == 2
    assert cfg.hidden_dim == 16
    assert cfg.epochs == 200
    assert cfg.optimizer == "adam"
    assert cfg.weight_decay > 0


def test_evaluate_returns_loss_and_acc():
    A_norm, X, y = _two_cliques()
    from src.model import GraphNet

    net = GraphNet(num_features=16, num_classes=2, seed=0)
    mask = np.zeros(20, dtype=bool)
    mask[:5] = True
    mask[10:15] = True
    loss, acc = evaluate(net, X, A_norm, y, mask)
    assert isinstance(loss, float) and isinstance(acc, float)
    assert loss > 0 and 0.0 <= acc <= 1.0


def test_train_reduces_loss():
    A_norm, X, y = _two_cliques()
    mask = np.zeros(20, dtype=bool)
    mask[:5] = True
    mask[10:15] = True
    cfg = TrainConfig(epochs=100, lr=0.05, weight_decay=0.0, seed=0)
    _, history = train(X, A_norm, y, mask, cfg, verbose=False)
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_train_updates_every_parameter():
    A_norm, X, y = _two_cliques()
    mask = np.zeros(20, dtype=bool)
    mask[:5] = True
    mask[10:15] = True

    from src.model import GraphNet

    before = GraphNet(num_features=16, num_classes=2, seed=0)
    cfg = TrainConfig(epochs=5, lr=0.01, weight_decay=0.0, seed=0)
    net, _ = train(X, A_norm, y, mask, cfg, verbose=False)
    for name in before.params:
        assert not np.allclose(before.params[name], net.params[name]), f"{name} unchanged"


def test_history_keys_and_length():
    A_norm, X, y = _two_cliques()
    train_mask = np.zeros(20, dtype=bool)
    train_mask[:5] = True
    train_mask[10:15] = True
    val_mask = np.zeros(20, dtype=bool)
    val_mask[5:10] = True
    val_mask[15:20] = True
    cfg = TrainConfig(epochs=7, seed=0)
    _, history = train(X, A_norm, y, train_mask, cfg, val_mask=val_mask, verbose=False)
    for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
        assert key in history
        assert len(history[key]) == 7


def test_weight_decay_shrinks_weights():
    A_norm, X, y = _two_cliques()
    mask = np.zeros(20, dtype=bool)
    mask[:5] = True
    mask[10:15] = True

    cfg_no = TrainConfig(epochs=100, lr=0.05, weight_decay=0.0, seed=0)
    net_no, _ = train(X, A_norm, y, mask, cfg_no, verbose=False)

    cfg_wd = TrainConfig(epochs=100, lr=0.05, weight_decay=0.5, seed=0)
    net_wd, _ = train(X, A_norm, y, mask, cfg_wd, verbose=False)

    norm_no = np.linalg.norm(net_no.params["W1"])
    norm_wd = np.linalg.norm(net_wd.params["W1"])
    assert norm_wd < norm_no


def test_graph_structure_helps():
    """GCN beats the feature-only (identity adjacency) baseline on clustered data."""
    A_norm, X, y = _two_cliques()
    train_mask = np.zeros(20, dtype=bool)
    train_mask[:2] = True
    train_mask[10:12] = True
    test_mask = ~train_mask

    cfg = TrainConfig(epochs=200, lr=0.05, weight_decay=0.0, seed=0)
    net_gcn, _ = train(X, A_norm, y, train_mask, cfg, verbose=False)
    _, acc_gcn = evaluate(net_gcn, X, A_norm, y, test_mask)

    # identity adjacency = no message passing = MLP on node features
    A_id = np.eye(20)
    net_mlp, _ = train(X, A_id, y, train_mask, cfg, verbose=False)
    _, acc_mlp = evaluate(net_mlp, X, A_id, y, test_mask)

    assert acc_gcn >= 0.95
    assert acc_gcn > acc_mlp


def test_transductive_unlabeled_nodes_improve():
    """Nodes never seen by the loss still get classified correctly."""
    A_norm, X, y = _two_cliques()
    train_mask = np.zeros(20, dtype=bool)
    train_mask[0] = True
    train_mask[10] = True
    test_mask = ~train_mask

    from src.model import GraphNet

    net0 = GraphNet(num_features=16, num_classes=2, seed=0)
    _, acc_before = evaluate(net0, X, A_norm, y, test_mask)

    cfg = TrainConfig(epochs=300, lr=0.05, weight_decay=0.0, seed=0)
    net, _ = train(X, A_norm, y, train_mask, cfg, verbose=False)
    _, acc_after = evaluate(net, X, A_norm, y, test_mask)

    assert acc_after >= 0.95
    assert acc_after > acc_before
