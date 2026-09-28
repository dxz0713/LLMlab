from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.rnn_layers import rnn_backward, rnn_forward
from src.temporal_layers import temporal_affine_backward, temporal_affine_forward, temporal_softmax_loss

Array = NDArray[np.float64]
IntArray = NDArray[np.int64]


class AdderRNN:
    """A vanilla RNN that reads two numbers bit by bit and writes their sum.

    Architecture, applied at every time step t with the same weights:

        x_t (2 bits)  ->  h_t = tanh(x_t Wx + h_{t-1} Wh + b)  ->  scores_t = h_t Wy + by

    scores_t holds 2 class scores: "bit t of the sum is 0" and "... is 1".
    The initial hidden state h_0 is all zeros.

    Parameters are stored as a plain dict of numpy arrays, so that the
    optimizer in optim.py can be applied to each of them directly. None of
    their shapes depends on the sequence length T.

    Attributes:
        params: parameter name -> array, with
            Wx: (2, H)   input-to-hidden weights
            Wh: (H, H)   hidden-to-hidden weights
            b:  (H,)     hidden bias
            Wy: (H, 2)   hidden-to-output weights
            by: (2,)     output bias
        hidden_dim: H.
    """

    params: dict[str, Array]
    hidden_dim: int

    def __init__(self, hidden_dim: int = 16, seed: int = 0) -> None:
        """Allocate and initialize the parameters.

        Weights are drawn from N(0, 1 / fan_in) (LeCun initialization, the
        usual choice for tanh units); biases start at zero.
        """
        rng = np.random.default_rng(seed)
        D, H, C = 2, hidden_dim, 2
        self.hidden_dim = H
        self.params = {
            "Wx": rng.normal(0.0, 1.0 / np.sqrt(D), size=(D, H)),
            "Wh": rng.normal(0.0, 1.0 / np.sqrt(H), size=(H, H)),
            "b": np.zeros(H),
            "Wy": rng.normal(0.0, 1.0 / np.sqrt(H), size=(H, C)),
            "by": np.zeros(C),
        }

    def loss(self, X: Array, y: IntArray | None = None) -> Array | tuple[float, dict[str, Array]]:
        """Forward pass, and backward pass if labels are given.

        Args:
            X: input bits, float array of shape (N, T, 2), as produced by
                make_addition. Any T works with the same parameters.
            y: correct sum bits, integer array of shape (N, T). If None, only
                the forward pass runs.

        Returns:
            If y is None: scores, shape (N, T, 2).
            Otherwise: (loss, grads), where loss is the mean cross-entropy over
                all N * T positions (a Python float) and grads maps every key
                of self.params to the gradient of the loss w.r.t. that
                parameter, with the same shape as the parameter.
        """
        N, T, D = X.shape
        H = self.hidden_dim

        # Get parameters
        Wx = self.params["Wx"]
        Wh = self.params["Wh"]
        b = self.params["b"]
        Wy = self.params["Wy"]
        by = self.params["by"]

        # Forward: RNN
        h0 = np.zeros((N, H), dtype=np.float64)
        h, rnn_cache = rnn_forward(X, h0, Wx, Wh, b)  # h: (N, T, H)

        # Forward: temporal affine -> scores
        scores, affine_cache = temporal_affine_forward(h, Wy, by)  # (N, T, 2)

        if y is None:
            return scores

        # Loss + backward through softmax
        loss, dscores = temporal_softmax_loss(scores, y)

        # Backward through temporal affine
        dh, dWy, dby = temporal_affine_backward(dscores, affine_cache)

        # Backward through RNN
        dx, dh0, dWx, dWh, db = rnn_backward(dh, rnn_cache)

        grads = {
            "Wx": dWx,
            "Wh": dWh,
            "b": db,
            "Wy": dWy,
            "by": dby,
        }

        return loss, grads

    def predict(self, X: Array) -> IntArray:
        """Predict every bit of every sum.

        Args:
            X: input bits, shape (N, T, 2).

        Returns:
            Integer array of shape (N, T): at every position, the class with
            the higher score.
        """
        scores = self.loss(X)  # (N, T, 2)
        return np.argmax(scores, axis=2).astype(np.int64)
