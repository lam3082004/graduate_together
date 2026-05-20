"""
layers.py — Minimal numpy-only neural network layers with manual backpropagation.
"""

import numpy as np


# ── Activation functions ──────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1.0 - s)


def softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over last axis."""
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)


# ── Linear layer ─────────────────────────────────────────────────────────────

class Linear:
    """Fully-connected layer: y = x @ W + b"""

    def __init__(self, in_dim: int, out_dim: int, seed=None) -> None:
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / in_dim)
        self.W = rng.normal(0.0, scale, (in_dim, out_dim))
        self.b = np.zeros(out_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None  # cache for backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Accumulates gradients; returns dx."""
        B = max(dout.shape[0], 1)
        self.dW = self._x.T @ dout / B
        self.db = dout.mean(axis=0)
        return dout @ self.W.T

    def params_and_grads(self) -> list:
        return [(self.W, self.dW), (self.b, self.db)]


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP:
    """
    Multi-layer perceptron with manual forward/backward.

    Args:
        dims      : list of layer sizes, e.g. [4, 256, 256, 128, 2]
        hidden_act: 'relu' or 'tanh'
        out_act   : None, 'sigmoid', or 'tanh'
    """

    def __init__(self, dims: list, hidden_act: str = 'relu',
                 out_act: str | None = None) -> None:
        self.layers = [Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.hidden_act = hidden_act
        self.out_act = out_act
        self._cache: list = []  # pre-activation cache for backward

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = []
        h = x
        for i, layer in enumerate(self.layers):
            z = layer.forward(h)
            self._cache.append(z)
            is_last = (i == len(self.layers) - 1)
            if is_last:
                if self.out_act == 'sigmoid':
                    h = sigmoid(z)
                elif self.out_act == 'tanh':
                    h = tanh(z)
                else:
                    h = z
            else:
                h = relu(z) if self.hidden_act == 'relu' else tanh(z)
        return h

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backprop through all layers. Returns gradient w.r.t. input."""
        n = len(self.layers)
        for i in reversed(range(n)):
            z = self._cache[i]
            is_last = (i == n - 1)
            if is_last:
                if self.out_act == 'sigmoid':
                    dout = dout * sigmoid_grad(z)
                elif self.out_act == 'tanh':
                    dout = dout * tanh_grad(z)
                # else: linear, no change
            else:
                grad_fn = relu_grad if self.hidden_act == 'relu' else tanh_grad
                dout = dout * grad_fn(z)
            dout = self.layers[i].backward(dout)
        return dout

    def params_and_grads(self) -> list:
        pg = []
        for layer in self.layers:
            pg.extend(layer.params_and_grads())
        return pg

    def copy_weights_from(self, other: 'MLP') -> None:
        for sl, ol in zip(self.layers, other.layers):
            sl.W[:] = ol.W
            sl.b[:] = ol.b

    def soft_update_from(self, other: 'MLP', tau: float) -> None:
        """Polyak soft update: self ← tau*other + (1-tau)*self."""
        for sl, ol in zip(self.layers, other.layers):
            sl.W[:] = tau * ol.W + (1.0 - tau) * sl.W
            sl.b[:] = tau * ol.b + (1.0 - tau) * sl.b
