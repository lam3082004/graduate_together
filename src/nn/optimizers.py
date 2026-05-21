"""
optimizers.py — Adam optimizer for numpy MLP layers.
"""

import numpy as np


class Adam:
    """
    Adam optimizer operating directly on (param, grad) tuple lists.

    Usage:
        opt = Adam(lr=1e-3)
        opt.step(mlp.params_and_grads())   # in-place update of all params
    """

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict = {}
        self.v: dict = {}
        self.t: int = 0

    def step(self, params_and_grads: list) -> None:
        """
        In-place Adam update for all (param, grad) pairs.

        params_and_grads : list of (param_array, grad_array) tuples
        """
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for idx, (p, g) in enumerate(params_and_grads):
            if idx not in self.m:
                self.m[idx] = np.zeros_like(p)
                self.v[idx] = np.zeros_like(p)
            self.m[idx] = self.beta1 * self.m[idx] + (1.0 - self.beta1) * g
            self.v[idx] = self.beta2 * self.v[idx] + (1.0 - self.beta2) * g ** 2
            m_hat = self.m[idx] / bc1
            v_hat = self.v[idx] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self) -> None:
        """Reset moment estimates (useful when re-using optimizer)."""
        self.m.clear()
        self.v.clear()
        self.t = 0
