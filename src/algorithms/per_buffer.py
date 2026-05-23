"""
per_buffer.py — Prioritized Experience Replay buffer using a binary sum-tree.

Stores transitions: (su_obs, uav_obs, su_actions, uav_actions,
                     rewards, next_su_obs, next_uav_obs, done, channel_gains)
"""

import numpy as np


class SumTree:
    """Binary sum-tree for O(log n) priority sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        # Internal nodes + leaf nodes; leaves start at index capacity-1
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write = 0          # Next write position (circular)
        self.n_entries = 0

    # ── Private helpers ────────────────────────────────────────────────────────

    def _propagate(self, idx: int, delta: float) -> None:
        """Propagate priority change up from leaf to root."""
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        """Walk tree from node idx to the leaf whose prefix sum covers s."""
        # Leaf nodes occupy indices [capacity-1 .. 2*capacity-2]
        if idx >= self.capacity - 1:
            return idx
        left = 2 * idx + 1
        right = left + 1
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        """Insert data with given priority."""
        leaf_idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(leaf_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """Update priority at leaf index idx."""
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def sample(self, batch_size: int, beta: float):
        """
        Sample batch_size transitions proportional to priority.

        Returns
        -------
        data_batch : list of transitions
        indices    : leaf indices (for priority update)
        weights    : importance-sampling weights (numpy array, shape [batch_size])
        """
        indices = np.zeros(batch_size, dtype=np.int64)
        weights = np.zeros(batch_size, dtype=np.float32)
        data_batch = []

        segment = self.total / batch_size
        leaf_start = self.capacity - 1
        leaf_vals = self.tree[leaf_start: leaf_start + self.n_entries]
        nonzero = leaf_vals[leaf_vals > 0]
        min_prob = float(nonzero.min()) / self.total if len(nonzero) else 1e-8
        min_prob = max(min_prob, 1e-8)
        max_weight = (min_prob * self.n_entries) ** (-beta)

        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            s = np.random.uniform(lo, hi)
            leaf_idx = self._retrieve(0, s)
            data_idx = leaf_idx - self.capacity + 1
            prob = self.tree[leaf_idx] / self.total
            prob = max(prob, 1e-8)
            weights[i] = ((prob * self.n_entries) ** (-beta)) / max_weight
            indices[i] = leaf_idx
            data_batch.append(self.data[data_idx])

        return data_batch, indices, weights


class PERBuffer:
    """
    Prioritized Experience Replay.

    Each transition is a tuple:
        (su_obs, uav_obs, su_actions, uav_actions,
         rewards, next_su_obs, next_uav_obs, done, channel_gains)
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.max_priority = 1.0

    def push(self, transition: tuple) -> None:
        """Store transition with current maximum priority."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size: int, beta: float = 0.4):
        """
        Sample a prioritised mini-batch.

        Returns
        -------
        batch   : list[tuple] of transitions
        indices : leaf indices for priority update
        weights : IS weights, shape (batch_size,)
        """
        return self.tree.sample(batch_size, beta)

    def update_priorities(self, indices: np.ndarray,
                          td_errors: np.ndarray) -> None:
        """Re-weight sampled transitions by new TD errors."""
        for idx, err in zip(indices, td_errors):
            priority = (abs(float(err)) + 1e-6) ** self.alpha
            self.tree.update(int(idx), priority)
            if priority > self.max_priority:
                self.max_priority = priority

    def __len__(self) -> int:
        return self.tree.n_entries
