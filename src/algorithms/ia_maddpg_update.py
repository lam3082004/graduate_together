"""
ia_maddpg_update.py — Redirected to ia_maddpg.py (legacy stub).

All update logic has been consolidated into algorithms/ia_maddpg.py.
This file exists only to preserve any external imports that reference
`soft_update` from this module.
"""

from algorithms.ia_maddpg import soft_update_actor  # noqa: F401


def soft_update(src, tgt, tau: float) -> None:
    """Polyak soft-update: tgt ← tau*src + (1-tau)*tgt."""
    tgt.soft_update_from(src, tau)
