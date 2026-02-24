"""
Tensor Network / MPO (Matrix Product Operator) utilities.

This module implements:
 1. MPOLayer  — a lightweight wrapper that represents a weight matrix as a
                pair of low-rank factor matrices (U @ diag(S) @ Vt), which is
                the standard output of a truncated SVD and the basic unit of
                a Tensor Network in this context.
 2. TensorNetwork — a container that holds the MPO decompositions of all
                profiled layers and exposes utilities for:
                  - analysing singular value spectra
                  - computing inter-layer activation correlations used by the
                    width pruner to decide which neurons to remove
                  - exporting importance scores

References
----------
Tensor Network Randomized SVD (TNrSVD): arXiv:1707.07803
Minitron activation importance: arXiv:2408.11796
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MPO layer: low-rank representation of a single weight matrix
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MPOLayer:
    """Matrix Product Operator representation of a single linear layer.

    A weight matrix W ∈ ℝ^{m×n} is stored as:
        W ≈ U @ diag(S) @ Vt
    where U ∈ ℝ^{m×r}, S ∈ ℝ^r, Vt ∈ ℝ^{r×n}.

    Attributes:
        name: Fully-qualified module name within the parent model.
        U: Left singular vectors  (m, r).
        S: Singular values        (r,).
        Vt: Right singular vectors (r, n).
        original_shape: (m, n) shape of the original weight matrix.
        rank: Kept rank r.
        explained_variance_ratio: Fraction of total variance captured by rank-r.
    """
    name: str
    U: torch.Tensor
    S: torch.Tensor
    Vt: torch.Tensor
    original_shape: tuple[int, int]
    rank: int
    explained_variance_ratio: float

    # Activation statistics collected during calibration forward passes.
    # Shape: (out_features,) — one score per output neuron.
    activation_importance: Optional[torch.Tensor] = None

    @property
    def reconstructed(self) -> torch.Tensor:
        """Reconstruct the low-rank approximation of W."""
        return self.U @ torch.diag(self.S) @ self.Vt

    @property
    def compression_ratio(self) -> float:
        """Ratio of parameters stored vs original."""
        original = self.original_shape[0] * self.original_shape[1]
        compressed = self.rank * (self.original_shape[0] + 1 + self.original_shape[1])
        return compressed / original

    def singular_value_spectrum(self) -> np.ndarray:
        return self.S.cpu().float().numpy()

    def effective_rank(self, threshold: float = 0.99) -> int:
        """Minimum rank needed to capture `threshold` fraction of variance."""
        s = self.S.cpu().float().numpy()
        cumvar = np.cumsum(s ** 2) / np.sum(s ** 2)
        return int(np.searchsorted(cumvar, threshold)) + 1


@dataclass
class TensorNetwork:
    """Collection of MPO decompositions across the model's layers.

    The TensorNetwork acts as the profiling output of Phase 1 (SVD profiling).
    Phase 2 (width pruning) then queries it for per-neuron importance scores
    computed from both singular value spectra and activation magnitudes.

    Attributes:
        layers: Ordered dict mapping module name → MPOLayer.
    """
    layers: dict[str, MPOLayer] = field(default_factory=dict)

    # ── insertion ──────────────────────────────────────────────────────────────

    def add_layer(self, mpo: MPOLayer) -> None:
        self.layers[mpo.name] = mpo

    # ── queries ────────────────────────────────────────────────────────────────

    def get_mpo(self, name: str) -> MPOLayer:
        return self.layers[name]

    def all_names(self) -> list[str]:
        return list(self.layers.keys())

    # ── importance computation ─────────────────────────────────────────────────

    def compute_singular_value_importance(self) -> dict[str, torch.Tensor]:
        """Compute per-output-neuron importance from singular value spectra.

        For each layer we compute:
            importance_i = ||U[:, :] * S||_2  projected onto output dimension

        In practice, each output neuron i is associated with a row U[i, :].
        The neuron importance is ||U[i, :] * S||_2, i.e., how much that
        neuron participates in high-variance directions.

        Returns:
            Dict mapping name → 1-D tensor of shape (out_features,).
        """
        scores: dict[str, torch.Tensor] = {}
        for name, mpo in self.layers.items():
            # U: (m, r),  S: (r,)  →  weighted_U: (m, r)
            weighted_U = mpo.U * mpo.S.unsqueeze(0)          # broadcast
            importance = weighted_U.norm(dim=1)               # (m,)
            scores[name] = importance.cpu()
        return scores

    def compute_combined_importance(
        self,
        sv_weight: float = 0.5,
        act_weight: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Combine singular-value importance with activation importance.

        If activation_importance has not been set on an MPOLayer the method
        falls back to using only the singular value score for that layer.

        Args:
            sv_weight: Weight for singular-value based importance.
            act_weight: Weight for activation-magnitude importance.

        Returns:
            Dict mapping name → normalised 1-D importance tensor.
        """
        sv_scores = self.compute_singular_value_importance()
        combined: dict[str, torch.Tensor] = {}

        for name, sv in sv_scores.items():
            mpo = self.layers[name]
            if mpo.activation_importance is not None:
                act = mpo.activation_importance.cpu()
                # Normalise both to [0, 1] before combining
                sv_norm = (sv - sv.min()) / (sv.max() - sv.min() + 1e-8)
                act_norm = (act - act.min()) / (act.max() - act.min() + 1e-8)
                combined[name] = sv_weight * sv_norm + act_weight * act_norm
            else:
                sv_norm = (sv - sv.min()) / (sv.max() - sv.min() + 1e-8)
                combined[name] = sv_norm

        return combined

    def inter_layer_correlation(
        self,
        name_a: str,
        name_b: str,
    ) -> float:
        """Pearson correlation between the activation importances of two layers.

        This helps identify redundant layer pairs that can both be pruned
        more aggressively without double-penalising the model.

        Args:
            name_a: Module name of first layer.
            name_b: Module name of second layer.

        Returns:
            Correlation coefficient in [-1, 1], or 0 if data unavailable.
        """
        a = self.layers[name_a].activation_importance
        b = self.layers[name_b].activation_importance
        if a is None or b is None:
            return 0.0

        a_f = a.cpu().float()
        b_f = b.cpu().float()
        # Align lengths (min)
        n = min(len(a_f), len(b_f))
        a_f, b_f = a_f[:n], b_f[:n]

        a_c = a_f - a_f.mean()
        b_c = b_f - b_f.mean()
        denom = a_c.norm() * b_c.norm()
        if denom < 1e-8:
            return 0.0
        return float((a_c * b_c).sum() / denom)

    # ── summary ────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = ["TensorNetwork Summary", "=" * 60]
        for name, mpo in self.layers.items():
            lines.append(
                f"  {name:55s}  shape={mpo.original_shape}  "
                f"rank={mpo.rank:4d}  var={mpo.explained_variance_ratio:.3f}  "
                f"CR={mpo.compression_ratio:.3f}"
            )
        return "\n".join(lines)
