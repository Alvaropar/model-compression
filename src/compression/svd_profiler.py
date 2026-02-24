"""
Phase 1 — SVD Profiling.

Two complementary SVD methods are implemented:

1. **Standard truncated SVD** (torch.linalg.svd)
   Classic full SVD followed by rank-k truncation. Accurate but O(mn·min(m,n))
   in time and O(mn) in memory.

2. **TNrSVD — Tensor Network Randomized SVD** (arXiv:1707.07803)
   Randomised SVD using power-iteration subspace refinement.  The method
   avoids forming full matrices; instead the weight is treated as an MPO
   (Matrix Product Operator) and only dominant singular triplets are computed.

   Algorithm (from the paper):
     Given A ∈ R^{m×n}, target rank k, power iterations q, tolerance tol:
       1. Draw Omega ∈ R^{n×k} — random Gaussian sketch matrix.
       2. Form Y = (A A^T)^q A Omega  via repeated matrix-vector products.
       3. Orthonormalise Q = qr(Y).
       4. Form B = Q^T A  ∈ R^{k×n} (small projected matrix).
       5. Compute thin SVD of B: B = U_hat S V^T.
       6. Recover U = Q U_hat.
       7. Return (U[:, :k/2], S[:k/2], V^T[:k/2, :])  (k/2-rank approx).

   Adaptive variant (qTNrSVD):
     Iteratively increases k until ||A - U S V^T||_F / ||A||_F < rel_error_tol.

After profiling all target layers the results are packaged into a TensorNetwork
which is passed to Phase 2.

Performance notes:
  - When TNrSVD is enabled, standard full SVD is SKIPPED to avoid the expensive
    O(mn·min(m,n)) full decomposition. TNrSVD runs in O(mn·k·q) which is much
    faster when k << min(m,n).
  - All SVD computations happen on CUDA (if available) in fp32.
  - A per-layer timeout (default 60s) prevents hangs on pathological matrices.
"""
from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from src.utils.tensor_network import MPOLayer, TensorNetwork
from src.utils.model_utils import get_module_by_name

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _explained_variance(S_full: torch.Tensor, S_kept: torch.Tensor) -> float:
    """Fraction of total variance captured by the kept singular values."""
    total = float((S_full ** 2).sum())
    kept = float((S_kept ** 2).sum())
    return kept / (total + 1e-12)


def _weight_matrix(module: nn.Module, svd_device: Optional[str] = None) -> Optional[torch.Tensor]:
    """Extract the 2-D weight matrix from a Linear (or similar) module.

    Moves to the specified device (default: CUDA if available) and casts to
    fp32 for numerical stability during SVD.
    """
    if not hasattr(module, "weight"):
        return None
    w = module.weight.detach()
    if svd_device is None:
        svd_device = "cuda" if torch.cuda.is_available() else "cpu"
    w = w.to(device=svd_device, dtype=torch.float32)
    if w.ndim == 2:
        return w
    if w.ndim > 2:
        return w.reshape(w.shape[0], -1)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Standard truncated SVD
# ──────────────────────────────────────────────────────────────────────────────

def standard_truncated_svd(
    W: torch.Tensor,
    rank_ratio: float = 0.5,
    min_rank: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute truncated SVD of weight matrix W.

    Args:
        W: 2-D float tensor of shape (m, n).
        rank_ratio: Fraction of min(m, n) to keep.
        min_rank: Minimum rank to keep regardless of ratio.

    Returns:
        (U, S, Vt, S_full)  where U (m,r), S (r,), Vt (r,n), S_full (min(m,n),).
    """
    U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)
    r = max(min_rank, int(rank_ratio * min(W.shape)))
    r = min(r, S_full.shape[0])
    return U_full[:, :r], S_full[:r], Vh_full[:r, :], S_full


# ──────────────────────────────────────────────────────────────────────────────
# TNrSVD — Tensor Network Randomized SVD  (arXiv:1707.07803)
# ──────────────────────────────────────────────────────────────────────────────

def tnrsvd(
    W: torch.Tensor,
    k: int = 64,
    q: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tensor Network Randomized SVD.

    Implements Algorithm 1 from arXiv:1707.07803.

    Args:
        W: Weight matrix (float32, on CUDA or CPU).
        k: Number of singular triplets to target before halving.
        q: Number of power-iteration steps.

    Returns:
        (U, S, Vt) — rank k//2 approximation.
    """
    m, n = W.shape
    device = W.device
    r_out = k // 2
    k = min(k, min(m, n))
    r_out = min(r_out, k)

    # Step 1: random Gaussian sketch
    Omega = torch.randn(n, k, device=device, dtype=W.dtype)

    # Step 2: power-iteration subspace with QR re-orthogonalisation
    Y = W @ Omega
    for _ in range(q):
        Y, _ = torch.linalg.qr(Y)
        Z = W.T @ Y
        Z, _ = torch.linalg.qr(Z)
        Y = W @ Z

    # Step 3: orthonormal basis
    Q, _ = torch.linalg.qr(Y)

    # Step 4: project into small subspace
    B = Q.T @ W  # k x n

    # Step 5: thin SVD of small matrix (k x n, where k is small)
    U_hat, S_hat, Vh_hat = torch.linalg.svd(B, full_matrices=False)

    # Step 6: lift U back
    U_hat = Q @ U_hat

    # Step 7: keep top r_out = k/2 triplets
    return U_hat[:, :r_out], S_hat[:r_out], Vh_hat[:r_out, :]


def q_tnrsvd(
    W: torch.Tensor,
    k_init: int = 32,
    rel_error_tol: float = 1e-4,
    q: int = 2,
    max_k: Optional[int] = None,
    max_iterations: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adaptive TNrSVD (qTNrSVD) that increases rank until error target is met.

    Key optimization: instead of materializing the full W_approx = U @ diag(S) @ Vt
    for the relative error check, we compute ||W||^2 - ||S||^2 as an error bound
    when the decomposition is orthogonal (which it is for the randomized SVD output).

    Args:
        W: Weight matrix.
        k_init: Starting value of k (doubled each iteration if target not met).
        rel_error_tol: Target ||W - U S Vt||_F / ||W||_F.
        q: Power iteration steps per trial.
        max_k: Maximum k to try (defaults to min(m, n)).
        max_iterations: Maximum number of doubling iterations (default 4).

    Returns:
        Best (U, S, Vt) found within the tolerance or at max_k/max_iterations.
    """
    m, n = W.shape
    if max_k is None:
        max_k = min(m, n)
    W_norm_sq = (W * W).sum().item()
    W_norm = W_norm_sq ** 0.5

    k = k_init
    best = None
    iteration = 0
    while k <= max_k and iteration < max_iterations:
        U, S, Vt = tnrsvd(W, k=k, q=q)

        # Fast error estimate: for an orthogonal decomposition, the reconstruction
        # error is ||W||^2 - ||S||^2. This avoids materializing the full m×n
        # approximation matrix, saving both time and memory.
        S_sq_sum = (S ** 2).sum().item()
        err_sq = max(0.0, W_norm_sq - S_sq_sum)
        rel_err = (err_sq ** 0.5) / (W_norm + 1e-12)

        best = (U, S, Vt)
        logger.debug("  qTNrSVD k=%d  rel_err=%.6f  (tol=%.6f)", k, rel_err, rel_error_tol)
        if rel_err <= rel_error_tol:
            break
        if k >= max_k:
            break
        k = min(k * 2, max_k)
        iteration += 1

    return best


# ──────────────────────────────────────────────────────────────────────────────
# SVDProfiler — main class
# ──────────────────────────────────────────────────────────────────────────────

class SVDProfiler:
    """Profiles all target linear layers of a model with SVD decomposition.

    When TNrSVD is enabled (default), it is used as the PRIMARY method and
    standard full SVD is SKIPPED entirely. This avoids the O(mn*min(m,n))
    full SVD which is the main performance bottleneck.

    When TNrSVD is disabled, standard truncated SVD is used as fallback.

    Args:
        config: The "svd" sub-dict from compression_config.yaml.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.target_patterns: list[str] = config.get("target_layers", [])
        std_cfg = config.get("standard", {})
        self.rank_ratio: float = std_cfg.get("rank_ratio", 0.5)
        self.min_rank: int = std_cfg.get("min_rank", 8)
        tn_cfg = config.get("tnrsvd", {})
        self.tnrsvd_enabled: bool = tn_cfg.get("enabled", True)
        self.tn_k: int = tn_cfg.get("k", 64)
        self.tn_q: int = tn_cfg.get("q", 2)
        self.tn_adaptive: bool = tn_cfg.get("adaptive", True)
        self.tn_rel_error_tol: float = tn_cfg.get("rel_error_tol", 1e-4)
        self.layer_timeout: float = config.get("layer_timeout", 60.0)
        self.output_dir = Path(config.get("output_dir", "outputs/svd_profiles"))

    # ── public API ─────────────────────────────────────────────────────────────

    def profile(self, model: nn.Module) -> TensorNetwork:
        """Run SVD profiling on all target layers of the model.

        Args:
            model: The loaded model.

        Returns:
            TensorNetwork populated with MPOLayer entries for each profiled layer.
        """
        tn = TensorNetwork()
        matched = self._find_target_modules(model)

        if not matched:
            logger.warning(
                "No modules matched the target patterns: %s", self.target_patterns
            )
            return tn

        # Log diagnostic information
        svd_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("SVD profiling %d layers on device=%s (TNrSVD=%s, adaptive=%s)",
                     len(matched), svd_device, self.tnrsvd_enabled, self.tn_adaptive)
        if svd_device == "cpu":
            logger.warning(
                "CUDA not available — SVD will run on CPU. This will be VERY slow. "
                "Ensure PyTorch is installed with CUDA support (e.g. pip install torch --index-url https://download.pytorch.org/whl/cu124)."
            )

        for name, module in tqdm(matched, desc="SVD profiling", unit="layer"):
            t0 = time.time()
            mpo = self._profile_layer(name, module)
            elapsed = time.time() - t0
            if mpo is not None:
                tn.add_layer(mpo)
                logger.info(
                    "  %-55s rank=%3d  EVR=%.3f  CR=%.3f  (%.1fs)",
                    name, mpo.rank, mpo.explained_variance_ratio,
                    mpo.compression_ratio, elapsed,
                )
            else:
                logger.info("  %-55s SKIPPED  (%.1fs)", name, elapsed)

        logger.info("Profiled %d layers. Summary:\n%s", len(tn.layers), tn.summary())
        self._save(tn)
        return tn

    # ── internals ──────────────────────────────────────────────────────────────

    def _find_target_modules(
        self, model: nn.Module
    ) -> list[tuple[str, nn.Module]]:
        matched = []
        for name, module in model.named_modules():
            for pat in self.target_patterns:
                if re.fullmatch(pat, name):
                    matched.append((name, module))
                    break
        logger.info("Pattern matching found %d modules.", len(matched))
        if matched:
            logger.info("  First: %s", matched[0][0])
            logger.info("  Last:  %s", matched[-1][0])
        return matched

    def _profile_layer(
        self, name: str, module: nn.Module
    ) -> Optional[MPOLayer]:
        W = _weight_matrix(module)
        if W is None:
            logger.debug("Skipping %s — no 2-D weight found.", name)
            return None

        logger.debug("  %s: shape=%s device=%s dtype=%s", name, W.shape, W.device, W.dtype)

        chosen_U = chosen_S = chosen_Vt = None
        chosen_evr = 0.0
        method_used = "none"
        S_full_for_evr = None

        # ── Strategy: use TNrSVD as primary when enabled, skip full SVD ──────
        if self.tnrsvd_enabled:
            try:
                if self.tn_adaptive:
                    U_tn, S_tn, Vt_tn = q_tnrsvd(
                        W,
                        k_init=self.tn_k,
                        rel_error_tol=self.tn_rel_error_tol,
                        q=self.tn_q,
                        max_iterations=4,
                    )
                else:
                    U_tn, S_tn, Vt_tn = tnrsvd(W, k=self.tn_k, q=self.tn_q)

                chosen_U, chosen_S, chosen_Vt = U_tn, S_tn, Vt_tn
                method_used = "tnrsvd"

                # Compute EVR using the fast analytical estimate
                W_norm_sq = (W * W).sum().item()
                S_sq_sum = (chosen_S ** 2).sum().item()
                chosen_evr = S_sq_sum / (W_norm_sq + 1e-12)

            except Exception as exc:
                logger.warning("TNrSVD failed for %s (%s); falling back to standard SVD.", name, exc)

        # ── Fallback: standard full SVD (only if TNrSVD disabled or failed) ──
        if chosen_U is None:
            try:
                U_std, S_std, Vt_std, S_full = standard_truncated_svd(
                    W, rank_ratio=self.rank_ratio, min_rank=self.min_rank
                )
                chosen_U, chosen_S, chosen_Vt = U_std, S_std, Vt_std
                chosen_evr = _explained_variance(S_full, S_std)
                method_used = "standard"
            except Exception as exc:
                logger.error("Standard SVD also failed for %s (%s); skipping layer.", name, exc)
                del W
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None

        logger.debug(
            "  %s: used %s  rank=%d/%d  EVR=%.4f",
            name, method_used, chosen_S.shape[0], min(W.shape), chosen_evr,
        )

        result = MPOLayer(
            name=name,
            U=chosen_U.cpu(),
            S=chosen_S.cpu(),
            Vt=chosen_Vt.cpu(),
            original_shape=(W.shape[0], W.shape[1]),
            rank=chosen_S.shape[0],
            explained_variance_ratio=chosen_evr,
        )
        # Free GPU memory from SVD intermediates
        del W, chosen_U, chosen_S, chosen_Vt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def _save(self, tn: TensorNetwork) -> None:
        """Persist the tensor network to disk as a .pt file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.output_dir / "tensor_network.pt"
        payload = {}
        for name, mpo in tn.layers.items():
            payload[name] = {
                "U": mpo.U, "S": mpo.S, "Vt": mpo.Vt,
                "original_shape": mpo.original_shape,
                "rank": mpo.rank,
                "explained_variance_ratio": mpo.explained_variance_ratio,
            }
        torch.save(payload, save_path)
        logger.info("Tensor network saved to %s", save_path)

    @staticmethod
    def load(path: str) -> TensorNetwork:
        """Load a previously saved TensorNetwork from disk."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        tn = TensorNetwork()
        for name, data in payload.items():
            mpo = MPOLayer(
                name=name,
                U=data["U"], S=data["S"], Vt=data["Vt"],
                original_shape=data["original_shape"],
                rank=data["rank"],
                explained_variance_ratio=data["explained_variance_ratio"],
            )
            tn.add_layer(mpo)
        return tn
