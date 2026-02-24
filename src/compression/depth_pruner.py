"""
Phase 3 — Depth Pruning via Sliding Window Layer Removal.

Algorithm
---------
Given a model with L transformer layers and a calibration dataset:

1. For each window of size w ∈ [min_size, max_size] starting at position i:
   a. Temporarily bypass layers i … i+w-1  (replace with identity / skip).
   b. Compute the mean cross-entropy loss on the calibration set.
   c. Record (i, w, Δloss) where Δloss = loss_pruned − loss_baseline.
   d. Restore the layers.

2. Select the window (i*, w*) that minimises Δloss while preferring windows
   whose Δloss ≤ loss_threshold (acceptable degradation).  The target_layer_removal
   config value biases the search toward windows of a particular size.

3. Physically remove layers i* … i*+w*-1 from the model's layer list.

Key design decisions (informed by Minitron findings):
  • Contiguous layer removal consistently outperforms importance-score-based
    non-contiguous removal.  We therefore search over *contiguous* windows only.
  • We evaluate against a fixed baseline loss so that all windows are comparable
    regardless of layer depth.
  • The calibration dataset is the same small text dataset used in Phase 2;
    256 samples at seq_len=512 is sufficient for a reliable signal.
  • After removal the remaining layers are re-indexed so that downstream code
    (and the model itself) sees a consistent numbering.

Block Importance (BI) score (computed but used as a secondary signal):
    BI_i = 1 - cosine_similarity(input_i, output_i)
Layers with BI close to 0 are nearly identity transforms and are prime
candidates for removal.  We use this as a tie-breaker when multiple windows
have similar Δloss.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.model_utils import get_transformer_layers, get_llm_submodule

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WindowResult:
    """Result for a single sliding window evaluation."""
    start: int          # Index of first removed layer
    size: int           # Number of layers in the window
    delta_loss: float   # loss_pruned − loss_baseline
    bi_score: float     # Mean Block Importance of layers in window (lower = more removable)

    @property
    def end(self) -> int:
        """Exclusive end index."""
        return self.start + self.size


# ──────────────────────────────────────────────────────────────────────────────
# Block importance helper
# ──────────────────────────────────────────────────────────────────────────────

class _BICollector:
    """Forward hook that measures Block Importance of a single layer.

    BI_i = 1 − cosine_similarity(layer_input, layer_output)

    We average over all (batch, token) positions.
    """

    def __init__(self) -> None:
        self.bi_values: list[float] = []

    def hook(self, module: nn.Module, inp: tuple, output) -> None:
        # inp[0]: (B, T, H) — the residual stream entering the block
        # output: (B, T, H) or tuple where first element is the hidden state
        x_in = inp[0].detach().float()
        x_out = output[0].detach().float() if isinstance(output, tuple) else output.detach().float()
        # Flatten batch × time → (N, H)
        x_in_flat = x_in.view(-1, x_in.shape[-1])
        x_out_flat = x_out.view(-1, x_out.shape[-1])
        cos_sim = F.cosine_similarity(x_in_flat, x_out_flat, dim=-1)  # (N,)
        bi = 1.0 - cos_sim.mean().item()
        self.bi_values.append(bi)

    def mean_bi(self) -> float:
        if not self.bi_values:
            return 0.0
        return sum(self.bi_values) / len(self.bi_values)


# ──────────────────────────────────────────────────────────────────────────────
# Identity skip wrapper
# ──────────────────────────────────────────────────────────────────────────────

class _IdentityLayer(nn.Module):
    """Replaces a transformer layer with a pure residual skip.

    Qwen3 decoder layers expose an `attention_type` attribute that the model's
    forward loop reads *before* calling the layer. We copy this from the
    original layer so the model loop works without modification.

    The forward method accepts all the same keyword arguments as a real decoder
    layer but simply returns hidden_states unchanged.
    """

    def __init__(self, original_layer: Optional[nn.Module] = None) -> None:
        super().__init__()
        # Copy attributes that the model loop accesses before forward()
        if original_layer is not None:
            self.attention_type = getattr(original_layer, "attention_type", "eager")
        else:
            self.attention_type = "eager"

    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


# ──────────────────────────────────────────────────────────────────────────────
# Loss evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 64,
    llm: Optional[nn.Module] = None,
) -> float:
    """Compute mean cross-entropy loss on the calibration loader.

    For VLMs like InternVL whose full forward() requires pixel_values,
    pass `llm` (the language model submodule) which accepts text-only input.
    """
    forward_module = llm if llm is not None else model
    forward_module.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        if n >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = forward_module(input_ids=input_ids, labels=labels)
        total_loss += out.loss.item()
        n += 1
    return total_loss / max(n, 1)


# ──────────────────────────────────────────────────────────────────────────────
# DepthPruner
# ──────────────────────────────────────────────────────────────────────────────

class DepthPruner:
    """Removes a contiguous window of transformer layers to reduce model depth.

    The search procedure evaluates all windows of sizes in [min_size, max_size]
    over the interior layers (first and last layers are typically skipped since
    they interact with embeddings / LM head directly).

    Args:
        config: The "depth_pruning" sub-dict from compression_config.yaml.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        sw = config.get("sliding_window", {})
        self.min_size: int = sw.get("min_size", 1)
        self.max_size: int = sw.get("max_size", 5)
        self.stride: int = sw.get("stride", 1)
        self.loss_threshold: float = sw.get("loss_threshold", 0.15)
        self.target_removal: int = config.get("target_layer_removal", 3)
        self.always_prune: bool = config.get("always_prune", True)
        self.output_dir = Path(config.get("output_dir", "outputs/depth_pruning"))

    # ── public API ─────────────────────────────────────────────────────────────

    def prune(
        self,
        model: nn.Module,
        calibration_loader: DataLoader,
        device: torch.device,
    ) -> nn.Module:
        """Run the depth pruning search and apply the best window.

        Args:
            model: The model to prune (returned modified in-place).
            calibration_loader: DataLoader for calibration sequences.
            device: Compute device.

        Returns:
            Depth-pruned model.
        """
        logger.info("=== Depth Pruning — Phase 3 ===")
        transformer_layers = get_transformer_layers(model)
        n_layers = len(transformer_layers)
        logger.info("Model has %d transformer layers.", n_layers)

        # ── Pre-compute Block Importance scores ────────────────────────────────
        bi_scores = self._compute_bi_scores(model, transformer_layers, calibration_loader, device)
        logger.info("Block Importance scores: %s", [f"{b:.4f}" for b in bi_scores])

        # ── Baseline loss ──────────────────────────────────────────────────────
        # Use LLM submodule for VLMs (avoids pixel_values requirement)
        llm = get_llm_submodule(model)
        baseline_loss = _compute_loss(model, calibration_loader, device, llm=llm)
        logger.info("Baseline loss: %.6f", baseline_loss)

        # ── Sliding window search ──────────────────────────────────────────────
        results: list[WindowResult] = []
        # Skip first and last layers to avoid disrupting embedding/LM head interface
        search_start = 1
        search_end = n_layers - 1

        layers_container = self._get_layers_container(model)

        total_windows = sum(
            max(0, (search_end - search_start - w) // self.stride + 1)
            for w in range(self.min_size, min(self.max_size, search_end - search_start) + 1)
        )

        with tqdm(total=total_windows, desc="Sliding window search") as pbar:
            for w in range(self.min_size, min(self.max_size, search_end - search_start) + 1):
                for i in range(search_start, search_end - w + 1, self.stride):
                    result = self._evaluate_window(
                        model,
                        layers_container,
                        transformer_layers,
                        i, w,
                        calibration_loader,
                        device,
                        baseline_loss,
                        bi_scores,
                        llm=llm,
                    )
                    results.append(result)
                    pbar.update(1)
                    pbar.set_postfix(
                        start=i, size=w, delta_loss=f"{result.delta_loss:.4f}"
                    )

        if not results:
            logger.warning("No windows evaluated; skipping depth pruning.")
            return model

        # ── Select best window ─────────────────────────────────────────────────
        best = self._select_window(results)
        logger.info(
            "Selected window: layers [%d, %d) (size=%d)  Δloss=%.6f  BI=%.4f",
            best.start, best.end, best.size, best.delta_loss, best.bi_score,
        )

        # ── Apply: remove the selected layers ─────────────────────────────────
        self._remove_layers(model, layers_container, best.start, best.end)

        # Save search results
        self._save_results(results, best)

        remaining = len(get_transformer_layers(model))
        logger.info(
            "Depth pruning complete: %d → %d layers (removed %d).",
            n_layers, remaining, n_layers - remaining,
        )
        return model

    # ── Block Importance ───────────────────────────────────────────────────────

    @torch.no_grad()
    def _compute_bi_scores(
        self,
        model: nn.Module,
        layers: list[nn.Module],
        loader: DataLoader,
        device: torch.device,
    ) -> list[float]:
        """Compute per-layer Block Importance over the calibration set."""
        collectors = [_BICollector() for _ in layers]
        handles = [
            layer.register_forward_hook(c.hook)
            for layer, c in zip(layers, collectors)
        ]
        # Use LLM submodule for VLMs (avoids pixel_values requirement)
        llm = get_llm_submodule(model)
        llm.eval()
        for i, batch in enumerate(loader):
            if i >= 32:   # 32 batches is enough for stable BI
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            llm(input_ids=input_ids, attention_mask=attention_mask)
        for h in handles:
            h.remove()
        return [c.mean_bi() for c in collectors]

    # ── Window evaluation ──────────────────────────────────────────────────────

    def _evaluate_window(
        self,
        model: nn.Module,
        layers_container: nn.ModuleList,
        transformer_layers: list[nn.Module],
        start: int,
        size: int,
        loader: DataLoader,
        device: torch.device,
        baseline_loss: float,
        bi_scores: list[float],
        llm: Optional[nn.Module] = None,
    ) -> WindowResult:
        """Temporarily remove layers [start, start+size) and measure loss."""
        # Save originals
        original_layers = {i: layers_container[i] for i in range(start, start + size)}

        # Replace with identity layers (copy attention_type from originals
        # so Qwen3's forward loop can read it before calling forward)
        for i in range(start, start + size):
            identity = _IdentityLayer(original_layer=layers_container[i]).to(device)
            layers_container[i] = identity

        loss = _compute_loss(model, loader, device, max_batches=32, llm=llm)

        # Restore
        for i, orig in original_layers.items():
            layers_container[i] = orig

        delta = loss - baseline_loss
        mean_bi = sum(bi_scores[start:start + size]) / size

        return WindowResult(start=start, size=size, delta_loss=delta, bi_score=mean_bi)

    # ── Window selection ───────────────────────────────────────────────────────

    def _select_window(self, results: list[WindowResult]) -> WindowResult:
        """Choose the best window to remove.

        Priority:
          1. Windows whose delta_loss ≤ loss_threshold.
             Among those, prefer windows whose size is closest to target_removal.
             Tie-break by smallest delta_loss, then smallest BI (most removable).
          2. If none meet the threshold (and always_prune=True), pick the window
             with minimum delta_loss, preferring target size.
        """
        # Sort by: size proximity to target, then delta_loss, then bi_score
        def sort_key(r: WindowResult):
            size_dist = abs(r.size - self.target_removal)
            return (size_dist, r.delta_loss, r.bi_score)

        candidates = [r for r in results if r.delta_loss <= self.loss_threshold]

        if candidates:
            return min(candidates, key=sort_key)

        if self.always_prune:
            logger.warning(
                "No window met loss_threshold=%.4f. Picking minimum-loss window.",
                self.loss_threshold,
            )
            return min(results, key=lambda r: (r.delta_loss, abs(r.size - self.target_removal)))

        raise RuntimeError(
            f"No window found with delta_loss ≤ {self.loss_threshold}. "
            "Set always_prune=true to force pruning."
        )

    # ── Layer removal ──────────────────────────────────────────────────────────

    def _get_layers_container(self, model: nn.Module) -> nn.ModuleList:
        """Return the nn.ModuleList that holds the transformer layers.

        Supports:
          - InternVL custom: model.language_model.model.layers
          - Qwen2-VL / Qwen3-VL: model.model.layers
          - Generic: model.layers
        """
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        if hasattr(model, "layers"):
            return model.layers
        raise AttributeError("Cannot find layers container on model.")

    def _remove_layers(
        self,
        model: nn.Module,
        layers_container: nn.ModuleList,
        start: int,
        end: int,
    ) -> None:
        """Physically delete layers [start, end) from the ModuleList."""
        keep_indices = [i for i in range(len(layers_container)) if i < start or i >= end]
        new_layers = nn.ModuleList([layers_container[i] for i in keep_indices])

        # Set the new layers in the right location
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                lm.model.layers = new_layers
            else:
                lm.layers = new_layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            model.model.layers = new_layers
        else:
            model.layers = new_layers

        # Update model config if it tracks num_hidden_layers
        # For InternVL: the LLM config is nested under model.config.llm_config
        cfg = getattr(model, "config", None)
        if cfg is not None:
            llm_cfg = getattr(cfg, "llm_config", cfg)
            if hasattr(llm_cfg, "num_hidden_layers"):
                llm_cfg.num_hidden_layers = len(new_layers)
            elif hasattr(cfg, "num_hidden_layers"):
                cfg.num_hidden_layers = len(new_layers)

            # Also update layer_types if present (Qwen3 validates this at load time)
            if hasattr(llm_cfg, "layer_types") and isinstance(llm_cfg.layer_types, list):
                llm_cfg.layer_types = [llm_cfg.layer_types[i] for i in keep_indices]
                logger.debug("Updated layer_types: %d entries", len(llm_cfg.layer_types))
            elif hasattr(cfg, "layer_types") and isinstance(cfg.layer_types, list):
                cfg.layer_types = [cfg.layer_types[i] for i in keep_indices]

            # Update max_window_layers if present (Qwen3 uses this for sliding window attn)
            if hasattr(llm_cfg, "max_window_layers"):
                llm_cfg.max_window_layers = len(new_layers)
            elif hasattr(cfg, "max_window_layers"):
                cfg.max_window_layers = len(new_layers)

            logger.debug("Updated model.config.num_hidden_layers = %d", len(new_layers))

    # ── persistence ───────────────────────────────────────────────────────────

    def _save_results(self, results: list[WindowResult], best: WindowResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "window_search_results.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write("start\tsize\tdelta_loss\tbi_score\n")
            for r in sorted(results, key=lambda x: x.delta_loss):
                marker = "  <-- SELECTED" if r is best else ""
                f.write(f"{r.start}\t{r.size}\t{r.delta_loss:.6f}\t{r.bi_score:.6f}{marker}\n")
        logger.info("Window search results saved to %s", path)
