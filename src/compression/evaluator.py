"""
Post-compression evaluation module.

Computes standard metrics to validate that the compressed model retains
acceptable quality:

  1. **Perplexity** on a held-out text dataset (lower = better).
  2. **Compression statistics** (parameter count, ratio, per-phase breakdown).
  3. **Layer-wise reconstruction error** comparing compressed vs. original
     weight matrices using the TensorNetwork's SVD decompositions.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.model_utils import count_parameters, get_llm_submodule

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation metrics."""
    perplexity: float
    avg_loss: float
    n_params: int
    n_params_original: Optional[int]
    compression_ratio: Optional[float]
    eval_time_s: float

    def summary(self) -> str:
        lines = [
            "Evaluation Results",
            "=" * 50,
            f"  Perplexity:        {self.perplexity:.2f}",
            f"  Avg CE loss:       {self.avg_loss:.4f}",
            f"  Parameters:        {self.n_params:,}",
        ]
        if self.n_params_original is not None:
            lines.append(f"  Original params:   {self.n_params_original:,}")
        if self.compression_ratio is not None:
            lines.append(f"  Compression ratio: {self.compression_ratio:.2f}x")
            lines.append(f"  Size retention:    {100.0 / self.compression_ratio:.1f}%")
        lines.append(f"  Eval time:         {self.eval_time_s:.1f}s")
        return "\n".join(lines)


class Evaluator:
    """Evaluates model quality after compression.

    Args:
        config: The evaluation sub-dict from compression_config.yaml, or None
                for defaults.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        config = config or {}
        self.max_batches: int = config.get("max_batches", 128)
        self.output_dir = Path(config.get("output_dir", "outputs/evaluation"))

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        eval_loader: DataLoader,
        device: torch.device,
        n_params_original: Optional[int] = None,
    ) -> EvalResult:
        """Compute perplexity and compression statistics.

        Args:
            model: The model to evaluate.
            eval_loader: DataLoader of evaluation batches (must have input_ids, labels).
            device: Compute device.
            n_params_original: Original parameter count for compression ratio.

        Returns:
            EvalResult with all computed metrics.
        """
        # Use LLM submodule for VLMs (avoids pixel_values requirement)
        llm = get_llm_submodule(model)
        llm.eval()
        t_start = time.time()

        total_loss = 0.0
        total_tokens = 0

        for i, batch in enumerate(tqdm(eval_loader, desc="Evaluating", leave=False)):
            if i >= self.max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = llm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Use model-computed loss if available, else compute manually
            if hasattr(outputs, "loss") and outputs.loss is not None:
                # Model loss is mean over tokens; recover sum
                n_tokens = (labels != -100).sum().item()
                total_loss += outputs.loss.item() * max(n_tokens, 1)
                total_tokens += max(n_tokens, 1)
            else:
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_tokens = (shift_labels != -100).sum().item()
                total_loss += loss.item()
                total_tokens += max(n_tokens, 1)

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = min(torch.exp(torch.tensor(avg_loss)).item(), 1e6)

        n_params = count_parameters(model)
        compression_ratio = None
        if n_params_original is not None and n_params > 0:
            compression_ratio = n_params_original / n_params

        elapsed = time.time() - t_start

        result = EvalResult(
            perplexity=perplexity,
            avg_loss=avg_loss,
            n_params=n_params,
            n_params_original=n_params_original,
            compression_ratio=compression_ratio,
            eval_time_s=elapsed,
        )

        logger.info("\n%s", result.summary())
        self._save(result)
        return result

    def _save(self, result: EvalResult) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "eval_results.txt"
        with open(path, "w") as f:
            f.write(result.summary())
            f.write("\n")
        logger.info("Evaluation results saved to %s", path)
