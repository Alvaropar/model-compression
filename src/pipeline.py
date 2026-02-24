"""
Main compression pipeline orchestrator.

Runs the four phases in sequence:
  Phase 1 — SVD Profiling  (SVDProfiler)
  Phase 2 — Width Pruning  (WidthPruner)
  Phase 3 — Depth Pruning  (DepthPruner)
  Phase 4 — Recovery       (RecoveryTrainer)
  (optional) Evaluation    (Evaluator)

Each phase can be independently skipped via the `skip_phases` argument, which
is useful for resuming a partially completed run or for ablation studies.

Usage
-----
  python -m scripts.run_pipeline --config configs/compression_config.yaml

Or from Python:
  from src.pipeline import CompressionPipeline
  pipeline = CompressionPipeline.from_config("configs/compression_config.yaml")
  pipeline.run()
"""
from __future__ import annotations

import copy
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import yaml

from src.utils.model_utils import load_model_and_tokenizer, save_model, count_parameters
from src.utils.data_utils import build_calibration_loader, build_training_loader
from src.compression.svd_profiler import SVDProfiler
from src.compression.width_pruner import WidthPruner
from src.compression.depth_pruner import DepthPruner
from src.compression.recovery import RecoveryTrainer
from src.compression.evaluator import Evaluator

logger = logging.getLogger(__name__)


class CompressionPipeline:
    """End-to-end model compression pipeline.

    Args:
        config: Parsed YAML configuration dictionary.
        skip_phases: Set of phase names to skip.
                     Valid values: {"svd", "width", "depth", "recovery"}.
        device: Override device (default: auto-detect CUDA).
        run_eval: Whether to run evaluation after compression.
    """

    PHASES = ("svd", "width", "depth", "recovery")

    def __init__(
        self,
        config: dict,
        skip_phases: Optional[set[str]] = None,
        device: Optional[torch.device] = None,
        run_eval: bool = False,
    ) -> None:
        self.config = config
        self.skip_phases: set[str] = skip_phases or set()
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.run_eval = run_eval
        self._model = None
        self._processor = None
        self._teacher_model = None

    # ── factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_path: str,
        skip_phases: Optional[set[str]] = None,
        device: Optional[torch.device] = None,
        run_eval: bool = False,
    ) -> "CompressionPipeline":
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config, skip_phases=skip_phases, device=device, run_eval=run_eval)

    # ── main entry point ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full compression pipeline."""
        t_start = time.time()
        logger.info("=" * 70)
        logger.info("Qwen3-VL Model Compression Pipeline")
        logger.info("Device: %s", self.device)
        logger.info("Skipping phases: %s", self.skip_phases or "none")
        logger.info("=" * 70)

        # ── Load model ─────────────────────────────────────────────────────────
        model_cfg = self.config["model"]
        self._model, self._processor = load_model_and_tokenizer(
            model_name=model_cfg["name"],
            dtype=model_cfg.get("dtype", "bfloat16"),
            device_map=model_cfg.get("device_map", "auto"),
        )
        n_params_original = count_parameters(self._model)
        logger.info("Original model parameters: %s", f"{n_params_original:,}")

        # Keep a reference to the original model name for teacher loading
        self._original_model_name = model_cfg["name"]

        # ── Phase 1: SVD Profiling ─────────────────────────────────────────────
        tensor_network = None
        if "svd" not in self.skip_phases:
            tensor_network = self._run_svd_profiling()
        else:
            logger.info("[SKIP] Phase 1 — SVD Profiling")
            # Try to load from disk
            tn_path = Path(self.config["svd"]["output_dir"]) / "tensor_network.pt"
            if tn_path.exists():
                logger.info("Loading tensor network from %s", tn_path)
                tensor_network = SVDProfiler.load(str(tn_path))
            else:
                logger.warning("No cached tensor network found at %s. Width pruning will use activation scores only.", tn_path)

        # Save intermediate if configured
        if self.config["output"].get("save_intermediate") and tensor_network is not None:
            logger.info("Tensor network ready with %d layers.", len(tensor_network.layers))

        # ── Phase 2: Width Pruning ─────────────────────────────────────────────
        if "width" not in self.skip_phases and tensor_network is not None:
            self._run_width_pruning(tensor_network)
        else:
            logger.info("[SKIP] Phase 2 — Width Pruning")

        if self.config["output"].get("save_intermediate"):
            self._save_intermediate("after_width_pruning")

        # ── Phase 3: Depth Pruning ─────────────────────────────────────────────
        if "depth" not in self.skip_phases:
            self._run_depth_pruning()
        else:
            logger.info("[SKIP] Phase 3 — Depth Pruning")

        if self.config["output"].get("save_intermediate"):
            self._save_intermediate("after_depth_pruning")

        # ── Phase 4: Recovery ──────────────────────────────────────────────────
        if "recovery" not in self.skip_phases:
            self._run_recovery()
        else:
            logger.info("[SKIP] Phase 4 — Recovery")

        # ── Final model save ───────────────────────────────────────────────────
        final_dir = self.config["output"]["compressed_model_dir"]
        save_model(self._model, self._processor, final_dir)

        n_params_final = count_parameters(self._model)
        elapsed = time.time() - t_start
        logger.info("=" * 70)
        logger.info("Pipeline complete in %.1f s", elapsed)
        logger.info("Original params : %s", f"{n_params_original:,}")
        logger.info("Compressed params: %s", f"{n_params_final:,}")
        logger.info(
            "Compression ratio: %.2fx  (%.1f%% of original)",
            n_params_original / max(n_params_final, 1),
            100.0 * n_params_final / n_params_original,
        )
        logger.info("Saved to: %s", final_dir)
        logger.info("=" * 70)

        # ── Evaluation (optional) ──────────────────────────────────────────────
        if self.run_eval:
            self._run_evaluation(n_params_original)

        # Clean up teacher if we loaded one
        if self._teacher_model is not None:
            del self._teacher_model
            self._teacher_model = None
            torch.cuda.empty_cache()

    # ── Phase implementations ──────────────────────────────────────────────────

    def _run_svd_profiling(self):
        logger.info("─── Phase 1: SVD Profiling ───")
        profiler = SVDProfiler(self.config["svd"])
        return profiler.profile(self._model)

    def _run_width_pruning(self, tensor_network) -> None:
        logger.info("─── Phase 2: Width Pruning ───")
        calib_cfg = self.config["width_pruning"]["calibration"]
        loader = build_calibration_loader(
            dataset_name=calib_cfg["dataset"],
            split=calib_cfg["dataset_split"],
            num_samples=calib_cfg["num_samples"],
            seq_len=calib_cfg["seq_len"],
            tokenizer=self._processor,
            batch_size=1,
        )
        pruner = WidthPruner(self.config["width_pruning"], tensor_network)
        self._model = pruner.prune(self._model, loader, self.device)

    def _run_depth_pruning(self) -> None:
        logger.info("─── Phase 3: Depth Pruning ───")
        calib_cfg = self.config["depth_pruning"]["calibration"]
        loader = build_calibration_loader(
            dataset_name=calib_cfg["dataset"],
            split=calib_cfg["dataset_split"],
            num_samples=calib_cfg["num_samples"],
            seq_len=calib_cfg["seq_len"],
            tokenizer=self._processor,
            batch_size=1,
        )
        pruner = DepthPruner(self.config["depth_pruning"])
        self._model = pruner.prune(self._model, loader, self.device)

    def _run_recovery(self) -> None:
        logger.info("─── Phase 4: Recovery ───")
        rec_cfg = self.config["recovery"]
        train_cfg = rec_cfg["training"]
        loader = build_training_loader(
            dataset_name=rec_cfg["dataset"],
            split=rec_cfg["dataset_split"],
            num_samples=rec_cfg.get("max_samples", 50000),
            seq_len=rec_cfg.get("seq_len", 1024),
            tokenizer=self._processor,
            batch_size=train_cfg["per_device_train_batch_size"],
            num_workers=train_cfg.get("dataloader_num_workers", 4),
        )

        # Set the teacher model name in the config so RecoveryTrainer can
        # load it directly in int4 quantization for memory efficiency.
        # RecoveryTrainer handles all teacher loading (int4 on GPU or CPU fallback).
        if rec_cfg.get("method") in ("distillation", "both"):
            dist_cfg = rec_cfg.setdefault("distillation", {})
            if dist_cfg.get("teacher_model") is None:
                dist_cfg["teacher_model"] = self._original_model_name
                logger.info("Set teacher_model to original: %s", self._original_model_name)

        trainer = RecoveryTrainer(rec_cfg)
        self._model = trainer.recover(
            student_model=self._model,
            processor=self._processor,
            train_loader=loader,
            device=self.device,
            teacher_model=None,  # Let trainer load in int4
        )

    def _run_evaluation(self, n_params_original: int) -> None:
        logger.info("─── Evaluation ───")
        eval_cfg = self.config.get("evaluation", {})
        loader = build_calibration_loader(
            dataset_name=eval_cfg.get("dataset", "wikitext"),
            split=eval_cfg.get("dataset_split", "test"),
            num_samples=eval_cfg.get("num_samples", 256),
            seq_len=eval_cfg.get("seq_len", 512),
            tokenizer=self._processor,
            batch_size=1,
        )
        evaluator = Evaluator(eval_cfg)
        evaluator.evaluate(
            self._model,
            loader,
            self.device,
            n_params_original=n_params_original,
        )

    def _save_intermediate(self, tag: str) -> None:
        out_dir = Path(self.config["output"]["base_dir"]) / tag
        save_model(self._model, self._processor, str(out_dir))
        logger.info("Intermediate model saved to %s", out_dir)
