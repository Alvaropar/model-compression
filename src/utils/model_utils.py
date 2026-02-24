"""
Utilities for loading, inspecting, and saving vision-language models.

Supports:
  - InternVL3/3.5 (custom InternVLChatModel via trust_remote_code)
  - Qwen2-VL / Qwen3-VL (AutoModelForVision2Seq)
  - Standard causal LMs (AutoModelForCausalLM)
"""
from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoTokenizer,
    AutoProcessor,
)

logger = logging.getLogger(__name__)


# ── Model type detection ──────────────────────────────────────────────────────

def _detect_model_type(model_name: str) -> str:
    """Detect the model type from the model name/path and its config.

    Returns:
        One of: "internvl_custom", "vlm_hf", "causal_lm"
    """
    # Try reading the config to detect architecture
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", "")
        architectures = getattr(config, "architectures", []) or []

        # InternVL custom format (uses trust_remote_code with InternVLChatModel)
        if model_type == "internvl_chat" or "InternVLChatModel" in architectures:
            return "internvl_custom"

        # InternVL HF-native format
        if model_type == "internvl" or "InternVLForConditionalGeneration" in architectures:
            return "vlm_hf"

        # Qwen-VL models
        if any("qwen" in a.lower() and "vl" in a.lower() for a in architectures):
            return "vlm_hf"

    except Exception:
        pass

    # Heuristic fallback from model name
    lower = model_name.lower()
    if "internvl" in lower:
        return "internvl_custom"
    if any(tag in lower for tag in ("vl", "vision", "visual")):
        return "vlm_hf"

    return "causal_lm"


# ── Model loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_name: str,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    ignore_mismatched_sizes: bool = False,
) -> tuple[nn.Module, object]:
    """Load a HuggingFace model and its tokenizer/processor.

    Handles multiple model architectures:
      - InternVL custom format (InternVLChatModel via trust_remote_code)
      - HF-native VLMs (AutoModelForVision2Seq)
      - Standard causal LMs (AutoModelForCausalLM)

    Args:
        model_name: HuggingFace model ID or local path.
        dtype: torch dtype string — "bfloat16", "float16", or "float32".
        device_map: accelerate device_map argument.
        trust_remote_code: Required for custom model architectures.
        ignore_mismatched_sizes: If True, ignore size mismatches between
            config and checkpoint (useful for loading width-pruned models
            whose vision-LLM bridge layers weren't pruned).

    Returns:
        (model, processor)
    """
    torch_dtype = getattr(torch, dtype)
    model_type = _detect_model_type(model_name)

    logger.info("Loading model: %s (dtype=%s, device_map=%s, type=%s)",
                model_name, dtype, device_map, model_type)

    if model_type == "internvl_custom":
        # InternVL custom format: auto_map routes to custom InternVLChatModel
        # via AutoModel or AutoModelForCausalLM with trust_remote_code
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        logger.info("Loaded InternVL custom model: %s", type(model).__name__)

    elif model_type == "vlm_hf":
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        logger.info("Loaded HF VLM: %s", type(model).__name__)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
        )
        logger.info("Loaded causal LM: %s", type(model).__name__)

    model.eval()

    # Try processor first (VLMs), fall back to tokenizer
    try:
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        logger.info("Loaded AutoProcessor.")
    except Exception:
        try:
            processor = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
            logger.info("Loaded AutoTokenizer (no vision processor found).")
        except Exception:
            processor = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, use_fast=False
            )
            logger.info("Loaded AutoTokenizer (slow) as fallback.")

    return model, processor


def save_model(model: nn.Module, processor: object, output_dir: str) -> None:
    """Save model and processor to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info("Saving model to %s", output_path)
    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))
    logger.info("Model saved.")


def get_layer_names(
    model: nn.Module,
    patterns: list[str] | None = None,
) -> list[str]:
    """Return module names that match any of the given regex patterns.

    If patterns is None, returns all named modules.

    Args:
        model: The model to inspect.
        patterns: List of regex pattern strings.

    Returns:
        List of matching fully-qualified module name strings.
    """
    all_names = [name for name, _ in model.named_modules()]
    if patterns is None:
        return all_names

    matched: list[str] = []
    for name in all_names:
        for pat in patterns:
            if re.fullmatch(pat, name):
                matched.append(name)
                break
    return matched


def get_transformer_layers(model: nn.Module) -> list[nn.Module]:
    """Heuristically extract the list of transformer decoder layers.

    Supports:
      - InternVL custom: model.language_model.model.layers
      - Qwen2-VL / Qwen3-VL: model.model.layers
      - Generic HF causal LM: model.layers

    Args:
        model: The loaded model.

    Returns:
        List of transformer layer modules in order.
    """
    # InternVL custom format: model.language_model.model.layers
    if hasattr(model, "language_model"):
        lm = model.language_model
        if hasattr(lm, "model") and hasattr(lm.model, "layers"):
            return list(lm.model.layers)
        if hasattr(lm, "layers"):
            return list(lm.layers)

    # Qwen2-VL / Qwen3-VL: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)

    # Generic fallback
    if hasattr(model, "layers"):
        return list(model.layers)

    raise AttributeError(
        "Cannot locate transformer layers. Tried: "
        "model.language_model.model.layers, model.model.layers, model.layers"
    )


def get_llm_submodule(model: nn.Module) -> nn.Module:
    """Return the language-model submodule for VLMs, or the model itself.

    For InternVL custom: model.language_model
    For Qwen-VL / generic: model
    """
    if hasattr(model, "language_model"):
        return model.language_model
    return model


def count_parameters(model: nn.Module) -> int:
    """Return total number of parameters (trainable and non-trainable)."""
    return sum(p.numel() for p in model.parameters())


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Traverse dot-separated name to retrieve a sub-module."""
    parts = name.split(".")
    module = model
    for part in parts:
        module = getattr(module, part)
    return module


def set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a sub-module identified by its dot-separated name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)
