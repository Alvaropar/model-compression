"""
Dataset utilities for calibration and recovery training.

Calibration loaders return batches of tokenized text for forward-pass-only
profiling (SVD analysis, activation scoring, loss measurement).

Training loaders are used during the recovery / distillation phase.
"""
from __future__ import annotations

import logging
import random
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal text dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

class _TextDataset(Dataset):
    """Simple tokenized text dataset wrapping a list of input_ids tensors."""

    def __init__(self, input_ids_list: list[torch.Tensor]) -> None:
        self.input_ids_list = input_ids_list

    def __len__(self) -> int:
        return len(self.input_ids_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.input_ids_list[idx]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def _load_text_samples(
    dataset_name: str,
    split: str,
    num_samples: int,
    seq_len: int,
    tokenizer: object,
    seed: int = 42,
) -> list[torch.Tensor]:
    """Download a HuggingFace text dataset and tokenize random samples.

    Args:
        dataset_name: e.g. "c4", "wikitext".
        split: e.g. "train", "validation".
        num_samples: Number of tokenized sequences to produce.
        seq_len: Fixed token sequence length (truncate / skip shorter).
        tokenizer: HuggingFace tokenizer with __call__ interface.
        seed: Random seed for reproducibility.

    Returns:
        List of LongTensor of shape (seq_len,).
    """
    random.seed(seed)

    logger.info(
        "Loading %d calibration samples from %s[%s] (seq_len=%d)",
        num_samples, dataset_name, split, seq_len,
    )

    # c4 needs a streaming approach because it is huge
    if dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        text_iter = (sample["text"] for sample in ds)
    elif dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        texts = ds["text"]
        random.shuffle(texts)
        text_iter = iter(texts)
    else:
        # Generic: assume the dataset has a "text" column
        ds = load_dataset(dataset_name, split=split)
        texts = list(ds["text"])
        random.shuffle(texts)
        text_iter = iter(texts)

    samples: list[torch.Tensor] = []
    for text in text_iter:
        if len(samples) >= num_samples:
            break
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=seq_len,
            padding=False,
        )
        ids = enc["input_ids"].squeeze(0)  # (T,)
        if ids.shape[0] < seq_len:
            continue  # skip sequences that are too short
        samples.append(ids[:seq_len])

    if len(samples) < num_samples:
        logger.warning(
            "Only collected %d samples (requested %d). "
            "Consider a larger split or shorter seq_len.",
            len(samples), num_samples,
        )

    return samples


def build_calibration_loader(
    dataset_name: str,
    split: str,
    num_samples: int,
    seq_len: int,
    tokenizer: object,
    batch_size: int = 1,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader for profiling / calibration forward passes.

    All sequences have the same fixed length (seq_len) so collation is trivial.

    Args:
        dataset_name: HuggingFace dataset name.
        split: Dataset split.
        num_samples: How many tokenized sequences to include.
        seq_len: Token sequence length.
        tokenizer: HuggingFace tokenizer.
        batch_size: Batch size for the DataLoader.
        seed: Random seed.

    Returns:
        DataLoader yielding dicts with keys "input_ids", "attention_mask", "labels".
    """
    samples = _load_text_samples(dataset_name, split, num_samples, seq_len, tokenizer, seed)
    dataset = _TextDataset(samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def build_training_loader(
    dataset_name: str,
    split: str,
    num_samples: int,
    seq_len: int,
    tokenizer: object,
    batch_size: int = 4,
    seed: int = 42,
    num_workers: int = 4,
) -> DataLoader:
    """Build a shuffled DataLoader for the recovery training phase.

    Args:
        dataset_name: HuggingFace dataset name.
        split: Dataset split.
        num_samples: Max number of samples to use.
        seq_len: Token sequence length.
        tokenizer: HuggingFace tokenizer.
        batch_size: Training batch size per device.
        seed: Random seed.
        num_workers: DataLoader worker processes.

    Returns:
        Shuffled DataLoader for training.
    """
    samples = _load_text_samples(dataset_name, split, num_samples, seq_len, tokenizer, seed)
    dataset = _TextDataset(samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
