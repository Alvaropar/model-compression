"""
Benchmark evaluation script for comparing base vs compressed models.

Runs 4 diverse benchmarks commonly used in LLM compression papers
(Minitron, SliceGPT, ShortGPT, etc.):

  1. HellaSwag     — commonsense reasoning (accuracy, multiple-choice)
  2. ARC-Easy      — science knowledge (accuracy, multiple-choice)
  3. WikiText-2    — language modelling (perplexity)
  4. MMLU (subset) — multitask knowledge (accuracy, multiple-choice)

Saves per-sample predictions for each model alongside aggregate metrics
into outputs/evaluation/ for detailed analysis.

Usage:
  # Benchmark both base and compressed models
  python scripts/run_benchmarks.py

  # Benchmark only the base model
  python scripts/run_benchmarks.py --base-only

  # Benchmark only the compressed model
  python scripts/run_benchmarks.py --compressed-only

  # Custom paths
  python scripts/run_benchmarks.py --base-model path/to/base --compressed-model path/to/compressed
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.model_utils import load_model_and_tokenizer, get_llm_submodule, count_parameters

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    benchmark: str
    metric_name: str
    metric_value: float
    num_samples: int
    elapsed_s: float

    def __str__(self) -> str:
        return f"{self.benchmark:15s} | {self.metric_name}: {self.metric_value:8.4f} | n={self.num_samples} | {self.elapsed_s:.1f}s"


@dataclass
class ModelReport:
    model_name: str
    n_params: int
    results: list[BenchmarkResult] = field(default_factory=list)
    # Per-sample details keyed by benchmark name
    per_sample: dict[str, list[dict]] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"\n{'='*70}",
            f"Model: {self.model_name}",
            f"Parameters: {self.n_params:,}",
            f"{'='*70}",
        ]
        for r in self.results:
            lines.append(f"  {r}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: log-likelihood scoring for multiple-choice tasks
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _score_choices(
    model: nn.Module,
    tokenizer: object,
    context: str,
    choices: list[str],
    device: torch.device,
    max_len: int = 512,
) -> list[float]:
    """Score each choice by its average log-probability given the context.

    For each choice, we compute:
      score = mean log P(choice_token_i | context + choice_token_<i)

    This is the standard approach used in lm-evaluation-harness for
    multiple-choice benchmarks.
    """
    scores = []
    for choice in choices:
        # Tokenize context and full sequence
        ctx_enc = tokenizer(context, return_tensors="pt", truncation=True,
                            max_length=max_len, add_special_tokens=True)
        full_text = context + choice
        full_enc = tokenizer(full_text, return_tensors="pt", truncation=True,
                             max_length=max_len, add_special_tokens=True)

        ctx_len = ctx_enc["input_ids"].shape[1]
        full_ids = full_enc["input_ids"].to(device)
        full_mask = full_enc["attention_mask"].to(device)

        if full_ids.shape[1] <= ctx_len:
            # Choice was truncated away entirely
            scores.append(float("-inf"))
            continue

        out = model(input_ids=full_ids, attention_mask=full_mask)
        logits = out.logits  # (1, T, V)

        # Log probabilities of the choice tokens (shift by 1 for next-token prediction)
        # For tokens at positions ctx_len ... T-1, the label is at those positions
        # and the logit predicting them is at positions ctx_len-1 ... T-2
        choice_logits = logits[0, ctx_len - 1:-1, :]  # (choice_len, V)
        choice_labels = full_ids[0, ctx_len:]          # (choice_len,)

        log_probs = F.log_softmax(choice_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, choice_labels.unsqueeze(1)).squeeze(1)

        # Length-normalized log-probability
        avg_log_prob = token_log_probs.mean().item()
        scores.append(avg_log_prob)

    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 1: HellaSwag (commonsense reasoning)
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_hellaswag(
    model: nn.Module,
    tokenizer: object,
    device: torch.device,
    num_samples: int = 200,
) -> tuple[BenchmarkResult, list[dict]]:
    """Evaluate on HellaSwag — sentence completion commonsense reasoning."""
    from datasets import load_dataset

    logger.info("Loading HellaSwag dataset...")
    ds = load_dataset("Rowan/hellaswag", split="validation")

    # Limit samples
    if num_samples < len(ds):
        ds = ds.select(range(num_samples))

    model.eval()
    correct = 0
    total = 0
    per_sample = []
    t0 = time.time()

    for idx, sample in enumerate(tqdm(ds, desc="HellaSwag", leave=False)):
        ctx = sample["ctx"]
        endings = sample["endings"]  # list of 4 choices
        label = int(sample["label"])

        scores = _score_choices(model, tokenizer, ctx, endings, device)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        is_correct = pred == label

        if is_correct:
            correct += 1
        total += 1

        per_sample.append({
            "id": idx,
            "context": ctx[:200] + ("..." if len(ctx) > 200 else ""),
            "choices": endings,
            "label": label,
            "predicted": pred,
            "correct": is_correct,
            "scores": [round(s, 6) if s != float("-inf") else None for s in scores],
        })

    accuracy = correct / max(total, 1)
    elapsed = time.time() - t0
    logger.info("HellaSwag: accuracy=%.4f (%d/%d) in %.1fs", accuracy, correct, total, elapsed)

    result = BenchmarkResult(
        benchmark="HellaSwag",
        metric_name="accuracy",
        metric_value=accuracy,
        num_samples=total,
        elapsed_s=elapsed,
    )
    return result, per_sample


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 2: ARC-Easy (science knowledge)
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_arc_easy(
    model: nn.Module,
    tokenizer: object,
    device: torch.device,
    num_samples: int = 200,
) -> tuple[BenchmarkResult, list[dict]]:
    """Evaluate on ARC-Easy — elementary science questions."""
    from datasets import load_dataset

    logger.info("Loading ARC-Easy dataset...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")

    if num_samples < len(ds):
        ds = ds.select(range(num_samples))

    model.eval()
    correct = 0
    total = 0
    per_sample = []
    t0 = time.time()

    for idx, sample in enumerate(tqdm(ds, desc="ARC-Easy", leave=False)):
        question = sample["question"]
        choices_text = sample["choices"]["text"]
        choices_labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]

        # Build context: "Question: ...\nAnswer:"
        ctx = f"Question: {question}\nAnswer:"
        # Choices are the answer texts
        choice_strs = [f" {c}" for c in choices_text]

        scores = _score_choices(model, tokenizer, ctx, choice_strs, device)
        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        pred_label = choices_labels[pred_idx]
        is_correct = pred_label == answer_key

        if is_correct:
            correct += 1
        total += 1

        per_sample.append({
            "id": idx,
            "question": question,
            "choices": dict(zip(choices_labels, choices_text)),
            "answer_key": answer_key,
            "predicted_key": pred_label,
            "correct": is_correct,
            "scores": {lbl: round(s, 6) if s != float("-inf") else None
                       for lbl, s in zip(choices_labels, scores)},
        })

    accuracy = correct / max(total, 1)
    elapsed = time.time() - t0
    logger.info("ARC-Easy: accuracy=%.4f (%d/%d) in %.1fs", accuracy, correct, total, elapsed)

    result = BenchmarkResult(
        benchmark="ARC-Easy",
        metric_name="accuracy",
        metric_value=accuracy,
        num_samples=total,
        elapsed_s=elapsed,
    )
    return result, per_sample


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 3: WikiText-2 (perplexity)
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_wikitext2(
    model: nn.Module,
    tokenizer: object,
    device: torch.device,
    max_samples: int = 200,
    seq_len: int = 512,
) -> tuple[BenchmarkResult, list[dict]]:
    """Evaluate perplexity on WikiText-2 test set."""
    from datasets import load_dataset
    import math

    logger.info("Loading WikiText-2 dataset...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    # Concatenate all text and tokenize
    all_text = "\n\n".join([t for t in ds["text"] if t.strip()])
    encodings = tokenizer(all_text, return_tensors="pt", truncation=False)
    input_ids = encodings["input_ids"][0]  # (total_tokens,)

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    n_chunks = 0
    per_sample = []
    t0 = time.time()

    # Process in chunks of seq_len with stride seq_len//2 for overlap
    stride = seq_len // 2
    for begin in tqdm(range(0, len(input_ids) - seq_len, stride), desc="WikiText-2 PPL", leave=False):
        if n_chunks >= max_samples:
            break

        end = begin + seq_len
        chunk = input_ids[begin:end].unsqueeze(0).to(device)
        target = chunk.clone()

        # Only compute loss on the non-overlapping part (second half)
        # except for the first chunk
        if begin > 0:
            target[0, :stride] = -100

        with torch.no_grad():
            out = model(input_ids=chunk, labels=target)

        n_valid = (target != -100).sum().item()
        chunk_nll = out.loss.item() * n_valid
        chunk_ppl = math.exp(out.loss.item()) if out.loss.item() < 100 else float("inf")
        total_nll += chunk_nll
        total_tokens += n_valid
        n_chunks += 1

        # Decode a short snippet of the chunk for context
        snippet_tokens = input_ids[begin:min(begin + 50, end)]
        snippet_text = tokenizer.decode(snippet_tokens, skip_special_tokens=True)

        per_sample.append({
            "chunk_id": n_chunks - 1,
            "token_offset": begin,
            "n_scored_tokens": n_valid,
            "chunk_loss": round(out.loss.item(), 6),
            "chunk_perplexity": round(chunk_ppl, 4) if chunk_ppl != float("inf") else "inf",
            "text_snippet": snippet_text[:150] + ("..." if len(snippet_text) > 150 else ""),
        })

    ppl = torch.exp(torch.tensor(total_nll / max(total_tokens, 1))).item()
    elapsed = time.time() - t0
    logger.info("WikiText-2: perplexity=%.4f (tokens=%d, chunks=%d) in %.1fs",
                ppl, total_tokens, n_chunks, elapsed)

    result = BenchmarkResult(
        benchmark="WikiText-2",
        metric_name="perplexity",
        metric_value=ppl,
        num_samples=n_chunks,
        elapsed_s=elapsed,
    )
    return result, per_sample


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark 4: MMLU subset (multitask knowledge)
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_mmlu(
    model: nn.Module,
    tokenizer: object,
    device: torch.device,
    num_samples: int = 200,
    subjects: Optional[list[str]] = None,
) -> tuple[BenchmarkResult, list[dict]]:
    """Evaluate on MMLU — multitask multiple-choice knowledge benchmark.

    Uses a diverse subset of subjects for a representative score.
    """
    from datasets import load_dataset

    if subjects is None:
        # 5 diverse subjects covering STEM, humanities, social science, other
        subjects = [
            "abstract_algebra",
            "global_facts",
            "high_school_geography",
            "human_aging",
            "professional_medicine",
        ]

    logger.info("Loading MMLU dataset (subjects: %s)...", subjects)

    all_samples = []
    for subject in subjects:
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
            for sample in ds:
                sample["_subject"] = subject
                all_samples.append(sample)
        except Exception as e:
            logger.warning("Failed to load MMLU subject '%s': %s", subject, e)

    if not all_samples:
        logger.error("No MMLU samples loaded.")
        return BenchmarkResult("MMLU", "accuracy", 0.0, 0, 0.0), []

    # Limit samples
    if num_samples < len(all_samples):
        all_samples = all_samples[:num_samples]

    model.eval()
    correct = 0
    total = 0
    per_sample = []
    t0 = time.time()

    choice_letters = ["A", "B", "C", "D"]
    for idx, sample in enumerate(tqdm(all_samples, desc="MMLU", leave=False)):
        question = sample["question"]
        choices = sample["choices"]  # list of 4 strings
        answer_idx = int(sample["answer"])  # 0-3
        subject = sample.get("_subject", "unknown")

        # Build prompt in standard MMLU format
        ctx = f"Question: {question}\n"
        for i, c in enumerate(choices):
            ctx += f"{choice_letters[i]}. {c}\n"
        ctx += "Answer:"

        choice_strs = [f" {letter}" for letter in choice_letters[:len(choices)]]
        scores = _score_choices(model, tokenizer, ctx, choice_strs, device)
        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        is_correct = pred_idx == answer_idx

        if is_correct:
            correct += 1
        total += 1

        per_sample.append({
            "id": idx,
            "subject": subject,
            "question": question,
            "choices": dict(zip(choice_letters[:len(choices)], choices)),
            "answer": choice_letters[answer_idx],
            "predicted": choice_letters[pred_idx],
            "correct": is_correct,
            "scores": {letter: round(s, 6) if s != float("-inf") else None
                       for letter, s in zip(choice_letters, scores)},
        })

    accuracy = correct / max(total, 1)
    elapsed = time.time() - t0
    logger.info("MMLU: accuracy=%.4f (%d/%d) in %.1fs", accuracy, correct, total, elapsed)

    result = BenchmarkResult(
        benchmark="MMLU",
        metric_name="accuracy",
        metric_value=accuracy,
        num_samples=total,
        elapsed_s=elapsed,
    )
    return result, per_sample


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_all_benchmarks(
    model: nn.Module,
    tokenizer: object,
    device: torch.device,
    model_name: str,
    num_samples: int = 200,
) -> ModelReport:
    """Run all 4 benchmarks on the given model."""
    llm = get_llm_submodule(model)
    n_params = count_parameters(model)
    report = ModelReport(model_name=model_name, n_params=n_params)

    logger.info("Running benchmarks on: %s (%d params)", model_name, n_params)

    # 1. HellaSwag
    try:
        r, samples = benchmark_hellaswag(llm, tokenizer, device, num_samples=num_samples)
        report.results.append(r)
        report.per_sample["HellaSwag"] = samples
    except Exception as e:
        logger.error("HellaSwag failed: %s", e)

    # 2. ARC-Easy
    try:
        r, samples = benchmark_arc_easy(llm, tokenizer, device, num_samples=num_samples)
        report.results.append(r)
        report.per_sample["ARC-Easy"] = samples
    except Exception as e:
        logger.error("ARC-Easy failed: %s", e)

    # 3. WikiText-2 Perplexity
    try:
        r, samples = benchmark_wikitext2(llm, tokenizer, device, max_samples=num_samples, seq_len=512)
        report.results.append(r)
        report.per_sample["WikiText-2"] = samples
    except Exception as e:
        logger.error("WikiText-2 failed: %s", e)

    # 4. MMLU
    try:
        r, samples = benchmark_mmlu(llm, tokenizer, device, num_samples=num_samples)
        report.results.append(r)
        report.per_sample["MMLU"] = samples
    except Exception as e:
        logger.error("MMLU failed: %s", e)

    return report


def print_comparison(base_report: Optional[ModelReport], compressed_report: Optional[ModelReport]) -> None:
    """Print a side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    if base_report:
        print(f"\nBase model:       {base_report.model_name}")
        print(f"  Parameters:     {base_report.n_params:,}")
    if compressed_report:
        print(f"\nCompressed model: {compressed_report.model_name}")
        print(f"  Parameters:     {compressed_report.n_params:,}")

    if base_report and compressed_report:
        ratio = base_report.n_params / max(compressed_report.n_params, 1)
        pct = 100.0 * compressed_report.n_params / base_report.n_params
        print(f"\n  Compression:    {ratio:.2f}x ({pct:.1f}% of original)")

    print(f"\n{'Benchmark':15s} | {'Metric':12s} | {'Base':>10s} | {'Compressed':>10s} | {'Delta':>10s}")
    print("-" * 70)

    base_map = {r.benchmark: r for r in (base_report.results if base_report else [])}
    comp_map = {r.benchmark: r for r in (compressed_report.results if compressed_report else [])}

    all_benchmarks = list(dict.fromkeys(
        [r.benchmark for r in (base_report.results if base_report else [])] +
        [r.benchmark for r in (compressed_report.results if compressed_report else [])]
    ))

    for bm in all_benchmarks:
        b = base_map.get(bm)
        c = comp_map.get(bm)
        metric = (b or c).metric_name if (b or c) else "?"

        b_val = f"{b.metric_value:.4f}" if b else "N/A"
        c_val = f"{c.metric_value:.4f}" if c else "N/A"

        if b and c:
            delta = c.metric_value - b.metric_value
            # For perplexity, lower is better (negative delta is good)
            # For accuracy, higher is better (positive delta is good)
            sign = "+" if delta > 0 else ""
            delta_str = f"{sign}{delta:.4f}"
        else:
            delta_str = "N/A"

        print(f"{bm:15s} | {metric:12s} | {b_val:>10s} | {c_val:>10s} | {delta_str:>10s}")

    print("=" * 80)
    print("Note: For accuracy benchmarks, higher is better.")
    print("      For perplexity, lower is better.")
    print("=" * 80)


def save_per_sample_outputs(
    output_dir: Path,
    base_report: Optional[ModelReport],
    compressed_report: Optional[ModelReport],
) -> None:
    """Save per-sample outputs for each model and benchmark.

    Directory layout:
      outputs/evaluation/
        benchmark_results.json          — aggregate metrics + comparison
        base_model/
          hellaswag_samples.json        — per-sample predictions
          arc_easy_samples.json
          wikitext2_samples.json
          mmlu_samples.json
        compressed_model/
          hellaswag_samples.json
          arc_easy_samples.json
          wikitext2_samples.json
          mmlu_samples.json
    """
    benchmark_to_filename = {
        "HellaSwag": "hellaswag_samples.json",
        "ARC-Easy": "arc_easy_samples.json",
        "WikiText-2": "wikitext2_samples.json",
        "MMLU": "mmlu_samples.json",
    }

    for label, report in [("base_model", base_report), ("compressed_model", compressed_report)]:
        if report is None:
            continue
        model_dir = output_dir / label
        model_dir.mkdir(parents=True, exist_ok=True)

        for bm_name, samples in report.per_sample.items():
            filename = benchmark_to_filename.get(bm_name, f"{bm_name.lower()}_samples.json")
            path = model_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "benchmark": bm_name,
                    "model": report.model_name,
                    "n_params": report.n_params,
                    "num_samples": len(samples),
                    "samples": samples,
                }, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d per-sample results to %s", len(samples), path)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark base vs compressed models")
    parser.add_argument("--base-model", type=str,
                        default=r"C:\Projects\visual-llm-finance\models\InternVL3_5-1B",
                        help="Path to base (original) model")
    parser.add_argument("--compressed-model", type=str,
                        default=r"outputs\after_depth_pruning",
                        help="Path to compressed model")
    parser.add_argument("--base-only", action="store_true",
                        help="Only benchmark the base model")
    parser.add_argument("--compressed-only", action="store_true",
                        help="Only benchmark the compressed model")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of samples per benchmark (default: 200)")
    parser.add_argument("--output-dir", type=str, default="outputs/evaluation",
                        help="Directory for all evaluation outputs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    base_report = None
    compressed_report = None

    # ── Benchmark base model ──────────────────────────────────────────────────
    if not args.compressed_only:
        logger.info("=" * 70)
        logger.info("Loading BASE model: %s", args.base_model)
        logger.info("=" * 70)
        model, tokenizer = load_model_and_tokenizer(
            args.base_model, dtype="bfloat16", device_map="auto"
        )
        base_report = run_all_benchmarks(
            model, tokenizer, device, model_name=args.base_model,
            num_samples=args.num_samples,
        )
        print(base_report.summary())

        # Free GPU memory before loading compressed model
        del model
        torch.cuda.empty_cache()

    # ── Benchmark compressed model ────────────────────────────────────────────
    if not args.base_only:
        logger.info("=" * 70)
        logger.info("Loading COMPRESSED model: %s", args.compressed_model)
        logger.info("=" * 70)
        model, tokenizer = load_model_and_tokenizer(
            args.compressed_model, dtype="bfloat16", device_map="auto",
            ignore_mismatched_sizes=True,  # Bridge layers may not match pruned LLM dims
        )
        compressed_report = run_all_benchmarks(
            model, tokenizer, device, model_name=args.compressed_model,
            num_samples=args.num_samples,
        )
        print(compressed_report.summary())

        del model
        torch.cuda.empty_cache()

    # ── Comparison ────────────────────────────────────────────────────────────
    if base_report or compressed_report:
        print_comparison(base_report, compressed_report)

    # ── Save all results ──────────────────────────────────────────────────────
    # 1. Aggregate benchmark_results.json
    results_path = output_dir / "benchmark_results.json"
    results = {}
    if base_report:
        results["base"] = {
            "model_name": base_report.model_name,
            "n_params": base_report.n_params,
            "results": [asdict(r) for r in base_report.results],
        }
    if compressed_report:
        results["compressed"] = {
            "model_name": compressed_report.model_name,
            "n_params": compressed_report.n_params,
            "results": [asdict(r) for r in compressed_report.results],
        }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Aggregate results saved to %s", results_path)

    # 2. Per-sample outputs for each model
    save_per_sample_outputs(output_dir, base_report, compressed_report)

    logger.info("All evaluation outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
