# Qwen3-VL Model Compression Pipeline

A structured compression pipeline for Qwen3-VL-8B-Instruct (and compatible Qwen2-VL / Qwen3-VL models) combining SVD profiling, width pruning, depth pruning, and knowledge-distillation recovery.

---

## Pipeline Overview

```
Original Model (8B)
       │
       ▼
┌─────────────────────────────────────────────┐
│  Phase 1 — SVD Profiling                   │
│  • Standard truncated SVD on all linear     │
│    layers (attn + MLP)                      │
│  • TNrSVD — fast randomised SVD             │
│    (arXiv:1707.07803) run in parallel;      │
│    best decomposition is kept               │
│  • Output: TensorNetwork of MPO layers      │
└─────────────────┬───────────────────────────┘
                  │ TensorNetwork (singular value spectra)
                  ▼
┌─────────────────────────────────────────────┐
│  Phase 2 — Width Pruning                   │
│  • 1024 calibration forward passes          │
│  • Per-neuron activation magnitude scores   │
│    (L2 over batch, mean over sequence)      │
│  • Combined importance = SV importance +    │
│    activation importance (from TensorNet)   │
│  • Prune MLP intermediate dim (~35%)        │
│  • Prune global hidden dim (~25%)           │
└─────────────────┬───────────────────────────┘
                  │ Width-pruned model
                  ▼
┌─────────────────────────────────────────────┐
│  Phase 3 — Depth Pruning                   │
│  • Block Importance (BI) scores per layer   │
│    BI = 1 − cos_sim(layer_in, layer_out)    │
│  • Sliding window search [min_size..max_size]│
│    over contiguous layer blocks              │
│  • Evaluate Δloss for each window           │
│  • Select window with Δloss ≤ threshold     │
│    closest to target_removal size (~3 layers)│
│  • Physically remove selected layers        │
└─────────────────┬───────────────────────────┘
                  │ Depth + Width pruned model
                  ▼
┌─────────────────────────────────────────────┐
│  Phase 4 — Recovery                        │
│  • Knowledge Distillation (primary):        │
│    forward KL divergence on teacher logits  │
│    (50× more efficient than standard SFT)   │
│  • Optional: supervised fine-tuning (SFT)   │
│  • Cosine LR schedule with linear warmup    │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
       Compressed Model (~4–5B)
```

---

## Installation

```bash
pip install -r requirements.txt
```

For GPU training you need CUDA ≥ 12.1 and PyTorch ≥ 2.3 with CUDA support.

---

## Quick Start

```bash
# Full pipeline (all 4 phases)
python scripts/run_pipeline.py --config configs/compression_config.yaml

# Profiling + pruning only (skip recovery for now)
python scripts/run_pipeline.py --config configs/compression_config.yaml --skip recovery

# Resume from depth pruning onwards (SVD already cached on disk)
python scripts/run_pipeline.py --config configs/compression_config.yaml --skip svd width

# Verbose debug output
python scripts/run_pipeline.py --config configs/compression_config.yaml --log-level DEBUG
```

---

## Configuration

All parameters live in `configs/compression_config.yaml`.  Key sections:

| Section | Key parameters |
|---------|----------------|
| `model` | `name` — HuggingFace model ID or local path |
| `svd` | `target_layers`, `rank_ratio`, TNrSVD `k`/`q` |
| `width_pruning` | `pruning_ratios.hidden_size`, `pruning_ratios.intermediate_size` |
| `depth_pruning` | `sliding_window.min_size/max_size`, `loss_threshold`, `target_layer_removal` |
| `recovery` | `method` (`distillation`/`finetuning`/`both`), `training.*` |

---

## Methods

### Phase 1 — SVD Profiling

Each linear weight matrix **W ∈ ℝ^{m×n}** is decomposed as:

```
W ≈ U Σ Vᵀ   (rank-r approximation)
```

Two methods are compared; the lower-error decomposition is kept:

**Standard truncated SVD** — `torch.linalg.svd` (economy), keep top-r singular triplets.

**TNrSVD** (arXiv:1707.07803) — Randomised subspace iteration:
1. Draw random sketch Ω ∈ ℝ^{n×k}
2. Power iteration: Y = (AAᵀ)^q AΩ (re-orthogonalised at each step)
3. QR factorisation: Q = qr(Y)
4. Project: B = QᵀA ∈ ℝ^{k×n}
5. Thin SVD of B: B = Û Σ Vᵀ → recover U = QÛ
6. Return top k/2 singular triplets

The adaptive variant `qTNrSVD` doubles `k` until the relative reconstruction error drops below `rel_error_tol`.

Results are stored as a **TensorNetwork** of **MPOLayer** objects (U, S, Vt per layer).

### Phase 2 — Width Pruning

Per-neuron importance score (Minitron, arXiv:2408.11796):

```
importance_c = sqrt( mean_B( sum_T( activation[b,t,c]^2 ) ) )
```

Combined with singular-value importance from the TensorNetwork:

```
importance_sv_i = || U[i,:] ⊙ S ||_2
combined = 0.5 × norm(importance_sv) + 0.5 × norm(importance_act)
```

Lowest-scoring neurons are removed.  MLP gate/up/down projections are pruned consistently.  Hidden-dim pruning applies a global mask derived from attention output (`o_proj`) importances.

### Phase 3 — Depth Pruning

Block Importance (BI) score per layer:

```
BI_i = 1 − cosine_similarity(input_i, output_i)
```

Layers with BI ≈ 0 are near-identity and prime candidates for removal.

Sliding window search:
- For each window (start=i, size=w): temporarily replace layers with identity skip, measure Δloss.
- Select window where `Δloss ≤ loss_threshold` and size ≈ `target_layer_removal`.
- Physically remove selected layers and update `model.config.num_hidden_layers`.

Key insight from Minitron: **contiguous** layer removal consistently outperforms non-contiguous importance-based selection.

### Phase 4 — Recovery

**Distillation loss** (primary):
```
L = KL( softmax(z_teacher/T) ‖ softmax(z_student/T) ) × T²
```

**Fine-tuning loss** (optional):
```
L = CrossEntropy( logits[:, :-1], labels[:, 1:] )
```

Training uses AdamW + cosine LR schedule with linear warmup, gradient accumulation, and bf16 mixed precision.

---

## Project Structure

```
model-compression/
├── configs/
│   └── compression_config.yaml    # All hyperparameters
├── scripts/
│   └── run_pipeline.py            # CLI entry point
├── src/
│   ├── pipeline.py                # Pipeline orchestrator
│   ├── compression/
│   │   ├── svd_profiler.py        # Phase 1: Standard + TNrSVD
│   │   ├── width_pruner.py        # Phase 2: Activation-based width pruning
│   │   ├── depth_pruner.py        # Phase 3: Sliding window depth pruning
│   │   └── recovery.py            # Phase 4: Distillation + fine-tuning
│   └── utils/
│       ├── model_utils.py         # Model loading, saving, layer access
│       ├── data_utils.py          # Calibration and training data loaders
│       └── tensor_network.py      # MPOLayer and TensorNetwork containers
├── outputs/                       # Created at runtime
│   ├── svd_profiles/
│   ├── width_pruning/
│   ├── depth_pruning/
│   ├── recovery/
│   └── compressed_model/
└── requirements.txt
```

---

## References

- **TNrSVD** — Batselier & Wong, *Computing Low-Rank Approximations of Large-Scale Matrices with the Tensor Network Randomized SVD*, 2017. [arXiv:1707.07803](https://arxiv.org/abs/1707.07803)
- **Minitron** — Muralidharan et al., *LLM Pruning and Distillation in Practice: The Minitron Approach*, 2024. [arXiv:2408.11796](https://arxiv.org/abs/2408.11796)
- **Randomised SVD** — Halko, Martinsson & Tropp, *Finding Structure with Randomness*, SIAM Review 2011.
