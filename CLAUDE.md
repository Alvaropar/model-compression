# Model Compression Pipeline

## Project Overview
4-phase compression pipeline for vision-language models.
Current target: **InternVL3.5-1B** (InternViT-300M + Qwen3-0.6B).
1. SVD Profiling (standard + TNrSVD from arXiv:1707.07803)
2. Width Pruning (Minitron-style activation + SV importance)
3. Depth Pruning (sliding window contiguous layer removal)
4. Recovery (knowledge distillation / fine-tuning)

## Supported Architectures
- **InternVL custom** (`InternVLChatModel` via `trust_remote_code`) — loaded with `AutoModel`
- **Qwen-VL** (`AutoModelForVision2Seq`)
- **Standard causal LMs** (`AutoModelForCausalLM`)

## Key Architecture Details (InternVL3.5-1B)
- Model class: `InternVLChatModel` (custom, needs `trust_remote_code=True`)
- LLM backbone: Qwen3-0.6B (`hidden_size=1024`, `intermediate_size=3072`, `num_heads=16`, `num_kv_heads=8`, 28 layers)
- Vision encoder: InternViT-300M (24 layers, `hidden_size=1024`)
- Transformer layers at `model.language_model.model.layers`
- LLM config nested at `model.config.llm_config`
- GQA (Grouped Query Attention): `num_key_value_heads=8` ≠ `num_attention_heads=16`
- All SVD computations done in fp32 for numerical stability; model kept in bf16

## Running
```bash
# Full pipeline
python scripts/run_pipeline.py --config configs/compression_config.yaml

# With evaluation
python scripts/run_pipeline.py --config configs/compression_config.yaml --eval

# Skip phases
python scripts/run_pipeline.py --skip svd width
```

## Code Conventions
- All config in `configs/compression_config.yaml` (no hardcoded hyperparams)
- Each phase is a self-contained class in `src/compression/`
- Use PyTorch forward hooks for non-invasive activation collection
- Calibration data from HuggingFace `c4` or `wikitext` datasets
- Recovery uses plain PyTorch training loop (no HF Trainer dependency)
- Architecture-aware: use `get_llm_submodule()`, `get_transformer_layers()` to abstract model structure

## Testing Notes
- Run from project root (scripts add parent to sys.path)
- Requires CUDA for practical execution; CPU works but is slow
- The c4 dataset requires streaming (very large)
