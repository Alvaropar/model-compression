"""
Phase 2 — Width Pruning via Tensor Network Activation Analysis.

Strategy (Minitron-inspired, arXiv:2408.11796):
  1. Run `num_samples` calibration forward passes through the model with hooks
     that capture the output activations of each target linear layer.
  2. Compute per-neuron importance = L2-norm over batch × mean over sequence of
     the captured activation tensor.
  3. Store these activation importances into the TensorNetwork (MPOLayer objects)
     from Phase 1, then compute a combined importance score that blends:
       • Singular-value importance  (how much a neuron participates in high-
         variance directions of the weight space)
       • Activation magnitude importance  (how actively the neuron fires)
  4. For each prunable dimension (MLP intermediate, hidden), rank neurons by
     combined importance and zero-out / remove the bottom-k.

Width pruning targets:
  • MLP intermediate dimension: prune output neurons of gate_proj/up_proj and
    corresponding input neurons of down_proj.
  • Hidden size: prune output neurons of all projection layers simultaneously
    (requires consistent masking across q/k/v/o and mlp projections at a
    given layer — complex; the simpler approach is to use a global mask based
    on the embedding channel importance).

For the hidden dimension we follow the Minitron approach of using LayerNorm
output channel importance and applying a single global mask.

After building importance masks the pruner rewrites the model's Linear layer
weights in-place, reducing their shape.  No copy of the full model is made.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.tensor_network import TensorNetwork
from src.utils.model_utils import get_module_by_name, get_transformer_layers, get_llm_submodule

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Activation hook helpers
# ──────────────────────────────────────────────────────────────────────────────

class _ActivationCollector:
    """Forward hook that accumulates activation importance statistics.

    For a linear layer with output (B, T, C):
      importance_c += ||activations[:, :, c]||_2  (L2 over batch)
    and then averaged over sequence positions:
      importance_c  = mean_T( ||activations[:, t, c]||_2 )

    We accumulate a running L2 norm sum over batches to handle large
    calibration sets without storing everything in memory.
    """

    def __init__(self) -> None:
        self.importance: Optional[torch.Tensor] = None
        self.n_batches: int = 0

    def hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        # output shape: (B, T, C) or (B, C)
        act = output.detach().float()
        if act.ndim == 3:
            # L2 over batch, mean over seq
            # ||act[:, :, c]||_2 = sqrt(sum_b sum_t act[b, t, c]^2)
            # We store sum_b sum_t act^2 and take sqrt at the end
            score = act.pow(2).sum(dim=0).mean(dim=0)   # (C,)
        elif act.ndim == 2:
            score = act.pow(2).mean(dim=0)               # (C,)
        else:
            return

        if self.importance is None:
            self.importance = score.cpu()
        else:
            self.importance = self.importance + score.cpu()
        self.n_batches += 1

    def finalise(self) -> torch.Tensor:
        """Return per-channel importance (L2 norm, averaged over batches)."""
        if self.importance is None:
            raise RuntimeError("No activations collected.")
        # sqrt to get L2 norm (we accumulated squared values)
        return (self.importance / self.n_batches).sqrt()


# ──────────────────────────────────────────────────────────────────────────────
# Low-level pruning utilities
# ──────────────────────────────────────────────────────────────────────────────

def _prune_linear_output(linear: nn.Linear, keep_indices: torch.Tensor) -> nn.Linear:
    """Return a new Linear whose output dimension is pruned to keep_indices.

    Handles weight, optional bias, and optional lora/adapter attributes
    (ignored — base weight only).

    Args:
        linear: Original nn.Linear.
        keep_indices: 1-D int tensor, indices of output neurons to keep.

    Returns:
        New nn.Linear with out_features = len(keep_indices).
    """
    new_out = len(keep_indices)
    new_linear = nn.Linear(linear.in_features, new_out, bias=linear.bias is not None)
    new_linear.weight = nn.Parameter(linear.weight.data[keep_indices, :].clone())
    if linear.bias is not None:
        new_linear.bias = nn.Parameter(linear.bias.data[keep_indices].clone())
    new_linear.to(linear.weight.device)
    return new_linear


def _prune_linear_input(linear: nn.Linear, keep_indices: torch.Tensor) -> nn.Linear:
    """Return a new Linear whose input dimension is pruned to keep_indices."""
    new_in = len(keep_indices)
    new_linear = nn.Linear(new_in, linear.out_features, bias=linear.bias is not None)
    new_linear.weight = nn.Parameter(linear.weight.data[:, keep_indices].clone())
    if linear.bias is not None:
        new_linear.bias = nn.Parameter(linear.bias.data.clone())
    new_linear.to(linear.weight.device)
    return new_linear


def _importance_to_keep_mask(
    importance: torch.Tensor,
    prune_ratio: float,
    min_keep: int = 8,
) -> torch.Tensor:
    """Select indices to keep by discarding the lowest-importance neurons.

    Args:
        importance: 1-D tensor of per-neuron importance scores.
        prune_ratio: Fraction of neurons to REMOVE.
        min_keep: Always keep at least this many neurons.

    Returns:
        1-D int64 tensor of indices to keep, sorted ascending.
    """
    n = importance.shape[0]
    n_keep = max(min_keep, int(n * (1.0 - prune_ratio)))
    _, indices = torch.topk(importance, k=n_keep, largest=True, sorted=True)
    return indices.sort().values


# ──────────────────────────────────────────────────────────────────────────────
# WidthPruner
# ──────────────────────────────────────────────────────────────────────────────

class WidthPruner:
    """Prunes the width (hidden / intermediate dimensions) of transformer layers.

    Pipeline:
      1. register_hooks() on all target linear modules.
      2. Run calibration forward passes → collect activation statistics.
      3. Store per-neuron activation importance into the TensorNetwork.
      4. Compute combined importance (SV + activation).
      5. Build per-layer prune masks.
      6. Apply masks: rewrite Linear weights in-place.

    Args:
        config: The "width_pruning" sub-dict from compression_config.yaml.
        tensor_network: TensorNetwork from Phase 1.
    """

    def __init__(self, config: dict, tensor_network: TensorNetwork) -> None:
        self.config = config
        self.tn = tensor_network
        self.prune_hidden: bool = config["targets"].get("hidden_size", True)
        self.prune_intermediate: bool = config["targets"].get("intermediate_size", True)
        self.prune_heads: bool = config["targets"].get("num_heads", False)
        self.ratio_hidden: float = config["pruning_ratios"].get("hidden_size", 0.25)
        self.ratio_intermediate: float = config["pruning_ratios"].get("intermediate_size", 0.35)
        self.ratio_heads: float = config["pruning_ratios"].get("num_heads", 0.25)
        self.output_dir = Path(config.get("output_dir", "outputs/width_pruning"))

        # Filled during _collect_activations
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._collectors: dict[str, _ActivationCollector] = {}

    # ── public API ─────────────────────────────────────────────────────────────

    def prune(
        self,
        model: nn.Module,
        calibration_loader: DataLoader,
        device: torch.device,
    ) -> nn.Module:
        """Run the full width pruning pipeline on `model`.

        Args:
            model: The model to prune (modified in-place and returned).
            calibration_loader: DataLoader yielding calibration batches.
            device: Compute device.

        Returns:
            Pruned model.
        """
        logger.info("=== Width Pruning — Phase 2 ===")

        # ① Collect activations via hooks
        self._register_hooks(model)
        self._collect_activations(model, calibration_loader, device)
        self._remove_hooks()

        # ② Inject activation importance into TensorNetwork MPO layers
        self._update_tensor_network()

        # ③ Compute combined importance & derive keep masks
        importance_map = self.tn.compute_combined_importance(sv_weight=0.5, act_weight=0.5)

        # ④ Apply pruning
        transformer_layers = get_transformer_layers(model)
        self._prune_mlp_intermediate(model, transformer_layers, importance_map)
        if self.prune_heads:
            self._prune_attention_heads(model, transformer_layers, importance_map)
        if self.prune_hidden:
            self._prune_hidden_dim(model, transformer_layers, importance_map)

        # ⑤ Update model config to reflect pruned dimensions
        self._update_config_dimensions(model, transformer_layers)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Width pruning complete. Model parameter count: %d",
                    sum(p.numel() for p in model.parameters()))
        return model

    # ── hook management ────────────────────────────────────────────────────────

    def _register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on all linear modules that appear in the TN."""
        tn_names = set(self.tn.all_names())
        for name, module in model.named_modules():
            if name in tn_names and isinstance(module, nn.Linear):
                collector = _ActivationCollector()
                handle = module.register_forward_hook(collector.hook)
                self._hooks.append(handle)
                self._collectors[name] = collector
        logger.info("Registered %d activation hooks.", len(self._hooks))

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def _collect_activations(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> None:
        model.eval()
        # For VLMs like InternVL, the full model.forward() requires pixel_values.
        # Since we only need LLM backbone activations for width pruning, call
        # the language model submodule directly (which accepts input_ids only).
        llm = get_llm_submodule(model)
        logger.info("Collecting activations over %d batches (using LLM submodule: %s)...",
                     len(loader), type(llm).__name__)
        for batch in tqdm(loader, desc="Activation collection", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            llm(input_ids=input_ids, attention_mask=attention_mask)

    def _update_tensor_network(self) -> None:
        """Write finalised activation importances into the MPOLayer objects."""
        for name, collector in self._collectors.items():
            if name in self.tn.layers:
                self.tn.layers[name].activation_importance = collector.finalise()

    # ── MLP intermediate pruning ───────────────────────────────────────────────

    def _prune_mlp_intermediate(
        self,
        model: nn.Module,
        transformer_layers: list[nn.Module],
        importance_map: dict[str, torch.Tensor],
    ) -> None:
        """Prune the intermediate (hidden) dimension of each MLP block.

        In a Qwen-style MLP:
          x_gate = gate_proj(x)   # (B, T, intermediate)
          x_up   = up_proj(x)     # (B, T, intermediate)
          x      = down_proj(silu(x_gate) * x_up)

        We prune output neurons of gate_proj and up_proj jointly using the
        combined importance of gate_proj (more informative, has SiLU gate).
        The corresponding input neurons of down_proj are pruned to match.
        """
        if not self.prune_intermediate:
            return

        for layer_idx, layer in enumerate(tqdm(transformer_layers, desc="MLP pruning")):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue

            gate_proj: Optional[nn.Linear] = getattr(mlp, "gate_proj", None)
            up_proj: Optional[nn.Linear] = getattr(mlp, "up_proj", None)
            down_proj: Optional[nn.Linear] = getattr(mlp, "down_proj", None)

            if gate_proj is None or up_proj is None or down_proj is None:
                logger.debug("Layer %d has no standard MLP projections, skipping.", layer_idx)
                continue

            # Get importance for gate_proj output neurons
            gate_name = self._find_module_name(model, gate_proj)
            up_name = self._find_module_name(model, up_proj)

            gate_imp = importance_map.get(gate_name)
            up_imp = importance_map.get(up_name)

            if gate_imp is None and up_imp is None:
                logger.debug("No importance for MLP at layer %d, skipping.", layer_idx)
                continue

            # Use average of available importances
            if gate_imp is not None and up_imp is not None:
                combined = (gate_imp + up_imp) / 2
            else:
                combined = gate_imp if gate_imp is not None else up_imp

            keep_idx = _importance_to_keep_mask(combined, self.ratio_intermediate)

            # Prune gate_proj output
            setattr(mlp, "gate_proj", _prune_linear_output(gate_proj, keep_idx))
            # Prune up_proj output
            setattr(mlp, "up_proj", _prune_linear_output(up_proj, keep_idx))
            # Prune down_proj input to match
            setattr(mlp, "down_proj", _prune_linear_input(down_proj, keep_idx))

            logger.debug(
                "Layer %d MLP: intermediate %d → %d",
                layer_idx, len(combined), len(keep_idx),
            )

    # ── Attention head pruning ────────────────────────────────────────────────

    def _prune_attention_heads(
        self,
        model: nn.Module,
        transformer_layers: list[nn.Module],
        importance_map: dict[str, torch.Tensor],
    ) -> None:
        """Prune attention heads based on per-head importance scores.

        For each attention layer, the q/k/v projections are organized as
        (num_heads * head_dim, hidden_size).  We reshape importance scores
        into (num_heads, head_dim), compute per-head importance as the mean
        over head_dim, then prune the least important heads.

        This updates q_proj, k_proj, v_proj, o_proj weights and adjusts
        the model config's num_attention_heads (and num_key_value_heads for GQA).
        """
        if not self.prune_heads:
            return

        top_cfg = getattr(model, "config", None)
        if top_cfg is None:
            logger.warning("No model config found; skipping head pruning.")
            return
        # For InternVL: LLM config is nested under llm_config
        cfg = getattr(top_cfg, "llm_config", top_cfg)

        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        head_dim = getattr(cfg, "head_dim", None)
        if head_dim is None:
            hidden_size = getattr(cfg, "hidden_size", None)
            if hidden_size is not None and num_heads is not None:
                head_dim = hidden_size // num_heads
            else:
                logger.warning("Cannot determine head_dim; skipping head pruning.")
                return

        # For GQA: determine the query-to-KV grouping ratio
        gqa_ratio = num_heads // num_kv_heads if num_kv_heads else 1

        # Decide how many KV heads to keep, then derive query heads from that.
        # With GQA, each KV head serves `gqa_ratio` query heads. We must prune
        # at the KV-group granularity: keep N_kv KV heads → keep N_kv * gqa_ratio
        # query heads. This ensures the repeat_kv operation stays consistent.
        n_kv_keep = max(1, int(num_kv_heads * (1.0 - self.ratio_heads)))
        n_heads_keep = n_kv_keep * gqa_ratio

        logger.info("Attention head pruning: %d → %d heads, %d → %d kv_heads (head_dim=%d, GQA ratio=%d)",
                    num_heads, n_heads_keep, num_kv_heads, n_kv_keep, head_dim, gqa_ratio)

        for layer_idx, layer in enumerate(tqdm(transformer_layers, desc="Head pruning")):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue

            q_proj = getattr(attn, "q_proj", None)
            if q_proj is None:
                continue

            # Compute per-KV-group importance from q_proj activation importance.
            # Each KV group has `gqa_ratio` query heads; average their importance.
            q_name = self._find_module_name(model, q_proj)
            q_imp = importance_map.get(q_name)
            if q_imp is None:
                continue

            # q_imp has shape (num_heads * head_dim,) — reshape to (num_heads, head_dim)
            per_head_imp = q_imp[:num_heads * head_dim].view(num_heads, head_dim).mean(dim=1)
            # Average query head importance within each KV group
            kv_group_imp = per_head_imp.view(num_kv_heads, gqa_ratio).mean(dim=1)  # (num_kv_heads,)

            # Select top-k KV groups
            _, kv_keep_idx = torch.topk(kv_group_imp, k=n_kv_keep, largest=True, sorted=True)
            kv_keep_idx = kv_keep_idx.sort().values

            # Derive query head indices from kept KV groups
            q_head_keep_idx = torch.cat([
                torch.arange(h * gqa_ratio, (h + 1) * gqa_ratio) for h in kv_keep_idx
            ])

            # Build neuron-level indices
            q_neuron_idx = torch.cat([
                torch.arange(h * head_dim, (h + 1) * head_dim) for h in q_head_keep_idx
            ])
            kv_neuron_idx = torch.cat([
                torch.arange(h * head_dim, (h + 1) * head_dim) for h in kv_keep_idx
            ])

            # Prune projections
            attn.q_proj = _prune_linear_output(q_proj, q_neuron_idx)

            k_proj = getattr(attn, "k_proj", None)
            v_proj = getattr(attn, "v_proj", None)
            o_proj = getattr(attn, "o_proj", None)

            if k_proj is not None:
                attn.k_proj = _prune_linear_output(k_proj, kv_neuron_idx)
            if v_proj is not None:
                attn.v_proj = _prune_linear_output(v_proj, kv_neuron_idx)
            if o_proj is not None:
                attn.o_proj = _prune_linear_input(o_proj, q_neuron_idx)

            logger.debug("Layer %d: heads %d → %d, kv_heads %d → %d",
                        layer_idx, num_heads, n_heads_keep,
                        num_kv_heads, n_kv_keep)

        # Update global model config (all attention modules share same config object)
        cfg.num_attention_heads = n_heads_keep
        cfg.num_key_value_heads = n_kv_keep

        # Update cached attributes on each attention module.
        # Qwen3 attention computes num_key_value_groups at __init__ time
        # and caches it — we must update it after changing the config.
        for layer in transformer_layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            if hasattr(attn, "num_heads"):
                attn.num_heads = n_heads_keep
            if hasattr(attn, "num_key_value_heads"):
                attn.num_key_value_heads = n_kv_keep
            if hasattr(attn, "num_key_value_groups"):
                attn.num_key_value_groups = gqa_ratio  # unchanged: n_heads / n_kv = gqa_ratio

        logger.info("Updated model config: num_attention_heads=%d, num_key_value_heads=%d, kv_groups=%d",
                    cfg.num_attention_heads, cfg.num_key_value_heads, gqa_ratio)

    # ── Hidden dimension pruning ───────────────────────────────────────────────

    def _prune_hidden_dim(
        self,
        model: nn.Module,
        transformer_layers: list[nn.Module],
        importance_map: dict[str, torch.Tensor],
    ) -> None:
        """Prune the global hidden dimension (embedding channels).

        This is the most invasive width pruning: it removes channels that
        flow through *every* linear layer.  We build a global importance score
        by averaging over all o_proj outputs (which project back to hidden dim)
        and all q/k/v/gate/up input importances.

        A single global keep_mask is derived and applied consistently to:
          • Embedding table (output columns)
          • o_proj output (all attention layers)
          • q/k/v/gate/up input (all layers)
          • down_proj output (all layers)
          • LM head input

        NOTE: Hidden-dim pruning requires modifying the embedding and language
        model head as well, which changes the model configuration.  This is
        done in-place.  If the architecture wraps these differently (tied
        weights, etc.) the code detects and handles it.
        """
        if not self.prune_hidden:
            return

        logger.info("Computing global hidden dimension importance...")

        # Collect all o_proj output importances (these are in hidden space)
        hidden_importances: list[torch.Tensor] = []
        for name, imp in importance_map.items():
            if "o_proj" in name and imp is not None:
                hidden_importances.append(imp)

        if not hidden_importances:
            logger.warning("No o_proj importances found; skipping hidden pruning.")
            return

        # Global importance = mean across all o_proj layers
        global_imp = torch.stack(hidden_importances).mean(dim=0)
        keep_idx = _importance_to_keep_mask(global_imp, self.ratio_hidden)
        n_hidden = len(global_imp)
        n_keep = len(keep_idx)
        logger.info("Hidden dim pruning: %d → %d channels (%.1f%% kept)",
                    n_hidden, n_keep, 100.0 * n_keep / n_hidden)

        # Locate the LLM submodule (handles InternVL, Qwen-VL, etc.)
        from src.utils.model_utils import get_llm_submodule
        llm = get_llm_submodule(model)
        # Inner model (the actual transformer body with embed_tokens, layers, norm)
        inner = getattr(llm, "model", llm)

        # Prune embedding
        embed = getattr(inner, "embed_tokens", None)
        if embed is not None and isinstance(embed, nn.Embedding):
            new_embed = nn.Embedding(embed.num_embeddings, n_keep, padding_idx=embed.padding_idx)
            new_embed.weight = nn.Parameter(embed.weight.data[:, keep_idx].clone())
            new_embed.to(embed.weight.device)
            inner.embed_tokens = new_embed

        # Prune each transformer layer's attention and MLP projections
        for layer in tqdm(transformer_layers, desc="Hidden dim pruning"):
            self._prune_attn_hidden(layer, keep_idx)
            self._prune_mlp_hidden(layer, keep_idx)
            # LayerNorms
            for ln_name in ("input_layernorm", "post_attention_layernorm"):
                ln = getattr(layer, ln_name, None)
                if ln is not None:
                    self._prune_layernorm(ln, keep_idx)

        # Prune final norm
        final_norm = getattr(inner, "norm", None)
        if final_norm is not None:
            self._prune_layernorm(final_norm, keep_idx)

        # Prune LM head input
        lm_head = getattr(llm, "lm_head", None)
        if lm_head is not None and isinstance(lm_head, nn.Linear):
            setattr(llm, "lm_head", _prune_linear_input(lm_head, keep_idx))

    def _prune_attn_hidden(self, layer: nn.Module, keep_idx: torch.Tensor) -> None:
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            return
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(attn, proj_name, None)
            if proj is not None:
                setattr(attn, proj_name, _prune_linear_input(proj, keep_idx))
        o_proj = getattr(attn, "o_proj", None)
        if o_proj is not None:
            attn.o_proj = _prune_linear_output(o_proj, keep_idx)

    def _prune_mlp_hidden(self, layer: nn.Module, keep_idx: torch.Tensor) -> None:
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            return
        for proj_name in ("gate_proj", "up_proj"):
            proj = getattr(mlp, proj_name, None)
            if proj is not None:
                setattr(mlp, proj_name, _prune_linear_input(proj, keep_idx))
        down_proj = getattr(mlp, "down_proj", None)
        if down_proj is not None:
            mlp.down_proj = _prune_linear_output(down_proj, keep_idx)

    def _prune_layernorm(self, ln: nn.Module, keep_idx: torch.Tensor) -> None:
        if hasattr(ln, "weight"):
            ln.weight = nn.Parameter(ln.weight.data[keep_idx].clone())
        if hasattr(ln, "bias") and ln.bias is not None:
            ln.bias = nn.Parameter(ln.bias.data[keep_idx].clone())

    # ── config update ─────────────────────────────────────────────────────────

    def _update_config_dimensions(
        self,
        model: nn.Module,
        transformer_layers: list[nn.Module],
    ) -> None:
        """Update model config to reflect pruned hidden_size and intermediate_size.

        After width pruning, the actual weight dimensions may differ from what
        the config says.  We inspect the actual weights and update the config
        so that save_pretrained() writes a correct config.json.
        """
        top_cfg = getattr(model, "config", None)
        if top_cfg is None:
            return

        # For InternVL: LLM config is nested under llm_config
        cfg = getattr(top_cfg, "llm_config", top_cfg)

        # Detect actual hidden_size from embedding or first layer norm
        from src.utils.model_utils import get_llm_submodule
        llm = get_llm_submodule(model)
        inner = getattr(llm, "model", llm)
        embed = getattr(inner, "embed_tokens", None)
        if embed is not None and hasattr(embed, "embedding_dim"):
            actual_hidden = embed.embedding_dim
            if hasattr(cfg, "hidden_size") and cfg.hidden_size != actual_hidden:
                logger.info("Updating config hidden_size: %d → %d", cfg.hidden_size, actual_hidden)
                cfg.hidden_size = actual_hidden

        # Detect actual intermediate_size from the first layer's gate_proj
        if transformer_layers:
            mlp = getattr(transformer_layers[0], "mlp", None)
            if mlp is not None:
                gate_proj = getattr(mlp, "gate_proj", None)
                if gate_proj is not None and isinstance(gate_proj, nn.Linear):
                    actual_intermediate = gate_proj.out_features
                    if hasattr(cfg, "intermediate_size") and cfg.intermediate_size != actual_intermediate:
                        logger.info("Updating config intermediate_size: %d → %d",
                                    cfg.intermediate_size, actual_intermediate)
                        cfg.intermediate_size = actual_intermediate

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _find_module_name(model: nn.Module, target: nn.Module) -> str:
        """Return the full dotted name of `target` within `model`."""
        for name, module in model.named_modules():
            if module is target:
                return name
        return ""
