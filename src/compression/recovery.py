"""
Phase 4 — Recovery: Fine-tuning and Knowledge Distillation.

After width and depth pruning the model has lost representational capacity.
This phase restores accuracy via:

  A) **Knowledge Distillation** (primary, recommended — Minitron approach)
     The original uncompressed model acts as teacher.  The student (compressed
     model) is trained to minimise the forward KL divergence between teacher
     and student output logit distributions:

       L_distill = KL( softmax(z_teacher / T) || softmax(z_student / T) )

     where T is the temperature.  Using only the KL loss (no cross-entropy)
     requires 50× fewer training tokens than standard SFT to recover quality
     (arXiv:2408.11796, §4).

  B) **Standard Fine-tuning** (fallback / supplementary)
     Standard next-token prediction cross-entropy on a text corpus.  Used when
     the teacher model is unavailable (e.g., memory constraints) or as a
     supplement after distillation.

  C) **Both** — distillation first, then a short supervised fine-tuning pass.

Memory-efficient design (8GB VRAM target):
  - Teacher model quantized to int4 (NF4) via bitsandbytes (~0.5GB for 1B model).
  - Student optimizer uses AdamW8bit (int8 states, 4x less memory than fp32).
  - Student uses gradient checkpointing to reduce activation memory.
  - Total VRAM: student bf16 (~1.5GB) + optim8bit (~1.5GB) + teacher int4 (~0.5GB)
    = ~3.5GB, leaving ~4.5GB for activations/logits/gradients.

The trainer is intentionally kept dependency-light: it uses plain PyTorch
optimisation loops with gradient accumulation, a cosine LR scheduler, and
optional tensorboard logging.  This avoids requiring HuggingFace Trainer
which has non-trivial interaction with custom model architectures.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.model_utils import load_model_and_tokenizer, save_model, get_llm_submodule

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def _kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Forward KL divergence loss between teacher and student logit distributions.

    KL( p_teacher || p_student )  — matches Minitron's choice.

    Args:
        student_logits: (B, T, V) student output logits.
        teacher_logits: (B, T, V) teacher output logits (no grad needed).
        temperature: Softmax temperature T.  T=1 matches standard softmax.

    Returns:
        Scalar mean KL loss.
    """
    B, T, V = student_logits.shape
    s_log_prob = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t_prob = F.softmax(teacher_logits.float() / temperature, dim=-1)
    # kl_div expects log-probabilities for input, probabilities for target
    # reduction='batchmean' divides by batch size (standard convention)
    loss = F.kl_div(s_log_prob, t_prob, reduction="batchmean", log_target=False)
    return loss * (temperature ** 2)   # re-scale by T² (Hinton et al.)


def _lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Standard next-token prediction cross-entropy loss.

    Args:
        logits: (B, T, V).
        labels: (B, T) — shift handled here (predict token t+1 from t).

    Returns:
        Scalar mean cross-entropy.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Optimiser / scheduler helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_optimiser(model: nn.Module, lr: float, weight_decay: float) -> object:
    """Build AdamW8bit optimizer (bitsandbytes) for 4x less optimizer memory.

    Falls back to standard AdamW if bitsandbytes is unavailable.
    """
    # Separate weight-decay from bias / layernorm parameters (standard practice)
    no_decay = {"bias", "layer_norm.weight", "layernorm.weight", "norm.weight"}
    grouped = [
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    try:
        import bitsandbytes as bnb
        logger.info("Using bitsandbytes AdamW8bit optimizer (4x less memory than fp32 AdamW).")
        return bnb.optim.AdamW8bit(grouped, lr=lr, betas=(0.9, 0.95), eps=1e-8)
    except ImportError:
        logger.warning("bitsandbytes not available, falling back to standard AdamW (high memory).")
        return AdamW(grouped, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def _build_scheduler(optimiser, total_steps: int, warmup_steps: int) -> object:
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup = LinearLR(optimiser, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimiser, T_max=max(1, total_steps - warmup_steps), eta_min=1e-6)
    return SequentialLR(optimiser, schedulers=[warmup, cosine], milestones=[warmup_steps])


# ──────────────────────────────────────────────────────────────────────────────
# RecoveryTrainer
# ──────────────────────────────────────────────────────────────────────────────

class RecoveryTrainer:
    """Orchestrates the recovery phase after pruning.

    Args:
        config: The "recovery" sub-dict from compression_config.yaml.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.method: str = config.get("method", "distillation")   # distillation | finetuning | both
        dist_cfg = config.get("distillation", {})
        self.temperature: float = dist_cfg.get("temperature", 1.0)
        self.alpha: float = dist_cfg.get("alpha", 1.0)
        self.teacher_model_name: Optional[str] = dist_cfg.get("teacher_model", None)
        train_cfg = config.get("training", {})
        self.output_dir = Path(train_cfg.get("output_dir", "outputs/recovery"))
        self.num_epochs: int = train_cfg.get("num_epochs", 1)
        self.batch_size: int = train_cfg.get("per_device_train_batch_size", 2)
        self.grad_accum: int = train_cfg.get("gradient_accumulation_steps", 8)
        self.lr: float = train_cfg.get("learning_rate", 1e-4)
        self.lr_scheduler: str = train_cfg.get("lr_scheduler", "cosine")
        self.warmup_ratio: float = train_cfg.get("warmup_ratio", 0.01)
        self.weight_decay: float = train_cfg.get("weight_decay", 0.01)
        self.max_grad_norm: float = train_cfg.get("max_grad_norm", 1.0)
        self.bf16: bool = train_cfg.get("bf16", True)
        self.logging_steps: int = train_cfg.get("logging_steps", 50)
        self.save_steps: int = train_cfg.get("save_steps", 500)

    # ── public API ─────────────────────────────────────────────────────────────

    def recover(
        self,
        student_model: nn.Module,
        processor: object,
        train_loader: DataLoader,
        device: torch.device,
        teacher_model: Optional[nn.Module] = None,
    ) -> nn.Module:
        """Run the recovery training phase.

        Args:
            student_model: The pruned student model to recover.
            processor: Tokenizer / processor (for saving).
            train_loader: DataLoader of training batches.
            device: Compute device.
            teacher_model: Optional pre-loaded teacher.  If None and distillation
                           is requested, the teacher is loaded from
                           self.teacher_model_name.

        Returns:
            Recovered student model.
        """
        logger.info("=== Recovery Phase — method: %s ===", self.method)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.method in ("distillation", "both"):
            teacher = self._get_teacher(teacher_model, processor, device)
            student_model = self._run_distillation(student_model, teacher, train_loader, device)
            # Free teacher
            del teacher
            torch.cuda.empty_cache()

        if self.method in ("finetuning", "both"):
            student_model = self._run_finetuning(student_model, train_loader, device)

        save_model(student_model, processor, str(self.output_dir / "final"))
        return student_model

    # ── teacher loading ────────────────────────────────────────────────────────

    def _get_teacher(
        self,
        teacher_model: Optional[nn.Module],
        processor: object,
        device: torch.device,
    ) -> nn.Module:
        """Load or prepare the teacher model in int4 quantization.

        Memory budget (8GB VRAM):
          - Teacher int4 (NF4):  ~0.5 GB for 1B model
          - Student bf16:        ~1.5 GB for 745M model
          - AdamW8bit states:    ~1.5 GB
          - Total base:          ~3.5 GB → leaves ~4.5 GB for activations

        If a pre-loaded teacher is passed, we discard it and reload in int4.
        The pre-loaded teacher was in bf16 which would waste GPU memory.
        """
        teacher_name = self.teacher_model_name
        if teacher_name is None and teacher_model is not None:
            # Pipeline passed teacher but no name — we can't reload in int4,
            # so move it to CPU as fallback.
            logger.info("No teacher_model_name set; using pre-loaded teacher on CPU (slower).")
            teacher_model = teacher_model.cpu()
            torch.cuda.empty_cache()
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            return teacher_model

        if teacher_name is None:
            raise ValueError(
                "Distillation requires a teacher model. "
                "Set recovery.distillation.teacher_model in the config "
                "or pass teacher_model= to recover()."
            )

        # Free pre-loaded teacher if exists (we'll reload in int4)
        if teacher_model is not None:
            logger.info("Freeing pre-loaded bf16 teacher to reload in int4.")
            del teacher_model
            torch.cuda.empty_cache()

        # Load teacher in int4 (NF4) quantization on GPU
        try:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,  # nested quantization for extra savings
            )
            logger.info("Loading teacher in int4 (NF4) on GPU: %s", teacher_name)
            from transformers import AutoModel
            teacher = AutoModel.from_pretrained(
                teacher_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning("int4 loading failed (%s), falling back to CPU bf16.", e)
            teacher, _ = load_model_and_tokenizer(
                teacher_name,
                dtype="bfloat16",
                device_map="cpu",
            )

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        return teacher

    # ── distillation training ─────────────────────────────────────────────────

    def _run_distillation(
        self,
        student: nn.Module,
        teacher: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> nn.Module:
        """Knowledge distillation with int4 teacher and bf16 student, both on GPU.

        Memory strategy for 8GB VRAM:
          1. Teacher quantized to int4 NF4 (~0.5GB) — fast GPU inference.
          2. Student in bf16 (~1.5GB) with gradient checkpointing.
          3. AdamW8bit optimizer (~1.5GB instead of ~6GB for fp32 AdamW).
          4. Batch size 1 with large gradient accumulation (effective batch=16).
          5. Total base memory: ~3.5GB, leaving ~4.5GB for activations/logits.
        """
        logger.info("Starting knowledge distillation...")

        # Get LLM submodules
        student_llm = get_llm_submodule(student)
        teacher_llm = get_llm_submodule(teacher)

        # Determine if teacher is on GPU or CPU
        teacher_device = next(teacher_llm.parameters()).device
        teacher_on_gpu = teacher_device.type == "cuda"
        logger.info("Teacher on %s, student on %s", teacher_device, device)

        student_llm.to(device)
        student_llm.train()
        teacher_llm.eval()

        # Enable gradient checkpointing on student to save activation memory
        if hasattr(student_llm, "gradient_checkpointing_enable"):
            student_llm.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing on student.")

        optimiser = _build_optimiser(student_llm, self.lr, self.weight_decay)
        total_steps = len(loader) * self.num_epochs // self.grad_accum
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        scheduler = _build_scheduler(optimiser, total_steps, warmup_steps)

        use_cuda = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=self.bf16 and use_cuda)
        global_step = 0
        log_loss = 0.0

        # Log memory usage
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
            logger.info("GPU memory allocated: %.2f GB",
                        torch.cuda.memory_allocated() / 1e9)

        for epoch in range(self.num_epochs):
            for step, batch in enumerate(tqdm(loader, desc=f"Distillation epoch {epoch+1}")):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                # ── Teacher forward (int4 on GPU or bf16 on CPU) ───────────
                with torch.no_grad():
                    if teacher_on_gpu:
                        t_ids = input_ids.to(teacher_device)
                        t_mask = attention_mask.to(teacher_device)
                    else:
                        t_ids = input_ids.cpu()
                        t_mask = attention_mask.cpu()

                    teacher_out = teacher_llm(
                        input_ids=t_ids,
                        attention_mask=t_mask,
                    )
                    # Detach logits, move to student device
                    teacher_logits = teacher_out.logits.detach().to(device)
                    del teacher_out, t_ids, t_mask

                # ── Student forward on GPU ─────────────────────────────────
                input_ids_gpu = input_ids.to(device)
                attention_mask_gpu = attention_mask.to(device)

                with torch.amp.autocast(device.type, enabled=self.bf16, dtype=torch.bfloat16):
                    student_out = student_llm(
                        input_ids=input_ids_gpu,
                        attention_mask=attention_mask_gpu,
                    )

                    # KL divergence loss
                    loss = _kl_loss(
                        student_out.logits,
                        teacher_logits,
                        temperature=self.temperature,
                    ) * self.alpha

                    del teacher_logits
                    del student_out
                    loss = loss / self.grad_accum

                scaler.scale(loss).backward()
                log_loss += loss.item() * self.grad_accum

                # Free memory
                del loss, input_ids_gpu, attention_mask_gpu

                if (step + 1) % self.grad_accum == 0:
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(student_llm.parameters(), self.max_grad_norm)
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    if global_step % self.logging_steps == 0:
                        avg_loss = log_loss / self.logging_steps
                        lr_now = optimiser.param_groups[0]["lr"]
                        mem_gb = torch.cuda.memory_allocated() / 1e9 if use_cuda else 0
                        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0
                        logger.info(
                            "Distil step %d  loss=%.4f  lr=%.2e  mem=%.1fGB  peak=%.1fGB",
                            global_step, avg_loss, lr_now, mem_gb, peak_gb,
                        )
                        log_loss = 0.0

                    if global_step % self.save_steps == 0:
                        ckpt_dir = self.output_dir / f"distil_step_{global_step}"
                        student.save_pretrained(str(ckpt_dir))
                        logger.info("Checkpoint saved to %s", ckpt_dir)

        # Disable gradient checkpointing after training
        if hasattr(student_llm, "gradient_checkpointing_disable"):
            student_llm.gradient_checkpointing_disable()

        if use_cuda:
            logger.info("Peak GPU memory during distillation: %.2f GB",
                        torch.cuda.max_memory_allocated() / 1e9)

        return student

    # ── fine-tuning ────────────────────────────────────────────────────────────

    def _run_finetuning(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> nn.Module:
        logger.info("Starting supervised fine-tuning (next-token prediction)...")
        # Use LLM submodule for VLMs (avoids pixel_values requirement)
        llm = get_llm_submodule(model)
        llm.train()

        # Enable gradient checkpointing to save memory
        if hasattr(llm, "gradient_checkpointing_enable"):
            llm.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing for SFT.")

        optimiser = _build_optimiser(llm, self.lr * 0.1, self.weight_decay)  # lower LR for SFT
        total_steps = len(loader) * self.num_epochs // self.grad_accum
        warmup_steps = max(1, int(total_steps * self.warmup_ratio))
        scheduler = _build_scheduler(optimiser, total_steps, warmup_steps)

        use_cuda = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=self.bf16 and use_cuda)
        global_step = 0
        log_loss = 0.0

        for epoch in range(self.num_epochs):
            for step, batch in enumerate(tqdm(loader, desc=f"SFT epoch {epoch+1}")):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.amp.autocast(device.type, enabled=self.bf16, dtype=torch.bfloat16):
                    out = llm(input_ids=input_ids, attention_mask=attention_mask)
                    loss = _lm_loss(out.logits, labels) / self.grad_accum

                scaler.scale(loss).backward()
                log_loss += loss.item() * self.grad_accum

                if (step + 1) % self.grad_accum == 0:
                    scaler.unscale_(optimiser)
                    nn.utils.clip_grad_norm_(llm.parameters(), self.max_grad_norm)
                    scaler.step(optimiser)
                    scaler.update()
                    optimiser.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

                    if global_step % self.logging_steps == 0:
                        avg_loss = log_loss / self.logging_steps
                        lr_now = optimiser.param_groups[0]["lr"]
                        logger.info(
                            "SFT step %d  loss=%.4f  lr=%.2e",
                            global_step, avg_loss, lr_now,
                        )
                        log_loss = 0.0

        # Disable gradient checkpointing after training
        if hasattr(llm, "gradient_checkpointing_disable"):
            llm.gradient_checkpointing_disable()

        return model
