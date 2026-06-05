"""Test-time adaptation helpers for DeMemte VQSA.

E7 TTA variants:
- TENT-style entropy minimization over BatchNorm affine parameters.
- EATA-lite reliable/non-redundant entropy minimization without Fisher.

E7b conservative variants (avoid the BN batch-stats collapse):
- LayerNorm-only adaptation surface (``configure_tta_layernorm`` /
  ``collect_tta_ln_params``) that keeps BatchNorm on source running stats.
- ``NoUpdateAdapter`` for the reported "BN Stats" baseline.
- ``latent_memory_loss`` + ``MemoryTentAdapter`` for DeMemte pattern-completion
  preservation of z/zq/codebook assignments against a frozen source teacher.
- ``SourceFilterEATAAdapter`` whose reliability filter reads the teacher logits.

E7c-A codebook plasticity (SimVQ only):
- ``configure_tta_codebook`` / ``collect_tta_codebook_params`` expose the SimVQ
  ``codebook_transform.weight`` (~16k params) as the adaptation surface upstream
  of the straight-through estimator that blocks gradient to the codebook from
  ``q_st``. ``MemoryTentAdapter`` / ``SourceFilterEATAAdapter`` are reused as-is.
- ``SoftAssignTentAdapter`` minimizes entropy of the soft codebook assignment
  (a path with live gradient to the codebook through ``softmax(-distances)``).
- ``CodebookLossAdapter`` minimizes the VQ ``codebook_loss = MSE(q, z.detach())``
  which pushes the codebook toward target-domain latents (real adaptation).
- ``AlphaBNStatsAdapter`` (E7c-D) replaces ``projector.net.1`` BN stats with an
  alpha-mix of running and batch stats (TTN/alpha-BN), no gradient.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TTAStats:
    """Counters accumulated during online test-time adaptation."""

    updates: int = 0
    reliable: int = 0
    selected: int = 0
    seen: int = 0

    @property
    def selection_rate(self) -> float:
        return float(self.selected / max(1, self.seen))


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Entropy of a softmax distribution from logits, per sample."""
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def soft_assign_entropy(dbg: dict) -> torch.Tensor:
    """Mean per-location entropy of the soft codebook assignment, per sample.

    ``soft_assign`` has shape ``(B, H, W, K)`` (a softmax over K codes at each
    location) and depends on ``embedding.weight`` / ``codebook_transform.weight``
    without any straight-through detach, so minimizing this entropy has a live
    gradient to the codebook surface. Returns zeros for FSQ where
    ``soft_assign is None``.
    """
    soft = dbg.get("soft_assign")
    if soft is None:
        b = dbg["z"].size(0)
        return torch.zeros(b, device=dbg["z"].device)
    probs = soft.clamp_min(1e-8)
    entropy = -(probs * probs.log()).sum(dim=-1).mean(dim=(1, 2))
    return entropy


def collect_tta_bn_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    """Collect BatchNorm affine parameters used by TENT/EATA."""
    params: List[nn.Parameter] = []
    names: List[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in {"weight", "bias"}:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
    return params, names


def collect_tta_ln_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    """Collect LayerNorm affine parameters (the batch-agnostic adaptation surface).

    In DeMemte VQSA every ``nn.LayerNorm`` lives inside the self-attention blocks
    (``vqsa.attention.*.norm{1,2}``); the classifier and projector use BatchNorm.
    Adapting these affine params avoids the per-batch BN statistics that collapse
    the model on small ordered test batches.
    """
    params: List[nn.Parameter] = []
    names: List[str] = []
    for module_name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for param_name, param in module.named_parameters(recurse=False):
                if param_name in {"weight", "bias"}:
                    params.append(param)
                    names.append(f"{module_name}.{param_name}")
    return params, names


def collect_tta_codebook_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    """Collect the SimVQ ``codebook_transform.weight`` (the only adapter-reachable
    codebook surface).

    Vanilla VQ and EMA VQ both keep their straight-through estimator
    ``q_st = z + (q - z).detach()`` upstream of every classifier path, which
    zeros the gradient from ``zq`` to the codebook parameter. EMA VQ also
    registers its codebook as a buffer rather than a Parameter. FSQ is
    lookup-free. SimVQ alone exposes ``codebook_transform.weight`` (a
    learnable linear reparametrization of the codebook base) and routes its
    gradient through ``codebook_loss`` and ``soft_assign`` without detach.

    This helper raises if the model does not contain a ``codebook_transform``
    submodule (i.e. the quantizer is not ``simvq_linear``).
    """
    params: List[nn.Parameter] = []
    names: List[str] = []
    for module_name, module in model.named_modules():
        transform = getattr(module, "codebook_transform", None)
        if isinstance(transform, nn.Linear):
            params.append(transform.weight)
            names.append(f"{module_name}.codebook_transform.weight")
    if not params:
        raise ValueError(
            "collect_tta_codebook_params requires a simvq_linear quantizer; "
            "no nn.Linear codebook_transform was found."
        )
    return params, names


def configure_tta_model(model: nn.Module) -> nn.Module:
    """Freeze the model except BN affine params and force BN batch statistics.

    The full model remains in eval mode so dropout is disabled and EMA VQ
    codebooks are not updated. BatchNorm layers use per-batch statistics by
    clearing running statistics and disabling tracking.
    """
    model.eval()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def configure_tta_layernorm(model: nn.Module) -> nn.Module:
    """Freeze the model except LayerNorm affine params; keep BN on source stats.

    This is the E7b conservative configuration. Unlike :func:`configure_tta_model`,
    it never touches BatchNorm running statistics: BN stays in eval mode using the
    source running stats, which is what keeps the representation alive on small
    ordered test batches. Only the LayerNorm ``weight``/``bias`` inside the VQSA
    self-attention blocks (batch-agnostic) are made trainable. Dropout is disabled
    and the EMA VQ codebook is frozen because the model stays in eval mode.
    """
    model.eval()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.requires_grad_(True)
    return model


def configure_tta_codebook(model: nn.Module) -> nn.Module:
    """Freeze the model except the SimVQ ``codebook_transform.weight``.

    The E7c-A adaptation surface. BatchNorm stays in eval on source running
    stats; LayerNorm affine stays frozen; the backbone, projector, attention,
    and classifier are all frozen. Only the linear reparametrization of the
    codebook is trainable, so adaptation steps reshape the effective codebook
    while leaving every batch-statistics path untouched.
    """
    model.eval()
    model.requires_grad_(False)
    params, _ = collect_tta_codebook_params(model)
    for param in params:
        param.requires_grad_(True)
    return model


def make_tta_optimizer(params, lr: float = 2.5e-4, momentum: float = 0.9):
    return torch.optim.SGD(params, lr=lr, momentum=momentum)


def _freeze_teacher(model: nn.Module) -> nn.Module:
    """Put a source teacher model in frozen eval mode (running stats, no grad)."""
    model.eval()
    model.requires_grad_(False)
    return model


@torch.no_grad()
def _teacher_forward(source_model: nn.Module, x: torch.Tensor):
    """Run a frozen source teacher and return ``(logits, debug)``."""
    logits, _, dbg = source_model(x, return_debug=True)
    return logits, dbg


def latent_memory_loss(
    student_dbg: dict,
    teacher_dbg: dict,
    w_z: float = 1.0,
    w_zq: float = 1.0,
    w_assign: float = 1.0,
) -> torch.Tensor:
    """Pull the student's latent memory back toward the frozen source teacher.

    Implements the DeMemte pattern-completion regularizer: preserve the shared
    clean/corrupt latent space rather than only minimizing class entropy.

    ``L = w_z·MSE(z, z_src) + w_zq·MSE(zq, zq_src) + w_assign·KL(p_src ‖ p)``

    where ``p`` is the per-location soft codebook assignment. The KL term is
    skipped when ``soft_assign`` is ``None`` (e.g. the FSQ quantizer).
    """
    z = student_dbg["z"]
    loss = z.sum() * 0.0
    if w_z:
        loss = loss + w_z * F.mse_loss(z, teacher_dbg["z"].detach())
    if w_zq:
        loss = loss + w_zq * F.mse_loss(student_dbg["zq"], teacher_dbg["zq"].detach())
    soft = student_dbg.get("soft_assign")
    soft_src = teacher_dbg.get("soft_assign")
    if w_assign and soft is not None and soft_src is not None:
        log_p = soft.clamp_min(1e-8).log()
        log_p_src = soft_src.clamp_min(1e-8).log().detach()
        p_src = soft_src.detach()
        kl = (p_src * (log_p_src - log_p)).sum(dim=-1).mean()
        loss = loss + w_assign * kl
    return loss


class _BaseAdapter(nn.Module):
    def __init__(self, model: nn.Module, optimizer, steps: int = 1, episodic: bool = False):
        super().__init__()
        if steps < 1:
            raise ValueError("TTA adapters require steps >= 1")
        self.model = model
        self.optimizer = optimizer
        self.steps = int(steps)
        self.episodic = bool(episodic)
        self.stats = TTAStats()
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())

    def reset(self) -> None:
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.stats = TTAStats()


class NoUpdateAdapter(_BaseAdapter):
    """Forward-only adapter: runs the configured model without any gradient step.

    Used for the ``bn_stats_no_update`` baseline. Paired with a model configured
    by :func:`configure_tta_model` it reproduces the literature "BN Stats"
    baseline (per-batch statistics, no optimization), which is reported next to
    ``source`` rather than used as an exclusion gate.
    """

    method_name = "no_update"

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        logits, _, dbg = self.model(x, return_debug=True)
        self.stats.seen += x.size(0)
        if return_debug:
            return logits, dbg
        return logits


class TentAdapter(_BaseAdapter):
    """TENT-style online entropy minimization."""

    method_name = "tent_bn"

    @torch.enable_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        if self.episodic:
            self.reset()
        logits, dbg = None, None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits, _, dbg = self.model(x, return_debug=True)
            loss = softmax_entropy(logits).mean()
            loss.backward()
            self.optimizer.step()
            self.stats.updates += 1
            self.stats.reliable += x.size(0)
            self.stats.selected += x.size(0)
            self.stats.seen += x.size(0)
        if return_debug:
            return logits, dbg
        return logits


class EATALiteAdapter(_BaseAdapter):
    """EATA sample filtering without Fisher regularization."""

    method_name = "eata_lite"

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        num_classes: int,
        steps: int = 1,
        episodic: bool = False,
        e_margin: float | None = None,
        d_margin: float = 0.05,
        prob_momentum: float = 0.9,
    ):
        super().__init__(model, optimizer, steps=steps, episodic=episodic)
        self.e_margin = float(0.4 * math.log(num_classes) if e_margin is None else e_margin)
        self.d_margin = float(d_margin)
        self.prob_momentum = float(prob_momentum)
        self.current_model_probs: torch.Tensor | None = None

    def reset(self) -> None:
        super().reset()
        self.current_model_probs = None

    @torch.no_grad()
    def _update_model_probs(self, new_probs: torch.Tensor) -> None:
        if new_probs.numel() == 0:
            return
        mean_probs = new_probs.mean(dim=0).detach()
        if self.current_model_probs is None:
            self.current_model_probs = mean_probs
        else:
            self.current_model_probs = self.prob_momentum * self.current_model_probs + (1.0 - self.prob_momentum) * mean_probs

    @torch.enable_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        if self.episodic:
            self.reset()
        logits, dbg = None, None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits, _, dbg = self.model(x, return_debug=True)
            entropies = softmax_entropy(logits)
            reliable_mask = entropies < self.e_margin
            reliable_count = int(reliable_mask.sum().item())
            selected_mask = reliable_mask.clone()

            if self.current_model_probs is not None and reliable_count > 0:
                probs = logits.softmax(dim=1)
                similarities = F.cosine_similarity(
                    self.current_model_probs.unsqueeze(0),
                    probs[reliable_mask],
                    dim=1,
                )
                reliable_indices = reliable_mask.nonzero(as_tuple=False).flatten()
                selected_mask[:] = False
                selected_mask[reliable_indices[torch.abs(similarities) < self.d_margin]] = True

            selected_count = int(selected_mask.sum().item())
            self.stats.reliable += reliable_count
            self.stats.selected += selected_count
            self.stats.seen += x.size(0)

            if selected_count > 0:
                selected_entropies = entropies[selected_mask]
                coeff = torch.exp(-(selected_entropies.detach() - self.e_margin))
                loss = (selected_entropies * coeff).mean()
                loss.backward()
                self.optimizer.step()
                self.stats.updates += 1
                self._update_model_probs(logits.softmax(dim=1)[selected_mask])
            elif reliable_count > 0 and self.current_model_probs is None:
                self._update_model_probs(logits.softmax(dim=1)[reliable_mask])

        if return_debug:
            return logits, dbg
        return logits


class MemoryTentAdapter(_BaseAdapter):
    """TENT over LayerNorm + DeMemte latent-memory preservation regularizer.

    Adds :func:`latent_memory_loss` against a frozen source teacher to the
    entropy objective, so adaptation preserves the shared clean/corrupt latent
    space (pattern completion) instead of only collapsing class entropy.
    """

    method_name = "tent_ln_memreg"

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        source_model: nn.Module,
        steps: int = 1,
        episodic: bool = False,
        w_z: float = 1.0,
        w_zq: float = 1.0,
        w_assign: float = 1.0,
    ):
        super().__init__(model, optimizer, steps=steps, episodic=episodic)
        self.source_model = _freeze_teacher(source_model)
        self.w_z = float(w_z)
        self.w_zq = float(w_zq)
        self.w_assign = float(w_assign)

    @torch.enable_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        if self.episodic:
            self.reset()
        _, teacher_dbg = _teacher_forward(self.source_model, x)
        logits, dbg = None, None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits, _, dbg = self.model(x, return_debug=True)
            loss = softmax_entropy(logits).mean()
            loss = loss + latent_memory_loss(dbg, teacher_dbg, self.w_z, self.w_zq, self.w_assign)
            loss.backward()
            self.optimizer.step()
            self.stats.updates += 1
            self.stats.reliable += x.size(0)
            self.stats.selected += x.size(0)
            self.stats.seen += x.size(0)
        if return_debug:
            return logits, dbg
        return logits


class SoftAssignTentAdapter(_BaseAdapter):
    """TENT-style entropy minimization on the soft codebook assignment.

    Pure entropy on classifier logits cannot reach the codebook because the
    quantizer's straight-through estimator ``q_st = z + (q - z).detach()`` zeros
    the gradient from ``zq`` to ``embedding.weight`` / ``codebook_transform``.
    Entropy on ``soft_assign`` does have live gradient to the codebook surface,
    so this adapter is the minimal-mechanism positive for E7c-A.
    """

    method_name = "tent_codebook_softassign"

    @torch.enable_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        if self.episodic:
            self.reset()
        logits, dbg = None, None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits, _, dbg = self.model(x, return_debug=True)
            loss = soft_assign_entropy(dbg).mean()
            loss.backward()
            self.optimizer.step()
            self.stats.updates += 1
            self.stats.reliable += x.size(0)
            self.stats.selected += x.size(0)
            self.stats.seen += x.size(0)
        if return_debug:
            return logits, dbg
        return logits


class CodebookLossAdapter(_BaseAdapter):
    """Adapt the codebook by minimizing the VQ ``codebook_loss`` at test time.

    ``codebook_loss = MSE(q, z_e.detach())`` pulls the codebook entries toward
    the target-domain latents ``z_e`` of the corrupted batch (``q = one_hot @ emb``
    is direct, without any straight-through detach). This is the closest analogue
    to real synaptic adaptation among the available signals: the codebook moves
    toward the input distribution rather than collapsing class entropy.

    Optional ``source_model`` + ``memory_weights`` adds the source-anchored
    :func:`latent_memory_loss` (KL on ``soft_assign``) as an anti-drift term.
    """

    method_name = "codebook_loss_adapt"

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        source_model: nn.Module | None = None,
        steps: int = 1,
        episodic: bool = False,
        memory_weights: Tuple[float, float, float] | None = None,
    ):
        super().__init__(model, optimizer, steps=steps, episodic=episodic)
        self.source_model = _freeze_teacher(source_model) if source_model is not None else None
        self.memory_weights = memory_weights

    @torch.enable_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        if self.episodic:
            self.reset()
        teacher_dbg = None
        if self.source_model is not None and self.memory_weights is not None:
            _, teacher_dbg = _teacher_forward(self.source_model, x)
        logits, dbg = None, None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits, _, dbg = self.model(x, return_debug=True)
            loss = dbg["codebook_loss"]
            if teacher_dbg is not None:
                loss = loss + latent_memory_loss(dbg, teacher_dbg, *self.memory_weights)
            loss.backward()
            self.optimizer.step()
            self.stats.updates += 1
            self.stats.reliable += x.size(0)
            self.stats.selected += x.size(0)
            self.stats.seen += x.size(0)
        if return_debug:
            return logits, dbg
        return logits


class SourceFilterEATAAdapter(EATALiteAdapter):
    """EATA-lite whose reliability/diversity filter reads a frozen source teacher.

    The entropy/diversity selection is computed from the teacher logits (which
    are not contaminated by the adapting student), while the gradient is taken on
    the student logits. Optional ``memory_weights`` adds :func:`latent_memory_loss`.
    """

    method_name = "eata_ln_srcfilter"

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        num_classes: int,
        source_model: nn.Module,
        steps: int = 1,
        episodic: bool = False,
        e_margin: float | None = None,
        d_margin: float = 0.05,
        prob_momentum: float = 0.9,
        memory_weights: Tuple[float, float, float] | None = None,
    ):
        super().__init__(
            model,
            optimizer,
            num_classes,
            steps=steps,
            episodic=episodic,
            e_margin=e_margin,
            d_margin=d_margin,
            prob_momentum=prob_momentum,
        )
        self.source_model = _freeze_teacher(source_model)
        self.memory_weights = memory_weights

    @torch.enable_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        if self.episodic:
            self.reset()
        teacher_logits, teacher_dbg = _teacher_forward(self.source_model, x)
        logits, dbg = None, None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            logits, _, dbg = self.model(x, return_debug=True)

            # Reliability + diversity filtering on the descontaminated teacher logits.
            filter_entropies = softmax_entropy(teacher_logits)
            reliable_mask = filter_entropies < self.e_margin
            reliable_count = int(reliable_mask.sum().item())
            selected_mask = reliable_mask.clone()
            if self.current_model_probs is not None and reliable_count > 0:
                teacher_probs = teacher_logits.softmax(dim=1)
                similarities = F.cosine_similarity(
                    self.current_model_probs.unsqueeze(0),
                    teacher_probs[reliable_mask],
                    dim=1,
                )
                reliable_indices = reliable_mask.nonzero(as_tuple=False).flatten()
                selected_mask[:] = False
                selected_mask[reliable_indices[torch.abs(similarities) < self.d_margin]] = True

            selected_count = int(selected_mask.sum().item())
            self.stats.reliable += reliable_count
            self.stats.selected += selected_count
            self.stats.seen += x.size(0)

            if selected_count > 0:
                # Gradient on the student logits for the selected samples.
                selected_entropies = softmax_entropy(logits)[selected_mask]
                coeff = torch.exp(-(filter_entropies[selected_mask].detach() - self.e_margin))
                loss = (selected_entropies * coeff).mean()
                if self.memory_weights is not None:
                    loss = loss + latent_memory_loss(dbg, teacher_dbg, *self.memory_weights)
                loss.backward()
                self.optimizer.step()
                self.stats.updates += 1
                self._update_model_probs(teacher_logits.softmax(dim=1)[selected_mask])
            elif reliable_count > 0 and self.current_model_probs is None:
                self._update_model_probs(teacher_logits.softmax(dim=1)[reliable_mask])

        if return_debug:
            return logits, dbg
        return logits


class AlphaBNStatsAdapter(nn.Module):
    """E7c-D / TTN: alpha-mix of running stats and batch stats at ``projector.net.1``.

    Replaces only the projector BatchNorm forward (the BN immediately downstream
    of the frozen backbone, immediately upstream of the VQ codebook) with an
    alpha-mixed statistic where ``alpha`` is the **weight on the source running
    stats**:

    ``mean = alpha * running_mean + (1 - alpha) * batch_mean``
    ``var  = alpha * running_var  + (1 - alpha) * batch_var``

    Use ``alpha`` close to 1.0 (e.g. 0.90-0.95) to keep most of the source
    running statistics and only inject a small batch correction; ``alpha=1.0``
    is a true no-op (identical to source BN eval), ``alpha=0.0`` reproduces the
    ``bn_stats_no_update`` collapse. No gradient steps, no learnable parameters.
    Implemented via a forward hook that returns the alpha-mixed BN output;
    ``close()`` and ``reset()`` remove the hook to restore the original BN
    forward.
    """

    method_name = "ttn_alpha_bn"

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        target_module_name: str = "vqsa.projector.net.1",
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.alpha = float(alpha)
        self.target_module_name = str(target_module_name)
        self.stats = TTAStats()
        target = self._resolve_target()
        if not isinstance(target, nn.BatchNorm2d):
            raise TypeError(
                f"AlphaBNStatsAdapter target {target_module_name!r} must be nn.BatchNorm2d, "
                f"got {type(target).__name__}"
            )
        self._target = target
        self._handle = target.register_forward_hook(self._make_hook(self.alpha))

    def _resolve_target(self) -> nn.Module:
        return self.model.get_submodule(self.target_module_name)

    @staticmethod
    def _make_hook(alpha: float):
        def hook(module: nn.BatchNorm2d, inputs, output):
            x = inputs[0]
            if x.dim() == 4:
                dims = (0, 2, 3)
            else:
                dims = (0,)
            batch_mean = x.mean(dim=dims).detach()
            batch_var = x.var(dim=dims, unbiased=False).detach()
            running_mean = module.running_mean if module.running_mean is not None else batch_mean
            running_var = module.running_var if module.running_var is not None else batch_var
            mixed_mean = alpha * running_mean + (1.0 - alpha) * batch_mean
            mixed_var = alpha * running_var + (1.0 - alpha) * batch_var
            return F.batch_norm(
                x,
                mixed_mean,
                mixed_var,
                module.weight,
                module.bias,
                training=False,
                momentum=0.0,
                eps=module.eps,
            )

        return hook

    def reset(self) -> None:
        self.stats = TTAStats()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        logits, _, dbg = self.model(x, return_debug=True)
        self.stats.seen += x.size(0)
        if return_debug:
            return logits, dbg
        return logits
