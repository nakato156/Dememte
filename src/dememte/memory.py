"""E10 — Hippocampal associative memory module (TTA-only, no trainable params).

Implements three biologically-inspired mechanisms on top of a frozen DeMemte VQSA
checkpoint, without any gradient or parameter update:

1. **Associative recall** as Modern Hopfield Network update against the codebook
   (semantic) and an episodic buffer.  Equivalent to soft cross-attention
   ``softmax(-‖z - E‖²/τ) · E`` per Ramsauer et al. 2021 / Millidge et al. 2022.
2. **Pattern completion** as a short iterative loop on the pooled latent
   ``z_pool``, gated by familiarity (Tyulmankov 2022) or unfamiliarity
   (Krotov & Hopfield 2021).
3. **Complementary learning systems** via an EMA episodic buffer
   (Sun, Saxe, Fitzgerald 2023) with optional slow consolidation toward the
   semantic codebook (Spens & Burgess 2024).

The integration point is a soft blend in ``zq_pool`` space — the same token the
trained classifier already receives — so the downstream pipeline (self-attention
+ classifier) stays in distribution by construction.  At ``λ_max = 0`` the
adapter is bit-identical to the source forward.

References
----------
- Ramsauer et al. (2021).  Hopfield Networks is All You Need.  ICLR.
- Millidge et al. (2022).  Universal Hopfield Networks.  ICML.
- Tyulmankov, Yang & Abbott (2022).  Meta-learning synaptic plasticity and
  memory addressing for continual familiarity detection.  Neuron 110.
- Krotov & Hopfield (2021).  Large associative memory problem in neurobiology
  and machine learning.  ICLR.
- Sun, Advani, Spruston, Saxe & Fitzgerald (2023).  Organizing memories for
  generalization in complementary learning systems.  Nature Neuroscience 26.
- Spens & Burgess (2024).  A generative model of memory construction and
  consolidation.  Nature Human Behaviour 8.
- Kim, Papyan & Mhammedi (2021).  The Lipschitz Constant of Self-Attention.
  ICML.
- Lim et al. (2023).  TTN: A Domain-Shift Aware Batch Normalization in
  Test-Time Adaptation.  ICLR — α-mix philosophy applied to token blending.
- Wang et al. (2022).  Continual Test-Time Domain Adaptation.  CVPR — argues
  against FIFO buffers because of ordering artifacts; this module uses EMA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tta import TTAStats


# ---------------------------------------------------------------------------
# Pure functions: associative recall, gate, pattern completion
# ---------------------------------------------------------------------------


def _squared_distances(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Pairwise squared L2 distances.  ``query (B, D)``, ``keys (K, D)`` → ``(B, K)``."""
    # Ensure keys live on the same device and dtype as query to avoid
    # "Expected all tensors to be on the same device" runtime errors
    # when callers pass buffers that may be on CPU while query is on CUDA.
    if keys.device != query.device or keys.dtype != query.dtype:
        keys = keys.to(device=query.device, dtype=query.dtype)

    q2 = query.pow(2).sum(dim=-1, keepdim=True)
    k2 = keys.pow(2).sum(dim=-1, keepdim=True).t()
    return q2 - 2.0 * query @ keys.t() + k2


def associative_recall(
    query: torch.Tensor,
    keys: torch.Tensor,
    temperature: float = 1.0,
    return_sharpness: bool = False,
):
    """Modern Hopfield / soft cross-attention recall.

    Computes ``softmax(-‖query - keys‖² / τ) · keys`` per Ramsauer et al. 2021
    Eq. 7, with distance-based attention (matches the geometry that the VQ
    codebook was trained under).

    Parameters
    ----------
    query : ``(B, D)`` tensor of queries.
    keys : ``(K, D)`` tensor of stored patterns.  Used as both keys and values.
    temperature : softmax temperature.  τ → 0 = nearest-neighbour lookup,
        τ → ∞ = uniform mean of ``keys``.
    return_sharpness : if true, also returns ``(B,)`` of ``max(softmax)`` per
        sample — a scalar in ``[0, 1]`` indicating how concentrated the recall
        is on a single pattern.

    Returns
    -------
    recall : ``(B, D)``
    sharpness : ``(B,)`` only if ``return_sharpness``.
    """
    if keys.numel() == 0:
        # Empty memory: recall is identity.
        out = query.clone()
        if return_sharpness:
            return out, torch.zeros(query.size(0), device=query.device)
        return out
    distances = _squared_distances(query, keys)
    tau = max(1e-6, float(temperature))
    weights = F.softmax(-distances / tau, dim=-1)
    recall = weights @ keys
    if return_sharpness:
        sharpness = weights.max(dim=-1).values
        return recall, sharpness
    return recall


def familiarity_gate(
    query: torch.Tensor,
    codebook: torch.Tensor,
    sigma: float = 1.0,
    mode: str = "familiarity",
) -> torch.Tensor:
    """Per-sample gate in ``[0, 1]`` based on min-dist to a reference codebook.

    Three modes (selected via ``mode``):

    - ``familiarity``  : ``g = exp(-min_dist² / σ²)`` — high when query is
      close to a stored pattern (Tyulmankov 2022).  Coherent with the
      Hopfield basin property: completion is one-step only when query lies
      inside a stored attractor.
    - ``unfamiliarity``: ``g = 1 − exp(-min_dist² / σ²)`` — high when query is
      far from any stored pattern (Krotov & Hopfield 2021, pattern
      completion as response to partial cue).
    - ``const``        : ``g = 1`` for every sample (no gating).

    Parameters
    ----------
    query : ``(B, D)``.
    codebook : ``(K, D)`` reference patterns.
    sigma : width of the gaussian; should be calibrated so ``g`` is non-trivial
        in the regime of interest.
    """
    if mode not in {"familiarity", "unfamiliarity", "const"}:
        raise ValueError(
            f"familiarity_gate: mode must be one of "
            f"('familiarity', 'unfamiliarity', 'const'), got {mode!r}"
        )
    if mode == "const":
        return torch.ones(query.size(0), device=query.device, dtype=query.dtype)
    if codebook.numel() == 0:
        return torch.zeros(query.size(0), device=query.device, dtype=query.dtype)
    distances = _squared_distances(query, codebook)
    min_dist_sq = distances.min(dim=-1).values.clamp_min(0.0)
    sigma2 = max(1e-12, float(sigma) ** 2)
    fam = torch.exp(-min_dist_sq / sigma2)
    if mode == "familiarity":
        return fam
    return 1.0 - fam


def _blend_recall(
    z: torch.Tensor,
    sem_keys: Optional[torch.Tensor],
    epi_keys: Optional[torch.Tensor],
    tau: float,
    tau_epi: float,
    beta: float,
    return_sharpness: bool = False,
):
    """Two-subsystem recall: ``β · recall_sem(z) + (1 − β) · recall_epi(z)``.

    Each subsystem keeps its own temperature.  This is the mathematically clean
    way to mix semantic and episodic recall — concatenation would force a
    shared temperature, which mis-scales the relative attention.

    When only one subsystem is provided, returns the recall from that one.
    Sharpness is reported from the dominant subsystem (or weighted average).
    """
    has_sem = sem_keys is not None and sem_keys.numel() > 0
    has_epi = epi_keys is not None and epi_keys.numel() > 0
    if not has_sem and not has_epi:
        out = z.clone()
        if return_sharpness:
            return out, torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        return out
    if has_sem and not has_epi:
        return associative_recall(z, sem_keys, temperature=tau, return_sharpness=return_sharpness)
    if has_epi and not has_sem:
        return associative_recall(z, epi_keys, temperature=tau_epi, return_sharpness=return_sharpness)
    rec_sem, sh_sem = associative_recall(z, sem_keys, temperature=tau, return_sharpness=True)
    rec_epi, sh_epi = associative_recall(z, epi_keys, temperature=tau_epi, return_sharpness=True)
    out = beta * rec_sem + (1.0 - beta) * rec_epi
    if return_sharpness:
        sharpness = beta * sh_sem + (1.0 - beta) * sh_epi
        return out, sharpness
    return out


def pattern_completion(
    z_pool: torch.Tensor,
    sem_keys: Optional[torch.Tensor],
    epi_keys: Optional[torch.Tensor],
    gate_codebook: torch.Tensor,
    T: int = 1,
    lambda_max: float = 0.1,
    tau: float = 1.0,
    tau_epi: float = 1.0,
    beta: float = 1.0,
    sigma: float = 1.0,
    gate_mode: str = "familiarity",
) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
    """Iterative pattern completion on a pooled query.

    At each step ``t``:

    ``recall_t = β · recall_sem(z_t) + (1 − β) · recall_epi(z_t)``
    ``g_t     = familiarity_gate(z_t, gate_codebook, σ, mode)``
    ``λ_eff_t = λ_max · g_t``
    ``z_{t+1} = (1 − λ_eff_t) · z_t + λ_eff_t · recall_t``

    ``traj_diff`` tracks ``‖z_{t+1} − z_t‖`` per iteration — convergence
    diagnostic (Kim et al. 2021: softmax-attention is not globally Lipschitz).

    Returns
    -------
    z_pool_T : ``(B, D)``.
    traj_diff : list of length ``T`` of mean-batch step magnitudes.
    final_g : ``(B,)`` final gate values.
    """
    if T < 0:
        raise ValueError("pattern_completion: T must be >= 0")
    z_t = z_pool
    traj_diff: List[float] = []
    final_g = torch.ones(z_pool.size(0), device=z_pool.device, dtype=z_pool.dtype)
    for _ in range(T):
        recall = _blend_recall(z_t, sem_keys, epi_keys, tau, tau_epi, beta)
        g = familiarity_gate(z_t, gate_codebook, sigma=sigma, mode=gate_mode)
        lambda_eff = lambda_max * g
        z_next = (1.0 - lambda_eff).unsqueeze(-1) * z_t + lambda_eff.unsqueeze(-1) * recall
        traj_diff.append(float((z_next - z_t).norm(dim=-1).mean().item()))
        z_t = z_next
        final_g = g
    return z_t, traj_diff, final_g


# ---------------------------------------------------------------------------
# EpisodicBuffer: fast plasticity store (CLS hippocampal subsystem)
# ---------------------------------------------------------------------------


class EpisodicBuffer(nn.Module):
    """Fixed-size key-value buffer with EMA writes (no FIFO).

    Writes use a **k-NN slot assignment**: each incoming vector finds the
    nearest existing slot and EMA-updates it with weight ``alpha``.  This
    naturally clusters similar inputs into the same slot — an episodic
    analogue of the CLS hippocampal pattern-separated index (Sun, Saxe,
    Fitzgerald 2023).

    No FIFO: ordering effects in test-time streams are a known confound
    (Wang et al. 2022 CoTTA).  EMA + k-NN slot assignment is order-invariant
    in the limit of many writes.

    Initialization can be from a reference codebook (default: zeros + lazy
    init on first write).  Initializing from the trained codebook gives the
    buffer a sensible semantic prior to start from.
    """

    def __init__(self, size: int = 256, dim: int = 256, alpha: float = 0.1, device: Optional[torch.device] = None):
        super().__init__()
        if size < 1:
            raise ValueError("EpisodicBuffer: size must be >= 1")
        self.size = int(size)
        self.dim = int(dim)
        self.alpha = float(alpha)
        # Determine buffer device: explicit `device` if provided, else CPU.
        buf_device = torch.device(device) if device is not None else torch.device("cpu")
        self.register_buffer("memory", torch.zeros(self.size, self.dim, device=buf_device))
        self.register_buffer("memory_init", torch.zeros(self.size, self.dim, device=buf_device))
        self.register_buffer("initialized", torch.tensor(False, device=buf_device))
        self.register_buffer("write_count", torch.tensor(0, dtype=torch.long, device=buf_device))

    @torch.no_grad()
    def initialize_from(self, source: torch.Tensor) -> None:
        """Seed the buffer with ``size`` vectors from ``source`` (``(K, D)``)."""
        if source.dim() != 2 or source.size(1) != self.dim:
            raise ValueError(
                f"EpisodicBuffer.initialize_from expects (K, {self.dim}); got {tuple(source.shape)}"
            )
        if source.size(0) >= self.size:
            idx = torch.randperm(source.size(0), device=source.device)[: self.size]
            init = source[idx]
        else:
            # Repeat with small jitter to reach ``size`` slots.
            reps = (self.size + source.size(0) - 1) // source.size(0)
            tiled = source.repeat(reps, 1)[: self.size]
            init = tiled + 1e-3 * torch.randn_like(tiled)
        self.memory.copy_(init.to(self.memory.dtype).to(self.memory.device))
        self.memory_init.copy_(self.memory)
        self.initialized.fill_(True)
        self.write_count.zero_()

    @torch.no_grad()
    def write(self, z_pool_batch: torch.Tensor) -> None:
        """EMA-update the nearest slot for each row of ``z_pool_batch`` ``(B, D)``."""
        if z_pool_batch.dim() != 2 or z_pool_batch.size(1) != self.dim:
            raise ValueError(
                f"EpisodicBuffer.write expects (B, {self.dim}); got {tuple(z_pool_batch.shape)}"
            )
        if not bool(self.initialized.item()):
            # Lazy init from the first batch: replicate batch vectors into the buffer.
            self.initialize_from(z_pool_batch.detach())
            return
        z = z_pool_batch.detach().to(self.memory.dtype).to(self.memory.device)
        distances = _squared_distances(z, self.memory)
        slot_idx = distances.argmin(dim=-1)
        # Vectorized scatter EMA: aggregate writes per slot then blend.
        # Some slots may receive multiple writes in one batch; we average them.
        contrib = torch.zeros_like(self.memory)
        count = torch.zeros(self.size, device=self.memory.device, dtype=self.memory.dtype)
        contrib.index_add_(0, slot_idx, z)
        count.index_add_(0, slot_idx, torch.ones_like(slot_idx, dtype=self.memory.dtype))
        mask = count > 0
        new_values = contrib[mask] / count[mask].unsqueeze(-1)
        self.memory[mask] = (1.0 - self.alpha) * self.memory[mask] + self.alpha * new_values
        self.write_count += int(z.size(0))

    @torch.no_grad()
    def read(self) -> torch.Tensor:
        """Return the buffer ``(size, dim)``."""
        return self.memory

    @torch.no_grad()
    def churn(self, eps: float = 1e-4) -> float:
        """Fraction of slots that drifted more than ``eps`` from ``memory_init``."""
        if not bool(self.initialized.item()):
            return 0.0
        delta = (self.memory - self.memory_init).norm(dim=-1)
        return float((delta > eps).float().mean().item())

    @torch.no_grad()
    def reset(self) -> None:
        self.memory.zero_()
        self.memory_init.zero_()
        self.initialized.fill_(False)
        self.write_count.zero_()


# ---------------------------------------------------------------------------
# Helper: extract the effective codebook from any quantizer
# ---------------------------------------------------------------------------


def effective_codebook(quantizer: nn.Module) -> Optional[torch.Tensor]:
    """Return the effective ``(K, D)`` codebook of a DeMemte quantizer, or None.

    - VanillaVQ / EMAVectorQuantizer2D: ``embedding.weight``
    - SimVQLinearQuantizer2D: ``codebook_transform(codebook_base)``  (property
      ``.codebook`` already exists)
    - FSQQuantizer2D: ``None`` (lookup-free, no stored patterns to recall from)
    """
    cb_property = getattr(quantizer, "codebook", None)
    if isinstance(cb_property, torch.Tensor):
        return cb_property
    embedding = getattr(quantizer, "embedding", None)
    if isinstance(embedding, nn.Embedding):
        return embedding.weight
    return None


# ---------------------------------------------------------------------------
# HippocampalMemoryAdapter — the test-time wrapper
# ---------------------------------------------------------------------------


@dataclass
class HippocampalConfig:
    """Configuration for HippocampalMemoryAdapter.

    Defaults are the ``assoc_recall_const`` baseline used in Phase 0 P0.3:
    one step, β=1 (semantic only), λ_max=0.1, gate=const.  With ``lambda_max=0``
    the adapter is bit-identical to the source forward.
    """

    # What to recall against.
    recall_sem: bool = True
    recall_epi: bool = False

    # Pattern completion loop on z_pool (before final blend into zq_pool).
    T: int = 1

    # Mixing weights.
    beta: float = 1.0          # weight of semantic recall (1 − β goes to episodic)
    lambda_max: float = 0.1    # cap on inject magnitude

    # Gate.
    gate_mode: str = "const"   # 'const' | 'familiarity' | 'unfamiliarity'
    sigma: float = 1.0

    # Recall temperatures.
    tau: float = 1.0
    tau_epi: float = 1.0

    # Episodic buffer.
    episodic_size: int = 256
    alpha_w: float = 0.1
    episodic_init_from_codebook: bool = True

    # Slow semantic consolidation (off by default; modo demostración only).
    alpha_s: float = 0.0
    consolidation_every: int = 50


class HippocampalMemoryAdapter(nn.Module):
    """Test-time wrapper around DeMemte VQSA that injects associative memory.

    The wrapped model is left fully frozen (``eval()`` + ``requires_grad_(False)``).
    Every forward replays the source pipeline up to GAP, then optionally:

    1. Runs ``pattern_completion`` on ``z_pool`` against the effective codebook
       and/or episodic buffer.
    2. Blends the resulting recall into ``zq_pool`` with ``λ_eff = λ_max · g``.
    3. Writes the (pre-completion) ``z_pool`` into the episodic buffer with
       EMA weight ``α_w``.
    4. Optionally consolidates the episodic buffer toward a *view* of the
       semantic codebook (the codebook on the checkpoint is **not** mutated —
       this view lives only in the adapter).

    Diagnostics published into the ``debug`` dict (per-sample tensors):
    ``recall_sharpness``, ``completion_amount``, ``episodic_buffer_churn``,
    ``g_mean``, ``traj_max_step``.

    Parameters
    ----------
    model : a ``DeMemteVQSA`` checkpoint (must be ``vqsa.use_codebook == True``;
        FSQ checkpoints are rejected because the lookup-free quantizer has no
        stored patterns to recall from — Mentzer et al. 2024).
    cfg : ``HippocampalConfig``.
    """

    method_name = "hippocampal_memory"

    def __init__(self, model: nn.Module, cfg: Optional[HippocampalConfig] = None):
        super().__init__()
        cfg = cfg or HippocampalConfig()
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)
        self.cfg = cfg
        self.stats = TTAStats()
        self._batch_idx = 0

        vqsa = getattr(model, "vqsa", None)
        if vqsa is None or not getattr(vqsa, "use_codebook", False):
            raise ValueError(
                "HippocampalMemoryAdapter requires a DeMemte VQSA model with a codebook."
            )
        cb = effective_codebook(vqsa.vq)
        if cb is None and cfg.recall_sem:
            raise ValueError(
                "Semantic recall requires a lookup-based codebook (vq / ema_vq / simvq_linear); "
                "the configured quantizer is lookup-free (FSQ)."
            )
        # Snapshot the source codebook for the optional adapter-local view used
        # by consolidation.  We never mutate the model's codebook in-place.
        # Determine the device of the provided model (fall back to CPU).
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            # If the model has no parameters, try buffers, else CPU.
            try:
                model_device = next(model.buffers()).device
            except StopIteration:
                model_device = torch.device("cpu")

        if cb is not None:
            self.register_buffer("_codebook_view", cb.detach().clone().to(device=model_device))
            self.register_buffer("_codebook_source", cb.detach().clone().to(device=model_device))
        else:
            self._codebook_view = None
            self._codebook_source = None
        dim = int(vqsa.latent_dim)
        # Create episodic buffer on the same device as the model to avoid device mismatches.
        self.episodic = EpisodicBuffer(cfg.episodic_size, dim, cfg.alpha_w, device=model_device)
        if cfg.recall_epi and cfg.episodic_init_from_codebook and cb is not None:
            self.episodic.initialize_from(cb.detach())

    # ------------------------------------------------------------------
    # Properties for tests / consolidation
    # ------------------------------------------------------------------

    @property
    def codebook_view(self) -> Optional[torch.Tensor]:
        return self._codebook_view

    @property
    def codebook_source(self) -> Optional[torch.Tensor]:
        return self._codebook_source

    @torch.no_grad()
    def reset(self) -> None:
        """Reset adapter state (buffer + counters + codebook view)."""
        self.stats = TTAStats()
        self._batch_idx = 0
        self.episodic.reset()
        if (
            self._codebook_view is not None
            and self._codebook_source is not None
        ):
            self._codebook_view.copy_(self._codebook_source)
        if self.cfg.recall_epi and self.cfg.episodic_init_from_codebook and self._codebook_source is not None:
            self.episodic.initialize_from(self._codebook_source)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, x: torch.Tensor, return_debug: bool = False):
        cfg = self.cfg
        vqsa = self.model.vqsa
        feats = self.model.backbone(x)
        z = vqsa.projector(feats)

        zq, vq_loss, codebook_loss, commitment_loss, dq_map, soft_assign, encoding_indices = vqsa.vq(z)
        z_pool = vqsa.pool(z).flatten(1)
        zq_pool_orig = vqsa.pool(zq).flatten(1)

        # Build the effective keys for recall.  ``codebook_view`` carries any
        # slow-consolidation drift; the codebook on the model is never touched.
        sem_keys = self._codebook_view if cfg.recall_sem else None
        epi_keys = self.episodic.read() if cfg.recall_epi else None

        # Gate codebook is always the semantic codebook (familiarity is defined
        # relative to the stored prototypes, not the episodic buffer).
        gate_cb = sem_keys if sem_keys is not None else self._codebook_source
        if gate_cb is None:
            # No semantic codebook at all (FSQ + recall_sem=False); fall back to constant gate.
            forced_gate_mode = "const"
        else:
            forced_gate_mode = cfg.gate_mode

        any_keys = (sem_keys is not None and sem_keys.numel() > 0) or (
            epi_keys is not None and epi_keys.numel() > 0
        )

        # ------------------------------------------------------------------
        # Step 1: iterative pattern completion on z_pool (against sem + epi).
        # ------------------------------------------------------------------
        if cfg.T == 0 or not any_keys:
            z_pool_T = z_pool
            traj_diff: List[float] = []
            final_g = torch.ones(z_pool.size(0), device=z_pool.device, dtype=z_pool.dtype)
        else:
            z_pool_T, traj_diff, final_g = pattern_completion(
                z_pool=z_pool,
                sem_keys=sem_keys,
                epi_keys=epi_keys,
                gate_codebook=gate_cb if gate_cb is not None else (sem_keys if sem_keys is not None else epi_keys),
                T=cfg.T,
                lambda_max=cfg.lambda_max,
                tau=cfg.tau,
                tau_epi=cfg.tau_epi,
                beta=cfg.beta,
                sigma=cfg.sigma,
                gate_mode=forced_gate_mode,
            )

        # ------------------------------------------------------------------
        # Step 2: final recall + blend into zq_pool.
        # ------------------------------------------------------------------
        if any_keys:
            recall_final, sharpness = _blend_recall(
                z_pool_T, sem_keys, epi_keys, cfg.tau, cfg.tau_epi, cfg.beta, return_sharpness=True
            )
            g_final = familiarity_gate(
                z_pool_T,
                gate_cb if gate_cb is not None else (sem_keys if sem_keys is not None else epi_keys),
                sigma=cfg.sigma,
                mode=forced_gate_mode,
            )
            lambda_eff = cfg.lambda_max * g_final
            zq_pool_refined = (
                (1.0 - lambda_eff).unsqueeze(-1) * zq_pool_orig
                + lambda_eff.unsqueeze(-1) * recall_final
            )
        else:
            sharpness = torch.zeros(z_pool.size(0), device=z_pool.device, dtype=z_pool.dtype)
            g_final = torch.zeros(z_pool.size(0), device=z_pool.device, dtype=z_pool.dtype)
            zq_pool_refined = zq_pool_orig

        # ------------------------------------------------------------------
        # Step 3: episodic write (with pre-completion z_pool).
        # ------------------------------------------------------------------
        if cfg.recall_epi:
            self.episodic.write(z_pool)

        # ------------------------------------------------------------------
        # Step 4: optional slow consolidation of the adapter-local codebook view.
        # ------------------------------------------------------------------
        if (
            cfg.alpha_s > 0.0
            and cfg.recall_epi
            and self._codebook_view is not None
            and self._batch_idx > 0
            and (self._batch_idx % max(1, cfg.consolidation_every)) == 0
        ):
            self._consolidate()
        self._batch_idx += 1

        # ------------------------------------------------------------------
        # Step 5: continue through self-attention + classifier with refined token.
        # ------------------------------------------------------------------
        tokens = self._tokens(vqsa.fusion_mode, z_pool, zq_pool_refined)
        attn_weights: List[torch.Tensor] = []
        if vqsa.use_self_attention:
            for block in vqsa.attention:
                tokens, weights = block(tokens)
                attn_weights.append(weights)
        fused = tokens.flatten(1)
        logits = self.model.classifier(fused)

        # ------------------------------------------------------------------
        # Diagnostics — per-sample where possible; scalar broadcast otherwise.
        # ------------------------------------------------------------------
        batch_size = z_pool.size(0)
        completion_amount = (
            (z_pool_T - z_pool).norm(dim=-1)
            / z_pool.norm(dim=-1).clamp_min(1e-8)
        )
        traj_max_step = float(max(traj_diff)) if traj_diff else 0.0
        episodic_churn = self.episodic.churn() if cfg.recall_epi else 0.0
        device = z_pool.device
        dtype = z_pool.dtype
        debug = {
            "z": z,
            "zq": zq,
            "z_pool": z_pool,
            "zq_pool": zq_pool_refined,
            "tokens": tokens,
            "attention_weights": torch.stack(attn_weights, dim=1) if attn_weights else None,
            "vq_loss": vq_loss,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "dq_map": dq_map,
            "soft_assign": soft_assign,
            "encoding_indices": encoding_indices,
            "quantizer_type": vqsa.quantizer_type,
            "num_embeddings": int(getattr(vqsa.vq, "num_embeddings", 0)),
            # Hippocampal diagnostics.
            "recall_sharpness": sharpness.detach(),
            "completion_amount": completion_amount.detach(),
            "g_mean": g_final.detach(),
            "traj_max_step": torch.full((batch_size,), traj_max_step, device=device, dtype=dtype),
            "episodic_buffer_churn": torch.full(
                (batch_size,), float(episodic_churn), device=device, dtype=dtype
            ),
        }
        self.stats.seen += batch_size
        if return_debug:
            return logits, debug
        return logits

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _tokens(fusion_mode: str, z_pool: torch.Tensor, zq_pool: torch.Tensor) -> torch.Tensor:
        if fusion_mode == "concat":
            return torch.stack([z_pool, zq_pool], dim=1)
        if fusion_mode == "replace":
            return torch.stack([zq_pool, zq_pool], dim=1)
        fused = z_pool + zq_pool
        return torch.stack([fused, fused], dim=1)

    @torch.no_grad()
    def _consolidate(self) -> None:
        """Slow EMA of the adapter-local codebook view toward episodic centroids.

        Implements (in spirit) Spens & Burgess 2024: hippocampal episodic
        memories slowly teach the neocortical model.  We move
        ``codebook_view`` toward the closest episodic centroid by ``α_s``
        per consolidation event.  The model's codebook is never touched.
        """
        if self._codebook_view is None:
            return
        M = self.episodic.read()
        # For each row of codebook_view, find nearest episodic slot and EMA.
        distances = _squared_distances(self._codebook_view, M)
        nearest = distances.argmin(dim=-1)
        targets = M[nearest]
        self._codebook_view.mul_(1.0 - self.cfg.alpha_s).add_(targets, alpha=self.cfg.alpha_s)


__all__ = [
    "HippocampalConfig",
    "HippocampalMemoryAdapter",
    "EpisodicBuffer",
    "associative_recall",
    "familiarity_gate",
    "pattern_completion",
    "effective_codebook",
]


# Internal export for tests of the two-subsystem blend.
_blend_recall_for_test = _blend_recall
