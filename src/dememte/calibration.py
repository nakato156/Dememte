"""wtq.8 — calibration controls for the E11 retrieval-logit memory.

Two post-hoc mechanisms over the retrieval logits, both keeping the model frozen:

1. ``fit_temperature`` / ``apply_temperature`` — standard temperature scaling
   (Guo et al. 2017). Fit a scalar ``T>0`` on a clean-val split (min NLL) and divide
   the final logits by it. The argmax is invariant, so accuracy is unchanged; only
   calibration (ECE/NLL/brier) moves. This is the canonical C8 control.

2. ``confidence_gate`` — per-sample modulation of the retrieval weight ``alpha`` from
   signals the adapter already exposes (base confidence, cache top-affinity, cache
   margin). Suppresses the cache vote where the base is already confident — targets the
   wtq.7 frontier (CIFAR-10 negative) and over-confidence.

Both are designed to run **offline** on dumped per-sample tensors
(``base_logits``, ``cache_logits``, ``base_conf``, ``top_affinity``, ``margin``), so the
full slate can be evaluated from a single forward pass per condition.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .evaluation import _compute_ece

VALID_GATE_MODES = {"fixed", "unfamiliarity", "familiarity", "unfamiliarity_affinity", "margin_threshold"}


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    init: float = 1.5,
    max_iter: int = 100,
    grid=None,
) -> float:
    """Fit a scalar temperature ``T>0`` minimizing NLL of ``softmax(logits/T)`` vs labels.

    LBFGS over ``log T`` (positivity by construction); falls back to a 1-D grid search if
    the optimizer produces a non-finite/degenerate value. Returns the scalar T.
    """
    logits = logits.detach().float()
    labels = labels.detach().long()
    if logits.numel() == 0:
        return 1.0

    log_t = torch.zeros(1, requires_grad=True)  # T = exp(log_t), init exp(0)=1; nudged below
    with torch.no_grad():
        log_t.fill_(float(torch.log(torch.tensor(init))))
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def _closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    try:
        opt.step(_closure)
        t = float(log_t.detach().exp().item())
        if not (t > 0) or t != t or t == float("inf"):  # NaN/inf/non-positive guard
            raise ValueError("degenerate T")
        return t
    except Exception:
        # robust fallback: grid search minimizing NLL
        grid = grid if grid is not None else torch.linspace(0.5, 5.0, 91)
        best_t, best_nll = 1.0, float("inf")
        for t in grid.tolist():
            nll = F.cross_entropy(logits / t, labels).item()
            if nll < best_nll:
                best_nll, best_t = nll, t
        return float(best_t)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide logits by a positive temperature (argmax invariant)."""
    t = float(temperature)
    if not (t > 0):
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / t


def confidence_gate(
    base_conf: torch.Tensor,
    top_affinity: torch.Tensor,
    margin: torch.Tensor,
    *,
    alpha_max: float = 1.0,
    mode: str = "fixed",
    threshold: float = 0.0,
) -> torch.Tensor:
    """Per-sample retrieval weight ``alpha_eff`` from confidence signals.

    Modes (all scaled by ``alpha_max``):
      - ``fixed``                : constant alpha_max (baseline).
      - ``unfamiliarity``        : alpha_max * (1 - base_conf)        — vote when base unsure.
      - ``familiarity``          : alpha_max * top_affinity           — vote when cache confident.
      - ``unfamiliarity_affinity``: alpha_max * (1-base_conf) * top_affinity — both.
      - ``margin_threshold``     : alpha_max where cache margin >= threshold else 0.
    """
    if mode not in VALID_GATE_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_GATE_MODES)}, got {mode!r}")
    base_conf = base_conf.float()
    top_affinity = top_affinity.float()
    margin = margin.float()
    a = float(alpha_max)
    if mode == "fixed":
        return torch.full_like(base_conf, a)
    if mode == "unfamiliarity":
        return a * (1.0 - base_conf).clamp(0.0, 1.0)
    if mode == "familiarity":
        return a * top_affinity.clamp(0.0, 1.0)
    if mode == "unfamiliarity_affinity":
        return a * (1.0 - base_conf).clamp(0.0, 1.0) * top_affinity.clamp(0.0, 1.0)
    # margin_threshold
    return a * (margin >= float(threshold)).float()


def calibration_metrics(logits: torch.Tensor, labels: torch.Tensor, *, n_bins: int = 15) -> dict:
    """{acc, ece, nll, brier} from logits — matches ``evaluate_dememte`` definitions."""
    logits = logits.detach().float()
    labels = labels.detach().long()
    probs = torch.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    correct = pred == labels
    y_prob = probs.gather(1, labels.view(-1, 1)).squeeze(1).clamp_min(1e-12)
    onehot = F.one_hot(labels, num_classes=probs.size(1)).float()
    brier = ((probs - onehot) ** 2).sum(dim=1)
    return {
        "acc": correct.float().mean().item(),
        "ece": _compute_ece(conf.cpu().tolist(), correct.cpu().tolist(), n_bins=n_bins),
        "nll": (-torch.log(y_prob)).mean().item(),
        "brier": brier.mean().item(),
    }
