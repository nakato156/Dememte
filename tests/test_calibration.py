"""CPU tests for wtq.8 calibration controls (temperature scaling + confidence gate)."""

from __future__ import annotations

import torch

from dememte.calibration import (
    apply_temperature,
    calibration_metrics,
    confidence_gate,
    fit_temperature,
)


def _overconfident_logits(n=4000, c=10, seed=0, scale=3.0):
    """Calibrated base logits (labels sampled from their softmax) scaled up -> overconfident.

    By construction the unscaled logits are well-calibrated, so the NLL-optimal temperature for
    the scaled version is ~`scale` (> 1). This is the regime temperature scaling is meant to fix.
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(n, c, generator=g) * 1.5
    labels = torch.multinomial(base.softmax(dim=1), 1, generator=g).squeeze(1)
    return base * scale, labels


def test_temperature_reduces_nll_and_is_gt_one_for_overconfident():
    logits, labels = _overconfident_logits()
    t = fit_temperature(logits, labels)
    assert t > 1.0  # over-confident -> softening temperature
    nll_before = calibration_metrics(logits, labels)["nll"]
    nll_after = calibration_metrics(apply_temperature(logits, t), labels)["nll"]
    assert nll_after < nll_before


def test_temperature_preserves_accuracy_and_lowers_ece():
    logits, labels = _overconfident_logits()
    t = fit_temperature(logits, labels)
    before = calibration_metrics(logits, labels)
    after = calibration_metrics(apply_temperature(logits, t), labels)
    assert abs(after["acc"] - before["acc"]) < 1e-9  # argmax invariant
    assert after["ece"] <= before["ece"] + 1e-9


def test_apply_temperature_rejects_nonpositive():
    logits = torch.randn(4, 3)
    for bad in (0.0, -1.0):
        try:
            apply_temperature(logits, bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_confidence_gate_modes():
    n = 5
    base_conf = torch.tensor([0.99, 0.5, 0.1, 0.8, 0.2])
    top_aff = torch.tensor([0.9, 0.7, 0.2, 0.6, 0.3])
    margin = torch.tensor([2.0, 0.5, 0.1, 1.0, 0.05])

    fixed = confidence_gate(base_conf, top_aff, margin, alpha_max=1.0, mode="fixed")
    assert torch.allclose(fixed, torch.ones(n))

    unf = confidence_gate(base_conf, top_aff, margin, alpha_max=2.0, mode="unfamiliarity")
    # high base_conf -> near-zero alpha; low base_conf -> larger alpha
    assert unf[0] < unf[2]
    assert torch.allclose(unf, 2.0 * (1 - base_conf))

    fam = confidence_gate(base_conf, top_aff, margin, alpha_max=1.0, mode="familiarity")
    assert torch.allclose(fam, top_aff)

    gated = confidence_gate(base_conf, top_aff, margin, alpha_max=1.0, mode="margin_threshold", threshold=0.6)
    assert torch.equal(gated, (margin >= 0.6).float())


def test_confidence_gate_rejects_unknown_mode():
    z = torch.zeros(3)
    try:
        confidence_gate(z, z, z, mode="bogus")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_offline_temp_plus_gate_composition_is_deterministic():
    # emulate the offline path: final = base + gate*cache, then /T
    g = torch.Generator().manual_seed(1)
    base = torch.randn(50, 10, generator=g)
    cache = torch.rand(50, 10, generator=g)  # affinity histogram (non-negative)
    base_conf = base.softmax(1).max(1).values
    top_aff = torch.rand(50, generator=g)
    margin = torch.rand(50, generator=g)
    labels = torch.randint(0, 10, (50,), generator=g)

    def run():
        alpha = confidence_gate(base_conf, top_aff, margin, alpha_max=1.0, mode="unfamiliarity")
        final = base + alpha.unsqueeze(1) * cache
        return calibration_metrics(apply_temperature(final, 1.3), labels)

    a, b = run(), run()
    assert a.keys() == b.keys()
    for k in a:  # per-key tolerance: the offline path is deterministic, but avoid brittle float == across builds
        assert abs(a[k] - b[k]) < 1e-9
