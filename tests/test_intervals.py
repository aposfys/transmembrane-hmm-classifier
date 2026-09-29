"""Tests for the confidence intervals attached to every reported number."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmclass.intervals import (
    hanley_mcneil,
    separated,
    summarise,
    wilson,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "findings.json"


def test_wilson_matches_a_published_value():
    """Wilson interval for 0.5 at n=100, checked against the closed form."""
    interval = wilson(50, 100)
    assert interval.estimate == pytest.approx(0.5)
    assert interval.lower == pytest.approx(0.4038, abs=1e-3)
    assert interval.upper == pytest.approx(0.5962, abs=1e-3)


def test_wilson_gives_zero_counts_a_real_upper_bound():
    """The reason Wilson is used rather than the normal approximation.

    Observing 0 false positives on 172 GPCR decoys is not evidence the rate is
    zero, and a normal-approximation interval would report [0, 0] and imply it
    was. This project quotes exactly that count.
    """
    interval = wilson(0, 172)
    assert interval.estimate == 0.0
    assert interval.lower == 0.0
    assert 0.0 < interval.upper < 0.05


def test_wilson_stays_inside_the_unit_interval():
    for successes, trials in ((0, 5), (5, 5), (1, 3), (999, 1000)):
        interval = wilson(successes, trials)
        assert 0.0 <= interval.lower <= interval.estimate <= interval.upper <= 1.0


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson(5, 0)
    with pytest.raises(ValueError):
        wilson(6, 5)


def test_hanley_mcneil_narrows_as_the_sample_grows():
    small = hanley_mcneil(0.9, 50, 50)
    large = hanley_mcneil(0.9, 500, 500)
    assert (large.upper - large.lower) < (small.upper - small.lower)


def test_hanley_mcneil_rejects_impossible_input():
    with pytest.raises(ValueError):
        hanley_mcneil(1.5, 10, 10)
    with pytest.raises(ValueError):
        hanley_mcneil(0.9, 0, 10)


def test_separated_is_symmetric_and_strict():
    a = wilson(10, 100)
    b = wilson(90, 100)
    assert separated(a, b) and separated(b, a)
    assert not separated(a, a)


@pytest.fixture(scope="module")
def report():
    if not RESULTS.exists():
        pytest.skip(f"{RESULTS} missing; run the pipeline first")
    return summarise(json.loads(RESULTS.read_text(encoding="utf-8")))


def test_the_language_model_advantage_survives_intervals(report):
    """The headline claim, with its uncertainty attached."""
    pairs = {(p["a"], p["b"]): p["separated"] for p in report["roc_auc_pairs"]}
    assert pairs[("Profile HMM, TM regions", "ESM-2 + logistic regression")]
    assert pairs[("Profile HMM, TM regions", "ESM-2 + MLP")]


def test_the_two_esm_heads_are_not_distinguishable(report):
    """Their AUC intervals overlap, so neither head can be called the better one.

    Reporting one of them as the winner on a fourth decimal place would be
    reading noise.
    """
    pairs = {(p["a"], p["b"]): p["separated"] for p in report["roc_auc_pairs"]}
    assert not pairs[("ESM-2 + logistic regression", "ESM-2 + MLP")]


def test_type_ii_is_the_hard_class_only_for_the_esm_heads(report):
    """The claim that type II decoys are the hard class is model-dependent.

    Both ESM heads confuse type II proteins significantly more than GPCRs and
    globular decoys. The TM-region HMM confuses type II more than globular
    decoys but not separably more than GPCRs, and the full-length HMM
    separates none of its three decoy rates. An earlier scoring put positives
    and decoys on different E-value scales and made the full-length HMM look
    as if it confused GPCRs significantly more than type II. On one scale that
    inversion is gone, and this pins it so it cannot quietly come back.
    """
    by_model = {row["model"]: row for row in report["decoy_difficulty_ordering"]}
    full_length = by_model["Profile HMM, full-length"]
    assert not full_length["harder_than"]
    assert not full_length["confused_more_than_type_ii"]
    assert by_model["Profile HMM, TM regions"]["harder_than"] == ["globular"]
    assert by_model["Profile HMM, TM regions"]["not_separated_from"] == ["gpcr"]
    for label in ("ESM-2 + logistic regression", "ESM-2 + MLP"):
        assert "gpcr" in by_model[label]["harder_than"]
        assert not by_model[label]["confused_more_than_type_ii"]


def test_the_two_hmms_differ(report):
    """The TM-region HMM beats the full-length one on AUC, and the intervals agree."""
    pairs = {(p["a"], p["b"]): p["separated"] for p in report["roc_auc_pairs"]}
    assert pairs[("Profile HMM, full-length", "Profile HMM, TM regions")]
