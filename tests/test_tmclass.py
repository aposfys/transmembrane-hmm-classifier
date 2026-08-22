"""Tests for parsing, splitting, region extraction and the evaluation metrics."""

from __future__ import annotations

import gzip
import hashlib
import http.client
import io
import json
import math
import urllib.error
from pathlib import Path

import pytest

from tmclass import data, evaluate, pipeline, regions

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


# --- UniProt parsing ---------------------------------------------------------


def _tsv(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "uniprot.tsv"
    path.write_text("Entry\tEntry Name\tSequence\tTransmembrane\n" + body, encoding="utf-8")
    return path


def test_transmembrane_coordinates_are_parsed(tmp_path):
    proteins = data.parse_tsv(
        _tsv(
            tmp_path,
            'P00001\tTEST_HUMAN\tMKVLA\tTRANSMEM 2..4; /note="Helical"\n',
        )
    )
    assert proteins[0].tm_segments == ((2, 4),)


def test_multiple_transmembrane_segments_are_parsed(tmp_path):
    proteins = data.parse_tsv(
        _tsv(
            tmp_path,
            "P00001\tTEST_HUMAN\tMKVLAMKVLA\tTRANSMEM 2..4; TRANSMEM 7..9\n",
        )
    )
    assert proteins[0].tm_segments == ((2, 4), (7, 9))


def test_entries_are_sorted_for_determinism(tmp_path):
    proteins = data.parse_tsv(
        _tsv(
            tmp_path,
            "P00009\tB_HUMAN\tMKVLA\tTRANSMEM 2..4\nP00001\tA_HUMAN\tMKVLA\tTRANSMEM 2..4\n",
        )
    )
    assert [p.accession for p in proteins] == ["P00001", "P00009"]


def test_single_tm_filter_removes_multipass(tmp_path):
    proteins = data.parse_tsv(
        _tsv(
            tmp_path,
            "P00001\tONE_HUMAN\tMKVLA\tTRANSMEM 2..4\n"
            "P00002\tTWO_HUMAN\tMKVLAMKVLA\tTRANSMEM 2..4; TRANSMEM 7..9\n",
        )
    )
    kept = data.single_tm_only(proteins)
    assert [p.accession for p in kept] == ["P00001"]


def test_limit_truncates_after_sorting(tmp_path):
    """The subset must be deterministic, so truncation happens after the sort."""
    body = "".join(f"P0000{i}\tE{i}_HUMAN\tMKVLA\tTRANSMEM 2..4\n" for i in (5, 1, 9, 3))
    path = _tsv(tmp_path, body)
    assert [p.accession for p in data.parse_tsv(path, limit=2)] == ["P00001", "P00003"]
    assert len(data.parse_tsv(path)) == 4


def test_limit_of_none_keeps_everything(tmp_path):
    path = _tsv(tmp_path, "P00001\tA_HUMAN\tMKVLA\tTRANSMEM 2..4\n")
    assert len(data.parse_tsv(path, limit=None)) == 1


def test_entries_without_a_sequence_are_dropped(tmp_path):
    proteins = data.parse_tsv(_tsv(tmp_path, "P00001\tTEST_HUMAN\t\tTRANSMEM 2..4\n"))
    assert proteins == []


# --- UniProt downloads -------------------------------------------------------


class _Response(io.BytesIO):
    """Minimal stand-in for the urlopen context manager."""

    def __init__(self, payload: bytes = b"", link: str = "") -> None:
        super().__init__(payload)
        self.headers = {"Link": link}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_) -> None:
        self.close()


def _failing_opener(errors: list[Exception], payload: bytes = b"ok"):
    """An urlopen replacement that raises each error in turn, then succeeds."""
    attempts: list[str] = []

    def opener(url, timeout=None):
        attempts.append(url)
        index = len(attempts) - 1
        if index < len(errors) and errors[index] is not None:
            raise errors[index]
        return _Response(payload)

    return opener, attempts


def test_a_truncated_stream_is_retried(monkeypatch):
    """UniProt cuts chunked responses off mid-transfer; that must not abort a run."""
    opener, attempts = _failing_opener([http.client.IncompleteRead(b"partial"), None])
    monkeypatch.setattr(data.urllib.request, "urlopen", opener)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)

    assert data._fetch("https://example.invalid/stream") == b"ok"
    assert len(attempts) == 2


def test_a_network_error_is_retried(monkeypatch):
    opener, attempts = _failing_opener([urllib.error.URLError("connection reset"), None])
    monkeypatch.setattr(data.urllib.request, "urlopen", opener)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)

    assert data._fetch("https://example.invalid/stream") == b"ok"
    assert len(attempts) == 2


def test_fetch_gives_up_after_the_retry_budget(monkeypatch):
    opener, attempts = _failing_opener([http.client.IncompleteRead(b"") for _ in range(5)])
    monkeypatch.setattr(data.urllib.request, "urlopen", opener)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="after 5 attempts"):
        data._fetch("https://example.invalid/stream", retries=5)
    assert len(attempts) == 5


# --- pagination and the snapshot fallback ------------------------------------


def _paged_opener(pages: list[tuple[str, str]]):
    """Serve canned (body, Link header) pairs in order."""
    served: list[str] = []

    def opener(url, timeout=None):
        body, link = pages[len(served)]
        served.append(url)
        return _Response(body.encode(), link)

    return opener, served


def test_pagination_follows_cursor_links_and_drops_repeated_headers(monkeypatch):
    header = "Entry\tEntry Name\tSequence\tTransmembrane"
    opener, served = _paged_opener(
        [
            (f"{header}\nP00001\tA\tMKV\tTRANSMEM 1..2", '<https://next/2>; rel="next"'),
            (f"{header}\nP00002\tB\tMKV\tTRANSMEM 1..2", ""),
        ]
    )
    monkeypatch.setattr(data.urllib.request, "urlopen", opener)

    payload = data._fetch_paginated("type_i").decode()

    assert payload.count("Entry\tEntry Name") == 1
    assert "P00001" in payload and "P00002" in payload
    assert served[1] == "https://next/2"


def _snapshot(tmp_path: Path, name: str, payload: bytes, date: str = "2026-08-21"):
    (tmp_path / f"{name}.tsv.gz").write_bytes(gzip.compress(payload))
    (tmp_path / "MANIFEST.json").write_text(
        json.dumps(
            {
                "date": date,
                "classes": {name: {"sha256": hashlib.sha256(payload).hexdigest()}},
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_snapshot_is_used_when_uniprot_is_unreachable(tmp_path, monkeypatch):
    payload = b"Entry\tEntry Name\tSequence\tTransmembrane\nP00001\tA\tMKV\tTRANSMEM 1..2\n"
    snap = tmp_path / "snap"
    snap.mkdir()
    _snapshot(snap, "type_i", payload)
    monkeypatch.setattr(data, "SNAPSHOT_DIR", snap)
    monkeypatch.setattr(data, "PROVENANCE", {})

    def dead(url, timeout=None):
        raise urllib.error.URLError("uniprot is down")

    monkeypatch.setattr(data.urllib.request, "urlopen", dead)
    monkeypatch.setattr(data.time, "sleep", lambda _: None)

    destination = data.fetch_class("type_i", tmp_path / "out" / "type_i.tsv")

    assert destination.read_bytes() == payload
    assert data.PROVENANCE["type_i"]["source"] == "snapshot"
    assert data.PROVENANCE["type_i"]["snapshot_date"] == "2026-08-21"


def test_snapshot_fallback_can_be_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        lambda url, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )

    with pytest.raises(RuntimeError, match="UniProt request failed"):
        data.fetch_class("type_i", tmp_path / "type_i.tsv", allow_snapshot=False)


def test_a_corrupt_snapshot_is_rejected(tmp_path, monkeypatch):
    snap = tmp_path / "snap"
    snap.mkdir()
    _snapshot(snap, "type_i", b"real payload")
    (snap / "type_i.tsv.gz").write_bytes(gzip.compress(b"tampered payload"))
    monkeypatch.setattr(data, "SNAPSHOT_DIR", snap)

    with pytest.raises(RuntimeError, match="corrupt"):
        data.read_snapshot("type_i")


def test_a_stalling_uniprot_falls_back_within_its_budget(tmp_path, monkeypatch):
    """A hang must reach the snapshot, not outlast the job that would use it."""
    payload = b"Entry\tEntry Name\tSequence\tTransmembrane\nP00001\tA\tMKV\tTRANSMEM 1..2\n"
    snap = tmp_path / "snap"
    snap.mkdir()
    _snapshot(snap, "type_i", payload)
    monkeypatch.setattr(data, "SNAPSHOT_DIR", snap)
    monkeypatch.setattr(data, "PROVENANCE", {})

    # A clock that jumps 45s per reading, so the 120s budget lapses after a few
    # attempts without the test actually waiting.
    ticks = iter(range(0, 100_000, 45))
    monkeypatch.setattr(data.time, "monotonic", lambda: float(next(ticks)))
    monkeypatch.setattr(data.time, "sleep", lambda _: None)

    calls: list[str] = []

    def stalling(url, timeout=None):
        calls.append(url)
        raise TimeoutError("read timed out")

    monkeypatch.setattr(data.urllib.request, "urlopen", stalling)

    data.fetch_class("type_i", tmp_path / "out.tsv", budget=120)

    assert data.PROVENANCE["type_i"]["source"] == "snapshot"
    # The point is that it stopped early rather than retrying indefinitely.
    assert len(calls) < 6, f"kept trying past the budget: {len(calls)} requests"


def test_the_budget_stops_pagination_partway(tmp_path, monkeypatch):
    """A half-walked class must not be returned as if it were complete."""
    header = "Entry\tEntry Name\tSequence\tTransmembrane"
    opener, _ = _paged_opener(
        [(f"{header}\nP00001\tA\tMKV\tTRANSMEM 1..2", '<https://next/2>; rel="next"')] * 5
    )
    monkeypatch.setattr(data.urllib.request, "urlopen", opener)
    ticks = iter([0.0, 10.0, 500.0, 500.0, 500.0])
    monkeypatch.setattr(data.time, "monotonic", lambda: next(ticks))

    with pytest.raises(RuntimeError, match="budget exhausted"):
        data._fetch_paginated("type_i", deadline=100.0)


def test_a_missing_snapshot_says_how_to_rebuild_it(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "SNAPSHOT_DIR", tmp_path / "absent")
    monkeypatch.setattr(data.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        data.urllib.request,
        "urlopen",
        lambda url, timeout=None: (_ for _ in ()).throw(urllib.error.URLError("down")),
    )

    with pytest.raises(RuntimeError, match="make snapshot"):
        data.fetch_class("type_i", tmp_path / "type_i.tsv")


def test_live_uniprot_is_recorded_as_the_source(tmp_path, monkeypatch):
    payload = b"Entry\tEntry Name\tSequence\tTransmembrane\nP00001\tA\tMKV\tTRANSMEM 1..2\n"
    monkeypatch.setattr(data, "PROVENANCE", {})
    monkeypatch.setattr(
        data.urllib.request, "urlopen", lambda url, timeout=None: _Response(payload)
    )

    data.fetch_class("type_i", tmp_path / "type_i.tsv")

    assert data.PROVENANCE["type_i"] == {"source": "uniprot"}


# --- region extraction -------------------------------------------------------


def _protein(sequence: str, segments: tuple[tuple[int, int], ...]) -> data.Protein:
    return data.Protein(
        accession="P00001", entry_name="TEST_HUMAN", sequence=sequence, tm_segments=segments
    )


def test_padding_is_applied_on_both_sides():
    protein = _protein("A" * 50, ((21, 40),))
    region = regions.extract_regions([protein], padding=10)[0]
    assert (region.start, region.stop) == (11, 50)
    assert len(region.sequence) == 40


def test_padding_is_clipped_at_the_termini():
    """A TM segment at the very start must not produce a negative index."""
    protein = _protein("A" * 30, ((1, 20),))
    region = regions.extract_regions([protein], padding=10)[0]
    assert region.start == 1
    assert region.sequence == "A" * 30


def test_region_sequence_matches_its_coordinates():
    sequence = "".join(chr(65 + i % 26) for i in range(100))
    protein = _protein(sequence, ((30, 50),))
    region = regions.extract_regions([protein], padding=10)[0]
    assert region.sequence == sequence[region.start - 1 : region.stop]
    assert region.tm_length == 21


def test_topcons_membrane_spans():
    assert regions._membrane_spans("iiMMMoooMMi") == [(3, 5), (9, 10)]
    assert regions._membrane_spans("MMM") == [(1, 3)]
    assert regions._membrane_spans("iii") == []


# --- splitting ---------------------------------------------------------------


def test_split_is_a_partition():
    identifiers = [f"seq{i}" for i in range(100)]
    split = pipeline.split_identifiers(identifiers, test_fraction=0.2, seed=1)
    assert len(split.train) + len(split.test) == 100
    assert set(split.train) | set(split.test) == set(identifiers)


def test_split_is_reproducible():
    identifiers = [f"seq{i}" for i in range(100)]
    first = pipeline.split_identifiers(identifiers, seed=42)
    second = pipeline.split_identifiers(identifiers, seed=42)
    assert first == second


def test_split_rejects_an_impossible_fraction():
    with pytest.raises(ValueError):
        pipeline.split_identifiers(["a", "b"], test_fraction=1.0)


def test_overlapping_partitions_are_rejected():
    with pytest.raises(ValueError, match="both partitions"):
        pipeline.Split(train=("a", "b"), test=("b",))


def test_subset_fasta_errors_rather_than_dropping(tmp_path):
    """Silently skipping unmatched identifiers is what corrupted the original split."""
    fasta = tmp_path / "in.fasta"
    fasta.write_text(">seq1\nMKV\n>seq2\nMKV\n", encoding="utf-8")
    with pytest.raises(KeyError, match="not found"):
        pipeline.subset_fasta(fasta, {"seq1", "missing"}, tmp_path / "out.fasta")


# --- hmmsearch parsing -------------------------------------------------------


def test_tblout_keeps_the_best_evalue_per_target(tmp_path):
    path = tmp_path / "hits.tbl"
    path.write_text(
        "# comment line\n"
        "targetA - queryA - 1e-10 50.0 0.0 1e-9 49.0 0.0 1.0 1 0 0 1 1 1 1 desc\n"
        "targetA - queryB - 1e-20 90.0 0.0 1e-19 89.0 0.0 1.0 1 0 0 1 1 1 1 desc\n"
        "targetB - queryA - 0.5 10.0 0.0 0.6 9.0 0.0 1.0 1 0 0 1 1 1 1 desc\n",
        encoding="utf-8",
    )
    hits = pipeline.parse_tblout(path)
    assert hits == {"targetA": 1e-20, "targetB": 0.5}


# --- metrics -----------------------------------------------------------------


def test_confusion_matrix_at_a_threshold():
    metrics = evaluate.evaluate_at([1e-10, 1e-5, 1.0], [1e-8, 10.0, 20.0], threshold=0.05)
    assert (metrics.true_positives, metrics.false_negatives) == (2, 1)
    assert (metrics.false_positives, metrics.true_negatives) == (1, 2)


def test_perfect_classifier():
    metrics = evaluate.evaluate_at([1e-10, 1e-9], [10.0, 20.0], threshold=0.05)
    assert metrics.sensitivity == 1.0
    assert metrics.specificity == 1.0
    assert metrics.precision == 1.0
    assert metrics.mcc == 1.0


def test_mcc_is_zero_for_a_coin_flip():
    metrics = evaluate.Metrics(
        threshold=0.05,
        true_positives=25,
        false_positives=25,
        true_negatives=25,
        false_negatives=25,
    )
    assert metrics.mcc == pytest.approx(0.0)


def test_precision_exposes_what_specificity_hides():
    """9 positives against 121 negatives: 87% specificity still means 16 false
    positives, so precision is 0.30 even though sensitivity and specificity
    both look respectable. This is the metric the original evaluation omitted."""
    metrics = evaluate.Metrics(
        threshold=0.05,
        true_positives=7,
        false_positives=16,
        true_negatives=105,
        false_negatives=2,
    )
    assert metrics.sensitivity == pytest.approx(0.778, abs=1e-3)
    assert metrics.specificity == pytest.approx(0.868, abs=1e-3)
    assert metrics.precision == pytest.approx(0.304, abs=1e-3)


def test_non_hits_count_as_negatives_not_missing_data():
    scores = evaluate.scores_for(["a", "b", "c"], {"a": 1e-10})
    assert scores[0] == 1e-10
    assert all(math.isinf(score) for score in scores[1:])

    metrics = evaluate.evaluate_at(scores, [], threshold=0.05)
    assert metrics.true_positives == 1
    assert metrics.false_negatives == 2


def test_roc_auc_of_a_perfect_separation():
    curve = evaluate.sweep([1e-10, 1e-9], [10.0, 20.0])
    assert evaluate.roc_auc(curve) == pytest.approx(1.0)


def test_roc_auc_of_chance_is_near_one_half():
    curve = evaluate.sweep([1.0, 3.0], [2.0, 4.0])
    assert evaluate.roc_auc(curve) == pytest.approx(0.5, abs=0.3)


def test_sweep_covers_every_distinct_score():
    curve = evaluate.sweep([1e-10, 1e-5], [1e-8, 1.0])
    assert len(curve) == 5  # four distinct scores plus the empty-prediction point
    assert curve[0].true_positives == 0


def test_best_by_maximises_the_criterion():
    curve = evaluate.sweep([1e-10, 1e-9], [10.0, 20.0])
    assert evaluate.best_by(curve, "mcc").mcc == max(m.mcc for m in curve)


# --- end-to-end results ------------------------------------------------------


@pytest.fixture(scope="module")
def findings():
    import json

    path = RESULTS / "findings.json"
    if not path.exists():
        pytest.skip("run `make analysis` first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_tm_region_model_beats_full_length(findings):
    """Compare on threshold-independent measures and on MCC.

    Precision is *not* the right comparison here: each model is scored at its
    own MCC-optimal threshold, and the TM-region model deliberately trades
    precision for a large gain in sensitivity.
    """
    assert findings["tm_region"]["roc_auc"] > findings["full_length"]["roc_auc"]
    assert (
        findings["tm_region"]["average_precision"]
        > findings["full_length"]["average_precision"]
    )
    assert (
        findings["tm_region"]["at_optimal_cutoff"]["mcc"]
        > findings["full_length"]["at_optimal_cutoff"]["mcc"]
    )


def test_sweeping_the_threshold_beats_the_conventional_cutoff(findings):
    """E <= 0.05 is a homology-search convention, not an optimum for this task."""
    for model in ("full_length", "tm_region"):
        conventional = findings[model]["at_conventional_cutoff"]
        optimal = findings[model]["at_optimal_cutoff"]
        assert optimal["mcc"] > conventional["mcc"]
        assert optimal["sensitivity"] > conventional["sensitivity"]


def test_globular_decoys_are_separated_perfectly(findings):
    """No transmembrane segment means the TM-region model should never fire."""
    by_class = findings["tm_region"]["false_positives_by_class"]
    assert by_class["globular"]["false_positives"] == 0


def test_type_ii_is_the_hardest_decoy_class(findings):
    """Type II shares the single-helix architecture and differs only in orientation."""
    by_class = findings["tm_region"]["false_positives_by_class"]
    rates = {name: entry["false_positive_rate"] for name, entry in by_class.items()}
    assert max(rates, key=rates.get) == "type_ii"


def test_cross_validation_is_stable(findings):
    mcc = findings["tm_region"]["cross_validation"]["summary"]["mcc"]
    assert mcc["sd"] < 0.15, "fold-to-fold MCC varies more than reported"
