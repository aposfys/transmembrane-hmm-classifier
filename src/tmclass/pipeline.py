"""Redundancy reduction, dataset splitting, and the HMMER model itself."""

from __future__ import annotations

import hashlib
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO

# Sequence-identity ceiling for CD-HIT. Homologues above this level would appear
# in both the training and test sets and inflate every metric.
IDENTITY_THRESHOLD = 0.4


def _require(tool: str, install_hint: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise RuntimeError(f"{tool} not found on PATH. Install it with: {install_hint}")
    return path


def _run(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _is_fresh(output: Path, source: Path) -> bool:
    """True when ``output`` exists, is non-empty and postdates ``source``."""
    return (
        output.exists()
        and output.stat().st_size > 0
        and source.exists()
        and output.stat().st_mtime >= source.stat().st_mtime
    )


# --- redundancy reduction ----------------------------------------------------


def cluster(
    fasta: Path,
    output: Path,
    identity: float = IDENTITY_THRESHOLD,
    word_size: int | None = None,
) -> Path:
    """Collapse near-duplicate sequences with CD-HIT, keeping one per cluster.

    Cached against a checksum of the input FASTA and the parameters, stored next
    to the output. Keying the cache on the output path alone let a clustering of
    a different input (a truncated smoke run, or type I before the single-TM
    filter) be reused silently by a later full run.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    # CD-HIT requires a word size compatible with the identity threshold.
    if word_size is None:
        word_size = 2 if identity < 0.5 else 3 if identity < 0.6 else 5

    key = hashlib.sha256(fasta.read_bytes())
    key.update(f"identity={identity} word_size={word_size}".encode())
    stamp = output.with_name(output.name + ".input.sha256")
    if (
        output.exists()
        and output.stat().st_size > 0
        and stamp.exists()
        and stamp.read_text(encoding="utf-8").strip() == key.hexdigest()
    ):
        return output
    _run(
        [
            _require("cd-hit", "conda install -c bioconda cd-hit"),
            "-i",
            str(fasta),
            "-o",
            str(output),
            "-c",
            str(identity),
            "-n",
            str(word_size),
            "-d",
            "0",  # keep full FASTA headers rather than truncating at 20 chars
        ]
    )
    stamp.write_text(key.hexdigest() + "\n", encoding="utf-8")
    return output


# --- splitting ---------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """A train/test partition of sequence identifiers."""

    train: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.test)
        if overlap:
            raise ValueError(f"{len(overlap)} identifiers appear in both partitions")


def split_identifiers(
    identifiers: list[str], test_fraction: float = 0.2, seed: int = 20250101
) -> Split:
    """Shuffle and partition identifiers reproducibly.

    Unlike a dictionary lookup keyed on truncated FASTA headers, this never
    silently discards sequences: the two partitions always sum to the input.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1")

    shuffled = list(identifiers)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - test_fraction))
    return Split(train=tuple(shuffled[:cut]), test=tuple(shuffled[cut:]))


def subset_fasta(fasta: Path, identifiers: set[str], output: Path) -> Path:
    """Write out the records whose id is in the given set, erroring on misses."""
    output.parent.mkdir(parents=True, exist_ok=True)
    records = {record.id: record for record in SeqIO.parse(fasta, "fasta")}

    missing = identifiers - records.keys()
    if missing:
        raise KeyError(
            f"{len(missing)} identifiers not found in {fasta.name}, e.g. {sorted(missing)[:3]}"
        )

    SeqIO.write([records[key] for key in sorted(identifiers)], output, "fasta")
    return output


def fasta_identifiers(fasta: Path) -> list[str]:
    return [record.id for record in SeqIO.parse(fasta, "fasta")]


# --- alignment and model -----------------------------------------------------


def align(fasta: Path, output: Path, threads: int = 1, cache: bool = True) -> Path:
    """Align sequences with Clustal Omega, in Stockholm format for hmmbuild.

    Whole-protein alignment is quadratic in sequence length and dominates the
    runtime, so an existing alignment newer than its input is reused.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    if cache and _is_fresh(output, fasta):
        return output

    base = [
        _require("clustalo", "conda install -c bioconda clustalo"),
        "-i",
        str(fasta),
        "-o",
        str(output),
        "--outfmt",
        "st",
        "--force",
    ]
    # Bioconda's macOS build is compiled without OpenMP and aborts on --threads.
    for command in ([*base, "--threads", str(threads)], base) if threads > 1 else (base,):
        try:
            _run(command)
            return output
        except subprocess.CalledProcessError as error:
            if "without OpenMP" not in (error.stderr or ""):
                raise
    raise RuntimeError("clustalo failed")


def build_hmm(alignment: Path, output: Path, name: str, cache: bool = True) -> Path:
    """Build a profile HMM from a multiple sequence alignment."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if cache and _is_fresh(output, alignment):
        return output
    _run(
        [
            _require("hmmbuild", "conda install -c bioconda hmmer"),
            "--amino",
            "-n",
            name,
            str(output),
            str(alignment),
        ]
    )
    return output


def search(hmm: Path, target_fasta: Path, output: Path, threads: int = 1) -> Path:
    """Score every sequence in a FASTA file against a profile HMM.

    The E-value hmmsearch reports is scaled by the number of sequences in the
    target file, so E-values from two separate searches are not comparable.
    Anything that ranks sequences against each other should go through
    :func:`search_together`.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _require("hmmsearch", "conda install -c bioconda hmmer"),
            "--tblout",
            str(output),
            "--max",  # disable heuristic filters, so weak hits are still scored
            "-E",
            "1000",  # report generously; thresholding happens at evaluation time
            "--cpu",
            str(threads),
            str(hmm),
            str(target_fasta),
        ]
    )
    return output


def concatenate_fasta(fastas: list[Path], output: Path) -> Path:
    """Write the records of several FASTA files into one, in order."""
    output.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(
        [record for fasta in fastas for record in SeqIO.parse(fasta, "fasta")],
        output,
        "fasta",
    )
    return output


def search_together(
    hmm: Path, fastas: list[Path], combined: Path, output: Path, threads: int = 1
) -> dict[str, float]:
    """Score positives and decoys in one hmmsearch run and return their E-values.

    hmmsearch multiplies each P-value by the size of the target file. Searching
    358 positives and 888 decoys as two files put the two classes on E-value
    scales 2.5x apart, which shifted every threshold and inflated the ROC AUC.
    One search over the combined set gives every sequence the same scale.
    """
    return parse_tblout(
        search(hmm, concatenate_fasta(fastas, combined), output, threads=threads)
    )


# --- parsing hmmsearch output ------------------------------------------------

_WHITESPACE = re.compile(r"\s+")


def parse_tblout(path: Path) -> dict[str, float]:
    """Map each target sequence to its best full-sequence E-value.

    hmmsearch emits one row per query/target pair; a target absent from the file
    scored below the reporting threshold and is treated as a non-hit by the
    caller rather than being silently skipped.
    """
    best: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        fields = _WHITESPACE.split(line.strip())
        if len(fields) < 5:
            continue
        target, evalue = fields[0], float(fields[4])
        if evalue < best.get(target, float("inf")):
            best[target] = evalue
    return best
