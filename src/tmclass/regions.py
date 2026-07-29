"""Extraction of transmembrane regions, with flanking context."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .data import Protein

# Residues of flanking sequence kept on each side of the membrane segment. The
# flanks carry the charge asymmetry that distinguishes type I from type II
# topology (the positive-inside rule), so they are part of the signal, not noise.
DEFAULT_PADDING = 10


@dataclass(frozen=True)
class Region:
    """One transmembrane segment plus its flanks."""

    accession: str
    start: int          # 1-based, inclusive, in the parent sequence
    stop: int
    sequence: str
    tm_start: int       # 1-based coordinates of the membrane segment itself
    tm_stop: int

    @property
    def identifier(self) -> str:
        return f"{self.accession}/{self.start}-{self.stop}"

    @property
    def tm_length(self) -> int:
        return self.tm_stop - self.tm_start + 1


def extract_regions(
    proteins: list[Protein], padding: int = DEFAULT_PADDING
) -> list[Region]:
    """Slice every annotated transmembrane segment out, with padded flanks."""
    regions: list[Region] = []
    for protein in proteins:
        for tm_start, tm_stop in protein.tm_segments:
            start = max(1, tm_start - padding)
            stop = min(protein.length, tm_stop + padding)
            regions.append(
                Region(
                    accession=protein.accession,
                    start=start,
                    stop=stop,
                    sequence=protein.sequence[start - 1 : stop],
                    tm_start=tm_start,
                    tm_stop=tm_stop,
                )
            )
    return regions


def write_fasta(regions: list[Region], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for region in regions:
            handle.write(f">{region.identifier}\n{region.sequence}\n")
    return path


def parse_topcons(path: Path, padding: int = DEFAULT_PADDING) -> list[Region]:
    """Extract transmembrane regions from a TOPCONS2 result file.

    Retained so the original TOPCONS-based workflow stays reproducible, but the
    UniProt annotation path is the default: it needs no web-service submission
    and ships coordinates alongside the sequences.
    """
    regions: list[Region] = []
    accession: str | None = None
    sequence: str | None = None

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()

        if line.startswith("Sequence name:"):
            header = line.split("Sequence name:", 1)[1].strip()
            parts = header.split("|")
            accession = parts[1] if len(parts) > 1 else header.split()[0]

        elif line.startswith("Sequence:"):
            index += 1
            buffer: list[str] = []
            while index < len(lines) and not lines[index].startswith(
                ("TOPCONS predicted topology:", "Sequence name:")
            ):
                buffer.append(lines[index].strip())
                index += 1
            sequence = "".join(buffer)
            continue

        elif line.startswith("TOPCONS predicted topology:"):
            index += 1
            buffer = []
            while index < len(lines) and not lines[index].startswith(
                ("OCTOPUS predicted topology:", "Sequence name:")
            ):
                buffer.append(lines[index].strip())
                index += 1
            topology = "".join(buffer)

            if accession and sequence:
                for tm_start, tm_stop in _membrane_spans(topology):
                    start = max(1, tm_start - padding)
                    stop = min(len(sequence), tm_stop + padding)
                    regions.append(
                        Region(
                            accession=accession,
                            start=start,
                            stop=stop,
                            sequence=sequence[start - 1 : stop],
                            tm_start=tm_start,
                            tm_stop=tm_stop,
                        )
                    )
            accession = sequence = None
            continue

        index += 1

    return regions


def _membrane_spans(topology: str) -> list[tuple[int, int]]:
    """Find runs of 'M' in a TOPCONS topology string, as 1-based inclusive spans."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for position, character in enumerate(topology, start=1):
        if character == "M" and start is None:
            start = position
        elif character != "M" and start is not None:
            spans.append((start, position - 1))
            start = None
    if start is not None:
        spans.append((start, len(topology)))
    return spans
