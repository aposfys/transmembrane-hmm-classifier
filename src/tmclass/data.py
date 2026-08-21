"""Dataset construction from UniProt.

Every sequence class is defined by an explicit UniProt query, so the whole
dataset rebuilds from scratch with one command and no manual downloads.
Transmembrane coordinates come from UniProt's own ``ft_transmem`` annotation,
which removes the dependency on an external topology-prediction web service.
"""

from __future__ import annotations

import csv
import http.client
import io
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

# UniProt subcellular-location and family terms defining each class.
#   SL-9905  single-pass type I membrane protein   (the positive class)
#   SL-9906  single-pass type II membrane protein  (same topology family, opposite orientation)
#   SL-0091  cytoplasm
QUERIES: dict[str, str] = {
    "type_i": "(reviewed:true) AND (cc_scl_term:SL-9905) AND (ft_transmem:*)",
    "type_ii": "(reviewed:true) AND (cc_scl_term:SL-9906) AND (ft_transmem:*)",
    "gpcr": '(reviewed:true) AND (family:"g-protein coupled receptor 1 family")',
    "globular": (
        "(reviewed:true) AND (cc_scl_term:SL-0091) NOT (ft_transmem:*) AND (existence:1)"
    ),
}

# The three negative classes, chosen to probe different failure modes:
# type II shares the single-TM architecture with inverted orientation, GPCRs are
# polytopic membrane proteins, and globular proteins have no membrane segment.
NEGATIVE_CLASSES = ("type_ii", "gpcr", "globular")

TRANSMEM_PATTERN = re.compile(r"TRANSMEM (\d+)\.\.(\d+)")


@dataclass(frozen=True)
class Protein:
    """A UniProt entry with its annotated transmembrane segments."""

    accession: str
    entry_name: str
    sequence: str
    tm_segments: tuple[tuple[int, int], ...]

    @property
    def header(self) -> str:
        return f"sp|{self.accession}|{self.entry_name}"

    @property
    def length(self) -> int:
        return len(self.sequence)


# The stream endpoint sends chunked responses and regularly cuts one off
# part-way through, which surfaces as http.client.IncompleteRead rather than a
# URLError. IncompleteRead subclasses HTTPException, so catching URLError alone
# lets a truncated download escape the retry loop and abort the whole run.
_TRANSIENT_ERRORS = (urllib.error.URLError, http.client.HTTPException, TimeoutError)


def _fetch(url: str, retries: int = 5) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as response:
                return response.read()
        except _TRANSIENT_ERRORS as error:
            last_error = error
            if attempt < retries - 1:
                time.sleep(2.0**attempt)
    raise RuntimeError(f"UniProt request failed after {retries} attempts: {last_error}")


def fetch_class(name: str, destination: Path) -> Path:
    """Download one sequence class as TSV, including transmembrane features."""
    if name not in QUERIES:
        raise KeyError(f"Unknown class {name!r}; expected one of {sorted(QUERIES)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    params = urllib.parse.urlencode(
        {
            "query": QUERIES[name],
            "fields": "accession,id,sequence,ft_transmem",
            "format": "tsv",
        }
    )
    destination.write_bytes(_fetch(f"{UNIPROT_STREAM}?{params}"))
    return destination


def parse_tsv(path: Path, limit: int | None = None) -> list[Protein]:
    """Parse a UniProt TSV export into Protein records, sorted by accession.

    Sorting makes every downstream split deterministic regardless of the order
    UniProt happened to stream the results in.

    Args:
        limit: Keep only the first N entries after sorting. Intended for smoke
            runs that need to exercise the whole pipeline quickly; CD-HIT on the
            full type I set alone takes about 40 minutes. Truncating after the
            sort keeps the subset deterministic and reproducible.
    """
    reader = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")), delimiter="\t")
    proteins = [
        Protein(
            accession=row["Entry"],
            entry_name=row["Entry Name"],
            sequence=row["Sequence"],
            tm_segments=tuple(
                (int(start), int(stop))
                for start, stop in TRANSMEM_PATTERN.findall(row.get("Transmembrane") or "")
            ),
        )
        for row in reader
        if row.get("Sequence")
    ]
    proteins.sort(key=lambda protein: protein.accession)
    return proteins[:limit] if limit else proteins


def write_fasta(proteins: list[Protein], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for protein in proteins:
            handle.write(f">{protein.header}\n")
            for index in range(0, len(protein.sequence), 60):
                handle.write(protein.sequence[index : index + 60] + "\n")
    return path


def single_tm_only(proteins: list[Protein]) -> list[Protein]:
    """Keep entries with exactly one annotated transmembrane segment.

    UniProt occasionally annotates a second segment on an entry whose
    subcellular-location term still says single-pass. Those entries contradict
    the class definition and are removed rather than silently modelled.
    """
    return [protein for protein in proteins if len(protein.tm_segments) == 1]
