"""Dataset construction from UniProt.

Every sequence class is defined by an explicit UniProt query, so the whole
dataset rebuilds from scratch with one command and no manual downloads.
Transmembrane coordinates come from UniProt's own ``ft_transmem`` annotation,
which removes the dependency on an external topology-prediction web service.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import http.client
import io
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

# Records per page when falling back to the paginated search endpoint. Small
# enough that one slow page does not exhaust the timeout, large enough that a
# full class stays well under a hundred requests.
PAGE_SIZE = 200

# Seconds allowed on one HTTP request before it is retried. A degraded UniProt
# accepts a connection and then stalls, so this is what turns a hang into a
# retry rather than a wait.
REQUEST_TIMEOUT = 60

# Total seconds of live UniProt attempts per class before giving up and using
# the snapshot. Without a ceiling the tiers below cannot be reached in practice:
# retrying a stalled stream five times, then paginating a slow class, ran past a
# 30-minute CI limit and the job was killed before any fallback happened. A
# fallback that is only reachable in unbounded time is not a fallback.
#
# The deadline is only checked between attempts, so a class can overshoot by up
# to one REQUEST_TIMEOUT: budget the worst case as LIVE_BUDGET + REQUEST_TIMEOUT
# per class when sizing a CI limit.
LIVE_BUDGET = 300

FIELDS = "accession,id,sequence,ft_transmem"

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshot"

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

_NEXT_LINK = re.compile(r'<(.+?)>;\s*rel="next"')

# Where each class came from, keyed by class name. build_dataset copies this
# into findings.json so a result is never traceable to stale data by accident.
PROVENANCE: dict[str, dict[str, str]] = {}


def _request(url: str, retries: int = 5, deadline: float | None = None) -> tuple[bytes, str]:
    """Fetch one URL with backoff, returning the body and its Link header.

    ``deadline`` is a time.monotonic() value past which no further attempt is
    started, so a stalling endpoint cannot consume the whole run.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError(f"UniProt time budget exhausted after {attempt} attempts")
        try:
            with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as response:
                return response.read(), response.headers.get("Link", "")
        except _TRANSIENT_ERRORS as error:
            last_error = error
            if attempt < retries - 1:
                time.sleep(2.0**attempt)
    raise RuntimeError(f"UniProt request failed after {retries} attempts: {last_error}")


def _fetch(url: str, retries: int = 5, deadline: float | None = None) -> bytes:
    return _request(url, retries=retries, deadline=deadline)[0]


def _query_params(name: str, **extra: object) -> str:
    return urllib.parse.urlencode(
        {"query": QUERIES[name], "fields": FIELDS, "format": "tsv", **extra}
    )


def _fetch_paginated(name: str, deadline: float | None = None) -> bytes:
    """Walk the search endpoint page by page, following its cursor links.

    The stream endpoint builds the whole response server-side and times out on
    large classes whenever UniProt is under load. Paginating costs more requests
    but each one is small, so a stall costs one page instead of the entire class.

    A partial walk is never returned: a class cut short would silently shrink the
    dataset, which is worse than falling through to the snapshot.
    """
    url: str | None = f"{UNIPROT_SEARCH}?{_query_params(name, size=PAGE_SIZE)}"
    lines: list[str] = []
    while url:
        body, link = _request(url, deadline=deadline)
        page = body.decode("utf-8").splitlines()
        # Every page repeats the TSV header; keep it only from the first.
        lines.extend(page if not lines else page[1:])
        match = _NEXT_LINK.search(link)
        url = match.group(1) if match else None
    return ("\n".join(lines) + "\n").encode("utf-8")


def _snapshot_manifest() -> dict:
    manifest = SNAPSHOT_DIR / "MANIFEST.json"
    if not manifest.exists():
        raise RuntimeError(
            f"No snapshot at {SNAPSHOT_DIR}. Regenerate it with 'make snapshot', "
            f"or pass --no-snapshot-fallback to fail instead of falling back."
        )
    return json.loads(manifest.read_text(encoding="utf-8"))


def read_snapshot(name: str) -> bytes:
    """Read one class from the pinned snapshot, verifying its checksum."""
    manifest = _snapshot_manifest()
    if name not in manifest["classes"]:
        raise RuntimeError(
            f"Snapshot has no {name!r} class (it holds {sorted(manifest['classes'])}). "
            f"Rebuild the full set with 'make snapshot'."
        )
    entry = manifest["classes"][name]
    payload = gzip.decompress((SNAPSHOT_DIR / f"{name}.tsv.gz").read_bytes())
    digest = hashlib.sha256(payload).hexdigest()
    if digest != entry["sha256"]:
        raise RuntimeError(
            f"Snapshot for {name!r} is corrupt: expected sha256 {entry['sha256']}, got {digest}"
        )
    return payload


def fetch_class(
    name: str,
    destination: Path,
    *,
    allow_snapshot: bool = True,
    budget: float = LIVE_BUDGET,
    use_snapshot: bool = False,
) -> Path:
    """Download one sequence class as TSV, including transmembrane features.

    Tries the stream endpoint first because it is one request when UniProt is
    healthy, falls back to paginated search when it is not, and finally falls
    back to the pinned snapshot so an outage cannot stop a run. Whichever source
    is used gets recorded in PROVENANCE.

    ``use_snapshot`` skips UniProt and any cached download and reads the pinned
    snapshot directly, which is how the published results are reproduced.
    """
    if name not in QUERIES:
        raise KeyError(f"Unknown class {name!r}; expected one of {sorted(QUERIES)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if use_snapshot:
        destination.write_bytes(read_snapshot(name))
        PROVENANCE[name] = {
            "source": "snapshot",
            "snapshot_date": _snapshot_manifest()["date"],
            "reason": "requested with --snapshot",
        }
        return destination

    if destination.exists() and destination.stat().st_size > 0:
        PROVENANCE[name] = {"source": "cache", "path": str(destination)}
        return destination

    deadline = time.monotonic() + budget
    try:
        try:
            # Two attempts only: when the stream endpoint is unwell, pagination is
            # a better use of the remaining budget than retrying it three more times.
            payload = _fetch(
                f"{UNIPROT_STREAM}?{_query_params(name)}", retries=2, deadline=deadline
            )
        except RuntimeError as stream_error:
            print(f"  {name}: stream endpoint failed ({stream_error}); paginating instead")
            payload = _fetch_paginated(name, deadline=deadline)
        PROVENANCE[name] = {"source": "uniprot"}
    except RuntimeError as error:
        if not allow_snapshot:
            raise
        manifest = _snapshot_manifest()
        print(
            f"  WARNING: UniProt unreachable for {name!r} ({error}).\n"
            f"  Falling back to the pinned snapshot of {manifest['date']}."
        )
        payload = read_snapshot(name)
        PROVENANCE[name] = {
            "source": "snapshot",
            "snapshot_date": manifest["date"],
            "uniprot_error": str(error),
        }

    destination.write_bytes(payload)
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
