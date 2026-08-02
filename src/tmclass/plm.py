"""Protein language model embeddings as an alternative to the profile HMM.

A profile HMM is a generative model of one aligned family. A protein language
model instead produces a fixed-length vector per sequence from weights trained
across all of UniRef, which a small discriminative head can then classify. This
module provides the embedding half; :mod:`tmclass.heads` provides the head.

Embeddings are cached to ``.npz`` keyed by model and sequence set, because they
are the expensive part and are reused across every head and every fold.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ESM-2 checkpoints, smallest first. The 650M model is the usual quality/cost
# sweet spot; the smaller ones are useful for iterating on the pipeline.
ESM2_MODELS = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
}
DEFAULT_MODEL = "650M"

# ESM-2 was trained with 1024 position embeddings, two of which are consumed by
# the BOS and EOS tokens.
MAX_RESIDUES = 1022
# Overlap between windows of a long sequence, so a transmembrane helix straddling
# a window boundary is still seen whole in at least one window.
WINDOW_OVERLAP = 128


@dataclass(frozen=True)
class EmbeddingSet:
    """Mean-pooled per-sequence embeddings, aligned to ``identifiers``."""

    identifiers: tuple[str, ...]
    vectors: np.ndarray  # shape (n_sequences, embedding_dim)
    model: str

    def __post_init__(self) -> None:
        if len(self.identifiers) != len(self.vectors):
            raise ValueError(
                f"{len(self.identifiers)} identifiers but {len(self.vectors)} vectors"
            )

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1])

    def subset(self, wanted: Sequence[str]) -> np.ndarray:
        """Rows for the given identifiers, in the order requested."""
        index = {name: row for row, name in enumerate(self.identifiers)}
        missing = [name for name in wanted if name not in index]
        if missing:
            raise KeyError(f"{len(missing)} identifiers not embedded, e.g. {missing[:3]}")
        return self.vectors[[index[name] for name in wanted]]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            identifiers=np.array(self.identifiers, dtype=object),
            vectors=self.vectors,
            model=np.array(self.model),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> EmbeddingSet:
        payload = np.load(path, allow_pickle=True)
        return cls(
            identifiers=tuple(payload["identifiers"].tolist()),
            vectors=payload["vectors"],
            model=str(payload["model"]),
        )


def windows(
    sequence: str, size: int = MAX_RESIDUES, overlap: int = WINDOW_OVERLAP
) -> Iterator[str]:
    """Split a sequence into overlapping windows the model can accept.

    Truncating instead would silently discard the membrane segment of any
    C-terminally anchored protein, which is precisely the signal being modelled.
    """
    if len(sequence) <= size:
        yield sequence
        return

    step = size - overlap
    for start in range(0, len(sequence), step):
        window = sequence[start : start + size]
        yield window
        if start + size >= len(sequence):
            return


def _select_device(requested: str | None) -> str:
    import torch

    if requested and requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def cache_path(directory: Path, name: str, model: str, identifiers: Iterable[str]) -> Path:
    """A cache filename that changes whenever the model or sequence set changes."""
    digest = hashlib.sha256("\n".join(sorted(identifiers)).encode()).hexdigest()[:12]
    return directory / f"emb_{name}_{model}_{digest}.npz"


def embed(
    records: Sequence[tuple[str, str]],
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    batch_size: int = 8,
    progress_every: int = 200,
    half_precision: bool = True,
) -> EmbeddingSet:
    """Mean-pool the final hidden layer over the residues of each sequence.

    Args:
        records: ``(identifier, sequence)`` pairs.
        model: A key of :data:`ESM2_MODELS`, or a full HuggingFace model name.
        device: ``"mps"``, ``"cuda"``, ``"cpu"``, or ``None`` to pick the best.
        batch_size: Windows per forward pass.
        half_precision: Run the forward pass in float16 on an accelerator.
            Embeddings are pooled and returned in float32 regardless; the
            precision loss is far below the scale of the differences a
            downstream classifier uses, and it buys roughly 1.4x throughput.

    Returns:
        One vector per input record, in the input order.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    checkpoint = ESM2_MODELS.get(model, model)
    resolved = _select_device(device)

    dtype = torch.float16 if half_precision and resolved != "cpu" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    network = AutoModel.from_pretrained(checkpoint, dtype=dtype).to(resolved).eval()

    # Flatten to windows so long sequences are not truncated, remembering which
    # sequence each window came from.
    flat: list[str] = []
    owners: list[int] = []
    for position, (_, sequence) in enumerate(records):
        for window in windows(sequence):
            flat.append(window)
            owners.append(position)

    # Batch windows of similar length together. Transformer cost scales with the
    # longest member of a batch, so mixing a 40-residue window with a 1022-residue
    # one spends most of the forward pass on padding.
    order = sorted(range(len(flat)), key=lambda index: len(flat[index]))

    dimension = int(network.config.hidden_size)
    totals = np.zeros((len(records), dimension), dtype=np.float32)
    counts = np.zeros(len(records), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch = [flat[index] for index in indices]
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_RESIDUES + 2,
            ).to(resolved)

            hidden = network(**encoded).last_hidden_state

            # Mask out padding, BOS and EOS so the mean covers residues only.
            mask = encoded["attention_mask"].clone()
            mask[:, 0] = 0
            lengths = encoded["attention_mask"].sum(dim=1) - 1
            mask[torch.arange(mask.size(0), device=mask.device), lengths] = 0

            weights = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
            pooled = pooled.float().cpu().numpy()

            for index, vector in zip(indices, pooled, strict=True):
                owner = owners[index]
                totals[owner] += vector
                counts[owner] += 1

            if progress_every and (start // batch_size) % progress_every == 0:
                print(
                    f"    embedded {min(start + batch_size, len(order))}/{len(order)} windows",
                    flush=True,
                )

    vectors = totals / counts[:, None].clip(min=1)
    return EmbeddingSet(
        identifiers=tuple(identifier for identifier, _ in records),
        vectors=vectors,
        model=model,
    )


def embed_cached(
    records: Sequence[tuple[str, str]],
    directory: Path,
    name: str,
    model: str = DEFAULT_MODEL,
    device: str | None = None,
    batch_size: int = 8,
    half_precision: bool = True,
) -> EmbeddingSet:
    """Embed, reusing a cached ``.npz`` when the model and sequence set match."""
    path = cache_path(directory, name, model, (identifier for identifier, _ in records))
    if path.exists():
        cached = EmbeddingSet.load(path)
        if set(cached.identifiers) == {identifier for identifier, _ in records}:
            print(f"    reusing cached embeddings: {path.name}")
            return cached

    embeddings = embed(
        records,
        model=model,
        device=device,
        batch_size=batch_size,
        half_precision=half_precision,
    )
    embeddings.save(path)
    return embeddings
