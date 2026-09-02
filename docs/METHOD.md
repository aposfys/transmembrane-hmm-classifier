# Method

1. **Fetch** four sequence classes from the UniProt REST API by subcellular-location and
   family term, with `ft_transmem` coordinates in the same response — no external
   topology-prediction service.
2. **Filter** positives to entries with exactly one annotated TM segment.
3. **Reduce redundancy** with CD-HIT at 40% identity, so no homologue spans the train/test
   boundary.
4. **Split** 80/20 with a fixed seed; cap each decoy class at half its pool.
5. **Classical path** — extract TM segments ± 10 residues, align with Clustal Omega, build
   with `hmmbuild`, score with `hmmsearch --max` (heuristic filters off, so weak hits still
   get an E-value and the ROC curve is not truncated).
6. **Modern path** — mean-pool ESM-2's final hidden layer over each sequence's residues,
   then fit a head. Sequences longer than 1,022 residues are split into overlapping windows
   and pooled across them, because truncating would discard the membrane segment of any
   C-terminally anchored protein.
7. **Evaluate** by sweeping every threshold that changes the confusion matrix, reporting
   ROC AUC, average precision and the full metric suite.

## Full results detail

### Errors by decoy class

| Decoy class | HMM (TM regions) | ESM-2 + logreg | ESM-2 + MLP | |
| --- | ---: | ---: | ---: | --- |
| Globular (no TM segment) | 0.0% | 1.4% | 0.3% | trivially separable |
| GPCR (polytopic) | 8.7% | **0.0%** | **0.0%** | solved by the language model |
| Single-pass **type II** | 15.9% | 9.2% | 6.4% | the irreducible case |

The language model eliminates GPCR confusion entirely, which the HMM never manages:
distinguishing one membrane helix from seven requires context that a profile of the helix
alone does not contain.

### The HMM threshold sweep

| Profile HMM, TM regions | Sensitivity | Specificity | Precision | MCC |
| --- | ---: | ---: | ---: | ---: |
| Conventional, E ≤ 0.05 | 0.425 | 0.981 | 0.899 | 0.536 |
| MCC-optimal, E ≤ 0.74 | **0.724** | 0.919 | 0.782 | **0.658** |

### Generalisation

Trained on only 600 positives and 600 decoys, out-of-fold performance on the training pool
(AUC 0.979) matches held-out test performance (AUC 0.980), so nothing is being memorised.

## Why the profile HMM is limited here

Single-pass type I proteins are **not a homologous family** — they are a topological class
whose members share no common ancestor. A profile HMM models a family, so aligning whole
sequences produces a meaningless alignment and a model that matches generic composition
(AUC 0.834, errors spread evenly across all decoy classes including proteins with no
membrane segment at all).

Restricting the model to the membrane segment plus 10 residues of flank helps, because that
*is* a real motif: ~20 hydrophobic residues bounded by charge asymmetry. But it still models
a single conserved pattern, whereas an embedding encodes the whole sequence context. That
gap is what the 0.908 → 0.982 AUC improvement measures.

## A note on experimental design

The two model families are trained differently, and the asymmetry is deliberate:

- A profile HMM is **generative** — built from an alignment of positives only, it never sees
  a negative.
- A linear or MLP head is **discriminative** — it requires labelled negatives.

Head-training negatives are drawn strictly from the pool the shared test set did not
consume, so no test sequence is ever trained on. An earlier version of this benchmark let
the test set exhaust the GPCR pool, leaving the head with no GPCR training examples; it then
scored **100% false positives on GPCRs**, which looked like a catastrophic failure but was
purely an artefact of the split. Each decoy class is now capped at half its non-redundant
pool so every class contributes to both sides. The lesson is in
[`cli.py`](../src/tmclass/cli.py) as a comment, because it is the kind of mistake that
produces a plausible-looking number.

## Pitfalls this pipeline is built to avoid

Each of these silently produces numbers that look reasonable.

- **Sensitivity and specificity hide precision under class imbalance.** With 9 positives
  against 121 negatives, specificity 0.868 leaves 16 false positives — precision 0.30,
  meaning two of every three positive calls are wrong, from a pair of metrics that both look
  respectable. MCC and precision are primary here; a test pins that arithmetic.
- **A small test set cannot measure anything.** With 9 positives, one sequence moves
  sensitivity by 11 points. This uses 358 positives and 888 decoys, plus cross-validation
  with reported standard deviations.
- **E ≤ 0.05 is a convention, not an operating point.** Adopting it costs more than half the
  achievable sensitivity here.
- **CD-HIT truncates sequence names.** `cd-hit` defaults to `-d 20`, so
  `sp|Q8WXI7|MUC16_HUMAN` becomes `sp|Q8WXI7|MUC16_HUM` in the `.clstr` file. Any downstream
  `if identifier in records` filter keyed on full headers then drops most of the dataset
  without raising — in a 676-sequence clustering, **569 representatives (84%) vanish**. Here
  `-d 0` preserves headers, `subset_fasta` raises on an unmatched identifier, and `Split`
  validates that the partitions sum to their input.
- **Sequences missing from `hmmsearch` output are negatives, not missing data.** Dropping
  them shrinks the denominator and inflates every metric.
- **Padding dominates transformer batches.** Mixing a 40-residue window with a 1,022-residue
  one wastes most of the forward pass; length-sorted batching is worth more than half the
  embedding runtime.
- **A decoy class absent from training measures generalisation, not accuracy** — see the
  design note above.

## Repository layout

```
src/tmclass/
  data.py       UniProt queries, TSV parsing, TM-coordinate extraction
  snapshot/     Pinned UniProt copy (gzipped TSV + MANIFEST.json) for outages
  refresh_snapshot.py  Regenerates that snapshot; see `make snapshot`
  regions.py    TM region slicing with flanks; TOPCONS parser
  pipeline.py   CD-HIT, splitting, Clustal Omega, hmmbuild, hmmsearch, tblout parsing
  plm.py        ESM-2 embedding with windowing, length-bucketed batching, caching
  heads.py      Logistic-regression and MLP heads; embedding separability
  benchmark.py  Trains the heads and scores them on the shared test set
  evaluate.py   Confusion matrix, metric suite, threshold sweep, ROC/PR areas
  plots.py      Curves, confusion matrices, model and error-class comparisons
  cli.py        Staged pipeline with cross-validation
tests/          pytest suite (42 tests)
results/        Models, sweeps, figures, findings.json
```

## Output files

| File | Contents |
| --- | --- |
| `results/tm_region.hmm` | The profile HMM, usable directly with `hmmsearch` |
| `results/*_sweep.csv` | Every threshold with its full metric row |
| `results/findings.json` | All four models, per-class errors, cross-validation |
| `results/model_comparison.png` | The headline figure |
| `results/error_by_decoy_class.png` | Where each model fails |
| `data/embeddings/*.npz` | Cached ESM-2 embeddings, keyed by model and sequence set |

## References

1. Lin, Z. *et al.* (2023). Evolutionary-scale prediction of atomic-level protein structure
   with a language model. *Science* **379**, 1123–1130. (ESM-2)
2. Rives, A. *et al.* (2021). Biological structure and function emerge from scaling
   unsupervised learning to 250 million protein sequences. *PNAS* **118**, e2016239118.
3. Eddy, S. R. (2011). Accelerated profile HMM searches. *PLoS Computational Biology* **7**,
   e1002195.
4. Eddy, S. R. (1998). Profile hidden Markov models. *Bioinformatics* **14**, 755–763.
5. von Heijne, G. (1992). Membrane protein structure prediction: hydrophobicity analysis and
   the positive-inside rule. *Journal of Molecular Biology* **225**, 487–494.
6. Fu, L., Niu, B., Zhu, Z., Wu, S. & Li, W. (2012). CD-HIT: accelerated for clustering the
   next-generation sequencing data. *Bioinformatics* **28**, 3150–3152.
7. Sievers, F. *et al.* (2011). Fast, scalable generation of high-quality protein multiple
   sequence alignments using Clustal Omega. *Molecular Systems Biology* **7**, 539.
8. Chicco, D. & Jurman, G. (2020). The advantages of the Matthews correlation coefficient
   (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*
   **21**, 6.
9. UniProt Consortium (2023). UniProt: the universal protein knowledgebase in 2023. *Nucleic
   Acids Research* **51**, D523–D531.

## Uncertainty in the reported numbers

Every headline figure is one estimate from one held-out split of 358 positives and 888
decoys. Several of the most quoted ones — the per-decoy-class false-positive rates — rest on
counts small enough that the point estimate alone misleads, and a rate reported as "0.0%"
from 0 of 172 GPCR decoys is the clearest case.

Two interval methods are used, both computable from counts the pipeline already saves, so
`python -m tmclass.intervals results/findings.json` regenerates the whole report without
refitting a model or recomputing an embedding:

- **Wilson score intervals** for proportions. Chosen over the normal approximation because
  it stays inside [0, 1] and keeps near-nominal coverage at the extremes, which is exactly
  where the interesting counts sit here (Brown, Cai & DasGupta, *Statistical Science* 2001).
  A normal-approximation interval on 0/172 is [0, 0], which would assert the rate is zero.
- **Hanley–McNeil standard errors** for ROC AUC, which need only the AUC and the two class
  sizes (Hanley & McNeil, *Radiology* 1982).

### What the intervals change

| Claim | Verdict |
| --- | --- |
| ESM-2 beats both profile HMMs | **Holds.** All four cross-model AUC intervals separate. |
| ESM-2 + MLP is the best model | **Not established.** 0.982 [0.972, 0.992] against logreg's 0.980 [0.970, 0.990]. |
| Type II decoys are the hard class | **Holds for three models of four.** The full-length HMM confuses GPCRs significantly more. |
| Type II error rate falls across model generations | **Not established.** Only the TM-region HMM and the MLP separate on that class. |

### The caveat that belongs with these intervals

All four models are scored on the *same* sequences, so their errors are correlated. Asking
whether two Hanley–McNeil intervals overlap is therefore a **conservative** test — it will
miss differences a paired test would find, which is the likely situation for the two ESM
heads. A paired DeLong test on per-sequence scores is the sharper instrument and is the
natural next step; it needs the raw scores rather than the summary counts, which is the one
thing `findings.json` does not currently persist.
