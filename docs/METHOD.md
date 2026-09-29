# Method

1. **Fetch** four sequence classes from the UniProt REST API by subcellular-location and
   family term, with `ft_transmem` coordinates in the same response, so no external
   topology-prediction service is needed.
2. **Filter** positives to entries with exactly one annotated TM segment.
3. **Reduce redundancy** with CD-HIT at 40% identity, so no homologue spans the train/test
   boundary.
4. **Split** 80/20 with a fixed seed; cap each decoy class at half its pool.
5. **Classical path.** Extract TM segments ± 10 residues, align with Clustal Omega, build
   with `hmmbuild`, and score with `hmmsearch --max` (heuristic filters off, so weak hits
   still get an E-value and the ROC curve is not truncated). Positives and decoys go
   through one search, so every E-value is scaled by the same database size.
6. **Modern path.** Mean-pool ESM-2's final hidden layer over each sequence's residues,
   then fit a head. Sequences longer than 1,022 residues are split into overlapping windows
   and pooled across them, because truncating would discard the membrane segment of any
   C-terminally anchored protein.
7. **Evaluate** by sweeping every threshold that changes the confusion matrix, reporting
   ROC AUC, average precision and the full metric suite.

## Full results detail

### Errors by decoy class

False-positive rates at each model's MCC-optimal threshold, with 95% Wilson intervals.

| Decoy class | HMM (full-length) | HMM (TM regions) | ESM-2 + logreg | ESM-2 + MLP |
| --- | ---: | ---: | ---: | ---: |
| Globular (no TM segment) | 1.7% [0.8, 3.6] | 0/358 [0, 1.1] | 1.4% [0.6, 3.2] | 0.3% [0.05, 1.6] |
| GPCR (polytopic) | 5.8% [3.2, 10.4] | 8.7% [5.4, 13.9] | 0/172 [0, 2.2] | 0/172 [0, 2.2] |
| Single-pass **type II** | 2.2% [1.1, 4.3] | 17.3% [13.8, 21.6] | 9.2% [6.6, 12.7] | 6.4% [4.3, 9.5] |

Neither ESM head made a GPCR error in 172 decoys, while the TM-region HMM made 15. That
fits the idea that telling one membrane helix from seven needs context a profile of the
helix alone does not contain. 0/172 still allows a true rate of up to 2.2%. Type II is the
hardest class for both ESM heads, and on it both heads make significantly fewer errors than
the TM-region HMM. The full-length HMM has the lowest type II rate only because it calls
far fewer sequences positive at all (sensitivity 0.433).

### The HMM threshold sweep

| Profile HMM, TM regions | Sensitivity | Specificity | Precision | MCC |
| --- | ---: | ---: | ---: | ---: |
| Conventional, E ≤ 0.05 | 0.299 | 0.982 | 0.870 | 0.426 |
| MCC-optimal, E ≤ 1.2 | **0.654** | 0.913 | 0.752 | **0.593** |

The 5-fold cross-validation of the TM-region model scores each fold at E ≤ 0.05 and gives
mean sensitivity 0.140 ± 0.044 and MCC 0.206 ± 0.052. Each fold searches about 30 held-out
positives together with the 888 decoys, so its E-values sit on a slightly smaller scale than
the test set's, and the fold models are built from 120 rather than 150 sequences.

### Correction to the E-value scaling (September 2026)

The first published version searched the 358 test positives and the 888 decoys as two
separate files with no `-Z`. hmmsearch multiplies each P-value by the number of sequences in
the file, so a positive and a decoy with the same bit score got E-values 2.5x apart, in the
positive's favour. The cross-validation folds were worse, with about 30 positives in one
file against 888 decoys in the other. Scoring both classes in one search changed the HMM
results as follows. The ESM-2 results were unaffected and are carried over unchanged.

| | Before | After |
| --- | ---: | ---: |
| Full-length HMM, ROC AUC | 0.834 | 0.765 |
| TM-region HMM, ROC AUC | 0.908 | 0.879 |
| TM-region HMM, missed at E ≤ 0.05 | 57.5% | 70.1% |
| TM-region HMM, MCC-optimal cutoff | E ≤ 0.74, MCC 0.658 | E ≤ 1.2, MCC 0.593 |
| TM-region HMM, CV mean MCC at E ≤ 0.05 | 0.619 | 0.206 |
| Full-length HMM, GPCR vs type II false positives | 9.9% vs 3.1%, separated | 5.8% vs 2.2%, not separated |

The earlier claim that the full-length HMM confuses GPCRs significantly more than type II
proteins came from the scaling alone and is withdrawn. The regenerated dataset is identical
(built from the pinned snapshot of 2026-08-21), and so are the two profile HMMs.

### Generalisation

Trained on only 600 positives and 600 decoys, the logistic-regression head scores AUC 0.979
out of fold on its training pool and 0.980 on the held-out test set, so it is not memorising
its training data.

## Prior work

The threshold finding needs framing against two papers that reach it first, from different
directions.

- **HMMERCTTER** (*PLOS One* 2018, doi:10.1371/journal.pone.0193757) is a tool built because
  HMMER's default E-value cut-offs are not reliable operating points for supervised
  classification of superfamily sequences, as opposed to homology search. The conceptual
  claim is theirs.
- **Challenges in homology search: HMMER3 and convergent evolution of coiled-coil regions**
  (*Nucleic Acids Research* 2013, doi:10.1093/nar/gkt263) shows that HMMER3 E-value
  estimates are less accurate for families with periodic compositional bias, and names
  multi-span helical transmembrane domains among them. That is this benchmark's GPCR decoy
  class.

So the statement that E ≤ 0.05 is a homology-search convention rather than an operating
point for a topology classifier restates published work. What is new here is the
measurement. 70% of true single-pass type I proteins are missed at that threshold, on a
curated set with type II, GPCR and globular decoys, with the MCC-optimal cutoff located by
sweep and the numbers under confidence intervals.

The ESM-2 comparison is not benchmarked against the state of the art. DeepTMHMM, TMbed,
Phobius and TOPCONS are the tools a reader will expect, and none is run here. That a language
model beats a profile HMM at topology classification has been the field's position for some
years, and this repository demonstrates it on one dataset rather than establishing it.

## Why the profile HMM is limited here

Single-pass type I proteins are **not a homologous family**. They are a topological class
whose members share no common ancestor. A profile HMM models a family, so aligning whole
sequences produces a meaningless alignment and a model that matches generic composition
(AUC 0.765, with false positives in all three decoy classes, including 6 of 358 globular
proteins with no membrane segment at all).

Restricting the model to the membrane segment plus 10 residues of flank helps, because that
*is* a real motif: ~20 hydrophobic residues bounded by charge asymmetry. But it still models
a single conserved pattern, whereas an embedding encodes the whole sequence context. That
gap is what the 0.879 → 0.982 AUC improvement measures.

## A note on experimental design

The two model families are trained differently, and the asymmetry is deliberate:

- A profile HMM is **generative**. It is built from an alignment of positives only, it never sees
  a negative.
- A linear or MLP head is **discriminative**. It requires labelled negatives.

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
  against 121 negatives, specificity 0.868 leaves 16 false positives, so precision is 0.30,
  meaning two of every three positive calls are wrong, from a pair of metrics that both look
  respectable. MCC and precision are primary here, and a test pins that arithmetic.
- **A small test set cannot measure anything.** With 9 positives, one sequence moves
  sensitivity by 11 points. This uses 358 positives and 888 decoys, plus cross-validation
  with reported standard deviations.
- **E ≤ 0.05 is a convention, not an operating point.** Adopting it costs more than half the
  achievable sensitivity here.
- **E-values from separate searches are not comparable.** hmmsearch scales each E-value by
  the size of the target file, so positives and decoys searched as two files sit on two
  scales. Here they are searched together, and a test pins that.
- **CD-HIT truncates sequence names.** `cd-hit` defaults to `-d 20`, so
  `sp|Q8WXI7|MUC16_HUMAN` becomes `sp|Q8WXI7|MUC16_HUM` in the `.clstr` file. Any downstream
  `if identifier in records` filter keyed on full headers then drops most of the dataset
  without raising. In a 676-sequence clustering, **569 representatives (84%) vanish**. Here
  `-d 0` preserves headers, `subset_fasta` raises on an unmatched identifier, and `Split`
  validates that the partitions sum to their input.
- **Sequences missing from `hmmsearch` output are negatives, not missing data.** Dropping
  them shrinks the denominator and inflates every metric.
- **Padding dominates transformer batches.** Mixing a 40-residue window with a 1,022-residue
  one wastes most of the forward pass; length-sorted batching is worth more than half the
  embedding runtime.
- **A decoy class absent from training measures generalisation, not accuracy.** See the
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
tests/          pytest suite (58 tests)
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
decoys. Several of the most quoted ones, the per-decoy-class false-positive rates, rest on
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
| The TM-region HMM beats the full-length HMM | **Holds.** [0.854, 0.903] against [0.733, 0.796]. |
| ESM-2 + MLP is the best model | **Not established.** 0.982 [0.972, 0.992] against logreg's 0.980 [0.970, 0.990]. |
| Type II decoys are the hard class | **Holds for the two ESM heads.** The TM-region HMM does not separate type II from GPCRs, and the full-length HMM does not separate type II from either class. |
| Both ESM heads make fewer type II errors than the TM-region HMM | **Holds.** The two heads do not separate from each other. |

### The caveat that belongs with these intervals

All four models are scored on the *same* sequences, so their errors are correlated. Asking
whether two Hanley–McNeil intervals overlap is therefore a **conservative** test. It will
miss differences a paired test would find, which is the likely situation for the two ESM
heads. A paired DeLong test on per-sequence scores is the sharper instrument and is the
natural next step. It needs the raw scores rather than the summary counts, and
`findings.json` does not store those yet. The per-model thresholds are also chosen on the
test set, which flatters MCC, sensitivity, specificity and precision for every model.
