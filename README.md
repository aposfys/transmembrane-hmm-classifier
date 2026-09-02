# Single-Pass Type I Transmembrane Protein Classification
Profile HMMs versus ESM-2 embeddings on the same task, against decoys chosen to break the classifier.

[![Pipeline](https://github.com/aposfys/transmembrane-hmm-classifier/actions/workflows/pipeline.yml/badge.svg)](https://github.com/aposfys/transmembrane-hmm-classifier/actions/workflows/pipeline.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

1,789 reviewed single-pass type I proteins from UniProt (≤ 40% pairwise identity) against 358 single-pass **type II**, 172 **GPCRs** and 358 **globular** decoys. Classical profile HMMs (HMMER) are compared with ESM-2 650M embeddings plus a discriminative head, on an identical held-out test set.

<p align="center">
  <img src="results/model_comparison.png" width="880" alt="Profile HMM versus ESM-2 embeddings across five metrics">
</p>

### Results

All four models are scored on the same 358 positives and 888 decoys, at each model's own MCC-optimal threshold.

| Model | ROC AUC | 95% CI | Avg. precision | MCC | Sensitivity | Specificity | Precision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Profile HMM, full-length | 0.834 | [0.807, 0.862] | 0.767 | 0.610 | 0.570 | 0.961 | 0.854 |
| Profile HMM, TM regions | 0.908 | [0.886, 0.929] | 0.814 | 0.658 | 0.724 | 0.919 | 0.782 |
| ESM-2 + logistic regression | 0.980 | [0.970, 0.990] | 0.952 | 0.891 | 0.947 | 0.957 | 0.899 |
| ESM-2 + MLP | 0.982 | [0.972, 0.992] | 0.955 | 0.875 | 0.891 | 0.973 | 0.930 |

AUC intervals are Hanley–McNeil; proportions elsewhere are Wilson score intervals. Both are computed from the saved counts by `python -m tmclass.intervals`, so they need no refitting.

**The language model advantage is the only model difference the data establishes.** Both ESM heads separate cleanly from both HMMs. **The two ESM heads do not separate from each other** — 0.980 [0.970, 0.990] against 0.982 [0.972, 0.992] — so no row here is bolded as the winner, and reading a preference between logistic regression and the MLP off a fourth decimal place would be reading noise. Three things are more interesting than the headline.

**A linear probe on frozen embeddings is enough.** Logistic regression on unmodified ESM-2 vectors reaches MCC 0.891 and the MLP reaches 0.875, so the separation is already present in the representation rather than constructed by the classifier. ESM-2 was never trained on membrane topology — it learned it as a by-product of masked-language modelling.

**The errors are biologically structured — in three models out of four.** For both ESM heads and the TM-region HMM, type II decoys are the hard class: false-positive rate 15.9% [12.5, 20.1] for the TM-region HMM, 9.2% [6.6, 12.7] for logreg and 6.4% [4.3, 9.5] for the MLP, against 8.7% [5.4, 13.9], 0/172 [0, 2.2] and 0/172 [0, 2.2] on GPCRs. Type II proteins share the single-helix architecture of type I and differ only in which terminus faces the cytoplasm — carried by flanking charge asymmetry, not by the helix.

**The full-length HMM inverts that ordering**, and it is reported here rather than omitted: it confuses GPCRs at 9.9% [6.3, 15.3] against 3.1% [1.7, 5.4] on type II — significantly *more* GPCR errors than type II errors, the opposite of every other model. So the pattern is a property of three of the four classifiers, not a general fact about the task, and a test pins the inversion so it cannot quietly disappear. The between-model comparisons are weaker still: on type II decoys only the TM-region HMM and the MLP separate, while logreg separates from neither.

**The conventional HMM threshold is badly miscalibrated.** At E ≤ 0.05 the TM-region HMM misses 57% of true type I proteins (MCC 0.536); sweeping to the MCC-optimal E ≤ 0.74 recovers sensitivity 0.724 (MCC 0.658). E ≤ 0.05 is a convention inherited from homology search, not an operating point for a topology classifier.

### Quick start

```
conda env create -f environment.yml   # HMMER, Clustal Omega, CD-HIT, Python deps
conda activate tmclass
pip install -e ".[dev]"

make data       # fetch and cluster all four classes (~40 min, cached afterwards)
make analysis   # HMMs, ESM-2 benchmark, cross-validation, figures
make test
```

A full `make analysis` takes roughly 3 hours on an M4, dominated by ESM-2 650M embedding. `--esm-model 35M` cuts that to about 25 minutes with a modest accuracy cost, and `--max-sequences-per-class N` gives a one-minute smoke run.

### Prior work, and how much of the threshold result is new

The threshold finding above needs framing against two papers that reach it first, from
different directions:

- **HMMERCTTER** (*PLOS One* 2018) — a tool built specifically because HMMER's default
  E-value cut-offs are not reliable operating points for *supervised classification* of
  superfamily sequences, as opposed to homology search. The conceptual claim here is theirs.
- **Challenges in homology search: HMMER3 and convergent evolution of coiled-coil regions**
  (*Nucleic Acids Research* 2013) — shows HMMER3 E-value estimates are less accurate for
  families with periodic compositional bias, and names **multi-span helical transmembrane
  domains** among them. That is this benchmark's GPCR decoy class, and it supplies the
  mechanism behind the errors reported above.

So "E ≤ 0.05 is a convention inherited from homology search, not an operating point for a
topology classifier" is a restatement of published work. What is new is the **measurement**:
57% of true single-pass type I proteins missed at that threshold, on a curated set with
type II, GPCR and globular decoys chosen to be hard, with the MCC-optimal operating point
located by sweep and the whole thing under confidence intervals.

**The ESM-2 comparison is not benchmarked against the state of the art.** DeepTMHMM, TMbed,
Phobius and TOPCONS are the tools a reader will expect, and none is run here. That a language
model beats a profile HMM at topology classification has been the field's position since
around 2022; this repository demonstrates it cleanly rather than establishing it.

### More

- [Method, experimental design and the pitfalls this pipeline avoids](docs/METHOD.md)
- [Uncertainty in the reported numbers](docs/METHOD.md#uncertainty-in-the-reported-numbers) — what the confidence intervals do and do not support
- [CLI options, runtime and the UniProt offline fallback](docs/RUNNING.md)
- [Data sources and licences](docs/DATA.md)

---

Apostolos Fysekidis · [MIT Licence](LICENSE)
