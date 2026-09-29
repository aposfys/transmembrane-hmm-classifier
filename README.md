# Single-Pass Type I Transmembrane Protein Classification
Profile HMMs versus ESM-2 embeddings on the same task, against decoys chosen to break the classifier.

[![Pipeline](https://github.com/aposfys/transmembrane-hmm-classifier/actions/workflows/pipeline.yml/badge.svg)](https://github.com/aposfys/transmembrane-hmm-classifier/actions/workflows/pipeline.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

1,789 reviewed single-pass type I proteins from UniProt (≤ 40% pairwise identity) against 358 single-pass **type II**, 172 **GPCRs** and 358 **globular** decoys. Classical profile HMMs (HMMER) are compared with ESM-2 650M embeddings plus a discriminative head, on an identical held-out test set.

<p align="center">
  <img src="results/model_comparison.png" width="880" alt="Profile HMM versus ESM-2 embeddings across five metrics">
</p>

### Results

All four models are scored on the same 358 positives and 888 decoys, at each model's own MCC-optimal threshold. Both HMMs score positives and decoys in one `hmmsearch` run, so every E-value shares one database size.

| Model | ROC AUC | 95% CI | Avg. precision | MCC | Sensitivity | Specificity | Precision |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Profile HMM, full-length | 0.765 | [0.733, 0.796] | 0.662 | 0.524 | 0.433 | 0.973 | 0.866 |
| Profile HMM, TM regions | 0.879 | [0.854, 0.903] | 0.752 | 0.593 | 0.654 | 0.913 | 0.752 |
| ESM-2 + logistic regression | 0.980 | [0.970, 0.990] | 0.952 | 0.891 | 0.947 | 0.957 | 0.899 |
| ESM-2 + MLP | 0.982 | [0.972, 0.992] | 0.955 | 0.875 | 0.891 | 0.973 | 0.930 |

AUC intervals are Hanley–McNeil and proportions are Wilson score intervals, computed from the saved counts by `python -m tmclass.intervals`. The thresholds behind MCC, sensitivity, specificity and precision are chosen on the test set, so those columns are optimistic for every model.

**Both ESM heads beat both HMMs, and the TM-region HMM beats the full-length one.** All of these AUC intervals separate. The two ESM heads do not separate from each other (0.980 [0.970, 0.990] against 0.982 [0.972, 0.992]), so neither is called the winner. DeepTMHMM, TMbed and Phobius are not run here, so this is a comparison of two model families rather than a state-of-the-art benchmark.

**A linear probe on frozen embeddings is enough.** Logistic regression on unmodified ESM-2 vectors reaches MCC 0.891, so the separation is already present in the representation rather than built by the classifier.

**For the two ESM heads, type II decoys are the hard class.** Their type II false-positive rates are 9.2% [6.6, 12.7] and 6.4% [4.3, 9.5], against 0/172 [0, 2.2] on GPCRs. Type II proteins share the single-helix architecture of type I and differ only in which terminus faces the cytoplasm. The TM-region HMM also errs most on type II (17.3% [13.8, 21.6]), but not separably more than on GPCRs (8.7% [5.4, 13.9]). For the full-length HMM the type II rate (2.2% [1.1, 4.3]) separates from neither other class.

**The conventional HMM threshold is badly miscalibrated.** At E ≤ 0.05 the TM-region HMM misses 70% of true type I proteins (sensitivity 0.299, MCC 0.426). The MCC-optimal cutoff is E ≤ 1.2 (sensitivity 0.654, MCC 0.593). The same point has been made for supervised classification with HMMER by HMMERCTTER (*PLOS One* 2018). What this repository adds is the measurement on a curated set with hard decoys. See [docs/METHOD.md](docs/METHOD.md#prior-work) for the prior work and for what changed when the E-value scaling was corrected.

### Quick start

```
conda env create -f environment.yml   # HMMER, Clustal Omega, CD-HIT, Python deps
conda activate tmclass
pip install -e ".[dev,embeddings]"

make data       # fetch and cluster all four classes from live UniProt
make analysis   # HMMs, ESM-2 benchmark, cross-validation, figures
make test
```

To rebuild the published dataset from the pinned UniProt snapshot, run `python -m tmclass.cli --snapshot`. A full run takes roughly 3 hours on an M4, almost all of it ESM-2 650M embedding. Adding `--reuse-benchmark results/findings.json` keeps the committed ESM-2 results and re-runs the HMM side in about 7 minutes.

### More

- [Method, results detail, prior work and pitfalls](docs/METHOD.md)
- [Uncertainty in the reported numbers](docs/METHOD.md#uncertainty-in-the-reported-numbers)
- [CLI options, runtime and the UniProt offline fallback](docs/RUNNING.md)
- [Data sources and licences](docs/DATA.md)

---

Apostolos Fysekidis · [MIT Licence](LICENSE)
