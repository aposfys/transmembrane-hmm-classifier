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

| Model | ROC AUC | Avg. precision | MCC | Sensitivity | Specificity | Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Profile HMM, full-length | 0.834 | 0.767 | 0.610 | 0.570 | 0.961 | 0.854 |
| Profile HMM, TM regions | 0.908 | 0.814 | 0.658 | 0.724 | 0.919 | 0.782 |
| ESM-2 + logistic regression | 0.980 | 0.952 | **0.891** | **0.947** | 0.957 | 0.899 |
| ESM-2 + MLP | **0.982** | **0.955** | 0.875 | 0.891 | **0.973** | **0.930** |

The language model wins decisively, but three things are more interesting than the headline.

**A linear probe on frozen embeddings is enough.** Logistic regression on unmodified ESM-2 vectors reaches MCC 0.891 and the MLP reaches 0.875, so the separation is already present in the representation rather than constructed by the classifier. ESM-2 was never trained on membrane topology — it learned it as a by-product of masked-language modelling.

**The errors are biologically structured, in every model.** False-positive rate on type II decoys is 15.9% (HMM), 9.2% (logreg) and 6.4% (MLP); on GPCRs it is 8.7%, 0.0% and 0.0%. Type II proteins share the single-helix architecture of type I and differ only in which terminus faces the cytoplasm — carried by flanking charge asymmetry, not by the helix. That the error ordering survives a change of model generation is evidence it reflects the biology.

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

### More

- [Method, experimental design and the pitfalls this pipeline avoids](docs/METHOD.md)
- [CLI options, runtime and the UniProt offline fallback](docs/RUNNING.md)
- [Data sources and licences](docs/DATA.md)

---

Apostolos Fysekidis · [MIT Licence](LICENSE)
