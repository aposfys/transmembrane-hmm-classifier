# Data sources and licences

The MIT licence covers **the code in this repository only**. The data it retrieves, and the
third-party tools it invokes, carry their own terms.

| Source | Used for | Licence |
| --- | --- | --- |
| [UniProt](https://www.uniprot.org/help/license) | All four sequence classes and their transmembrane annotation | CC BY 4.0 |
| [HMMER](https://github.com/EddyRivasLab/hmmer) | Profile HMM construction and search | BSD-3-Clause |
| [Clustal Omega](http://www.clustal.org/omega/) | Multiple sequence alignment | GPL-2.0-or-later |
| [CD-HIT](https://github.com/weizhongli/cdhit) | Redundancy reduction | GPL-2.0-or-later |
| [ESM-2](https://github.com/facebookresearch/esm) | Sequence embeddings (code and weights) | MIT |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | Classifier heads | BSD-3-Clause |

External tools are invoked as separate executables via `subprocess`, not linked into this
codebase, so their copyleft terms do not extend to the code here. Anyone redistributing a
bundle that *includes* those binaries takes on their obligations.

**What this repository ships.** Every class is fetched from UniProt at run time and cached
locally (`data/` is git-ignored). One exception: `src/tmclass/snapshot/` commits a gzipped
copy of all four classes so a run survives a UniProt outage (see
[the offline fallback](RUNNING.md#data-fetching-and-the-offline-fallback)). That snapshot is
UniProt data redistributed under CC BY 4.0, which its licence permits with attribution;
`MANIFEST.json` records the date, the exact query, and a checksum per class. The profile
HMMs and metrics under `results/` are derived works built from the same CC BY 4.0 sequences,
so the attribution above applies to them too. If you reuse either, cite UniProt as well.
