.PHONY: install data analysis quick cv snapshot test clean clean-data all

PYTHON ?= python3

all: analysis

## Install the Python package. External tools (HMMER, Clustal Omega, CD-HIT)
## come from environment.yml:
##   conda env create -f environment.yml && conda activate tmclass
install:
	$(PYTHON) -m pip install -e ".[dev,embeddings]"

## Fetch all four sequence classes from UniProt and cluster them at 40% identity.
## CD-HIT takes about 2 minutes on the type I set, and its output is cached
## against a checksum of the input, so later runs skip it.
data:
	$(PYTHON) -c "from pathlib import Path; \
	from tmclass import data, pipeline; \
	[pipeline.cluster(data.write_fasta((data.single_tm_only if c == 'type_i' else list)(data.parse_tsv(data.fetch_class(c, Path('data')/f'{c}.tsv'))), Path('data')/f'{c}.fasta'), Path('data')/f'{c}_nr40.fasta') \
	 for c in ('type_i', 'type_ii', 'gpcr', 'globular')]"

## Both models, 5-fold cross-validation of the TM-region model, and figures
analysis: data
	$(PYTHON) -m tmclass.cli

## TM-region model only, no cross-validation, no ESM-2 benchmark
quick: data
	$(PYTHON) -m tmclass.cli --models tm_region --folds 0 --heads

## Cross-validate both models (slow: each full-length fold is a whole-protein alignment)
cv: data
	$(PYTHON) -m tmclass.cli --cv-models full_length tm_region

## Regenerate the pinned UniProt snapshot used when UniProt is unreachable.
## Changes the dataset any fallback run produces, so commit it on its own.
snapshot:
	$(PYTHON) -m tmclass.refresh_snapshot

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf results/*
	find . -name __pycache__ -type d -exec rm -rf {} +

## Also delete the cached UniProt downloads and clustering
clean-data: clean
	rm -f data/*.tsv data/*.fasta data/*.clstr
