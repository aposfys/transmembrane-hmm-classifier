.PHONY: install data analysis quick cv snapshot test clean all

PYTHON ?= python3

all: analysis

## Install the Python package. External tools (HMMER, Clustal Omega, CD-HIT)
## come from environment.yml:
##   conda env create -f environment.yml && conda activate tmclass
install:
	$(PYTHON) -m pip install -e ".[dev]"

## Fetch all four sequence classes from UniProt and cluster them at 40% identity.
## This is the slowest step (~40 min for CD-HIT on the type I set) but its
## output is cached, so later runs skip it.
data:
	$(PYTHON) -c "from pathlib import Path; \
	from tmclass import data, pipeline; \
	[pipeline.cluster(data.write_fasta(data.parse_tsv(data.fetch_class(c, Path('data')/f'{c}.tsv')), Path('data')/f'{c}.fasta'), Path('data')/f'{c}_nr40.fasta') \
	 for c in ('type_i', 'type_ii', 'gpcr', 'globular')]"

## Both models, 5-fold cross-validation of the TM-region model, and figures
analysis: data
	$(PYTHON) -m tmclass.cli

## TM-region model only, no cross-validation
quick: data
	$(PYTHON) -m tmclass.cli --models tm_region --folds 0

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
