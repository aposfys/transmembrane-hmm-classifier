# Running the pipeline

```
make quick                                        # TM-region HMM only, no benchmark
python -m tmclass.cli --heads                     # HMMs only, skip the language model
python -m tmclass.cli --snapshot                  # rebuild the published dataset exactly
python -m tmclass.cli --snapshot \
  --reuse-benchmark results/findings.json          # re-run the HMMs, keep the ESM-2 results
python -m tmclass.cli --esm-model 35M             # smaller, faster embeddings
python -m tmclass.cli --esm-model 650M --device cpu
python -m tmclass.cli --padding 0                 # membrane segment with no flanks
```

**Smoke runs.** `--max-sequences-per-class N` truncates each UniProt class so the whole
pipeline finishes in about a minute. Useful for checking a change end to end without a
three-hour wait; the numbers it produces are not comparable to the published ones.

```
python -m tmclass.cli --max-sequences-per-class 120 --max-train 40 \
  --folds 0 --heads logreg --esm-model 8M
```

**Runtime.** CD-HIT on the 6,343-sequence type I set takes about 2 minutes on an M4 and is
cached against a checksum of its input. A full `make analysis` takes roughly 3 hours on an
M4, dominated by ESM-2 650M embedding of ~2,400 sequences. The HMM side alone, with
`--reuse-benchmark`, took under 7 minutes from the snapshot. `--esm-model 35M` cuts that to about
25 minutes with a modest accuracy cost. Embeddings are cached by model and sequence set, so
re-running a head is instant.

**Optional dependencies.** The base install needs only Biopython and NumPy.
`pip install -e ".[embeddings]"` adds torch, transformers and scikit-learn. The CLI imports
them lazily, so the HMM path works without them when run with `--heads` and nothing after
it. `findings.json` is written before the embedding stage, so an ESM-2 failure keeps the
HMM results.

**Reusing the ESM-2 results.** `--reuse-benchmark FINDINGS` copies the `esm_*` blocks of an
earlier `findings.json` instead of embedding again, and records where they came from under
`benchmark_source`. It refuses a file whose dataset counts differ from the current run. The
heads are scored on the shared test set alone, so nothing on the HMM side changes them.

## Data fetching and the offline fallback

UniProt is the only source that carries both the subcellular-location terms and the
`ft_transmem` coordinates these class definitions rely on, so there is no equivalent
database to fail over to. What there is instead is a ladder of three sources, tried in
order:

1. **`/stream`**, one request for a whole class. The fast path when UniProt is healthy.
2. **Paginated `/search`**, the same query walked in 200-record pages, following cursor
   links. More requests, but a stall costs one page rather than the whole class. This is
   what carries a run through a degraded UniProt.
3. **The pinned snapshot**, `src/tmclass/snapshot/`, a gzipped copy of all four classes
   with a checksum per class. Used only when both live routes are exhausted.

Every step retries with exponential backoff. `IncompleteRead` counts as transient: UniProt's
stream endpoint truncates chunked responses under load, and because that exception is *not*
a `URLError` an earlier version let it escape the retry loop and kill the run outright.

Live attempts are capped by `--uniprot-budget` (default 300 s per class, plus up to one 60 s
request already in flight). **An unbounded wait is not a fallback.** The first version of
this had no ceiling, and a stalled UniProt outlasted the 30-minute CI limit, so the job was
killed before it ever reached the snapshot it was carrying.

**The fallback is never silent.** Whichever source is used is recorded per class under
`data_source` in `findings.json`, and a run that touches the snapshot prints a warning
naming the snapshot's date. A run that finds a cached TSV in the data directory records
only `cache`, which does not say where that TSV came from. `--snapshot` reads the pinned
copy over any cache and records `"reason": "requested with --snapshot"`, which is how the
committed results were produced.

```json
"data_source": {
  "type_i":   {"source": "uniprot"},
  "type_ii":  {"source": "snapshot", "snapshot_date": "2026-08-21",
               "uniprot_error": "UniProt request failed after 5 attempts: HTTP Error 500"}
}
```

Pass `--no-snapshot-fallback` to make a run fail rather than accept the stale copy,
appropriate when the results must reflect current UniProt. Refresh the snapshot with
`make snapshot`; because it changes the dataset any fallback run produces, commit it on its
own.
