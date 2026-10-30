# Personalised re-ranking for JD search (BT4222 Group 17)

JD shows a shopper about 89 products for a search. We **re-rank that list** using the query, the shopper's earlier behaviour and product metadata, and ask how much past behaviour should shape the ranking and which parts of it to trust. Data: the [JDsearch dataset](https://github.com/rucliujn/JDsearch) (Liu et al., SIGIR 2023), 173,831 searches with the outcome of every shown product (none, click, add to cart, purchase).

## Results

NDCG@10 on `validation_clean` (18,844 validation searches; complete candidate lists, linear gains 0-3, exact under random tie-breaking). Models trained on all clean training searches. Test results: notebook 07.

| Ranking | NDCG@10 |
|---|---:|
| Random order | 0.084 |
| Team 3 (prior baseline), re-implemented on our protocol (all validation searches) | 0.151 |
| Lagged history popularity (untrained, deployable) | 0.337 |
| LambdaMART base: query length, query-title relevance, lagged popularity | 0.350 |
| Untrained rule: shopper's own products first, then the base | 0.384 |
| **M1: base + four pooled history match counts** | **0.389** |
| M3: history split by action type and age | 0.390 |
| **M1 + item-item collaborative features (final LambdaMART)** | **0.392** |
| Feature MLP (same inputs as M1 + CF) | 0.389 |
| History attention over the last 50 events (regularised) | 0.387 |

## Pipeline

Run the notebooks in order. Each one starts with the same setup cell: on Colab it clones this repository (for `jdsearch.py`) and mounts Google Drive; all files are read from and written to the working folder `MyDrive/BT4222-G17-JDsearch/`, so every notebook finds the outputs of the previous ones.

| Notebook | What it does | Runtime | Writes (working folder) |
|---|---|---|---|
| `01_data_understanding` | Full-data summaries, label and history distributions, file-order audit, feature dictionary | CPU (high RAM), ~10 min | `outputs/data_understanding/` |
| `02_split_and_evaluation` | Search table, fixed train / validation / test split by whole search, evaluator checks, random-order floor | CPU, ~5 min | `processed/search_table.parquet`, `processed/split_v1.*` |
| `03_features` | 97 features in 13 groups for all splits (relevance, popularity, personal history, context, query history), plus a 10-minute history safety-gap version | CPU (high RAM), ~45 min | `processed/features_v3_*` |
| `04_baselines` | Untrained single-signal references; Team 3's evaluation protocol applied to known rankings | CPU, ~5 min | `results.csv` |
| `04b_team3_prep` | Inputs for our re-implementation of Team 3's model (the prior baseline) | CPU, ~20 min | `processed/team3/` |
| `04c_team3_ladder` | Team 3's model re-implemented and corrected one flaw at a time | T4 GPU, ~45 min | `processed/team3/ladder_*`, `results.csv` |
| `05_lambdamart` | LambdaMART: base, ablations, item-item collaborative features, final models on all training searches, robustness and sensitivity checks; test scores (no labels) | CPU (high RAM), ~3-4 h | `processed/models/`, `processed/cf/`, `processed/neural/lambdamart_validation_scores.parquet`, `processed/test/` |
| `05b_neural` | Feature MLP and three history-attention variants; test scores (no labels) | T4 GPU (high RAM), ~15 min | `processed/neural/`, `processed/test/` |
| `06_diagnostics` | Personal gain by segment, wins and losses, own-history / warm / cold candidates, report figures | CPU, ~5 min | `outputs/diagnostics/` |
| `07_test` | Checks that this run reproduces the reported validation numbers, then evaluates the final models once on test | CPU, ~5 min | `results.csv`, `outputs/test_*.csv` |

Every evaluated ranking is written to `results.csv` in the working folder (notebook, step, model, split, NDCG@10 with 95% bootstrap interval, Recall@10, MRR, commit).

**Running on Colab.** Open a notebook from GitHub (*File > Open notebook > GitHub*, repository `liamkunert/BT4222-group17-jdsearch-submission`), choose the runtime from the table and *Run all*. Notebook 05 caches every trained model in the working folder, so after a disconnect a re-run continues where it stopped. Notebooks 04c and 05b fall back to a short smoke test without a GPU.

**Running a single notebook.** Inputs that are missing from the working folder are downloaded from the public data folder (see Data), so any notebook can also be run on its own.

**Running locally.** Python 3.12 and `pip install -r requirements.txt`; the working folder is `./run/` (or the folder in the `JDSEARCH_WORK` environment variable). The full pipeline needs about 16 GB of RAM.

## Data

- **Raw data** (`user_behavior_data.txt`, `product_meta_data.txt`) and **all intermediate and final artefacts** of our run (split, features, CF features, models, scores, results, figures) are in the public Google Drive folder: *link to be added after the reference run*.
- The raw data is not stored in this repository.

## Repository

```text
README.md          this file
requirements.txt   package versions
jdsearch.py        shared code: setup and paths, reading the raw files, text and history matching, evaluation
notebooks/         the pipeline, 01 to 07, executed with all outputs
```

`jdsearch.py` holds only what several notebooks share; every modelling step is in the notebooks, with markdown explaining purpose, approach and results.
