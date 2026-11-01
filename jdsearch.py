"""Shared code for the JDsearch re-ranking pipeline (notebooks 01-07).

Four parts:
1. ``setup``: paths, the working folder (Google Drive on Colab) and downloads of missing inputs from the public
   data folder, so any notebook can run on its own.
2. Reading the raw files: ``iter_records``, ``search_table``, ``load_candidates``, ``load_metadata``.
3. Text and history matching: query terms (``tokenize``) and metadata keys for history matches.
4. Evaluation: NDCG@10, Recall@10 and MRR, exact under random tie-breaking; bootstrap intervals; paired
   differences; and the results table every notebook writes to.

``session_id`` is always the 0-based data row of ``user_behavior_data.txt`` (header excluded). Candidate file
order encodes the label (engaged candidates come first), so nothing here returns a candidate position and every
metric breaks ties at random, never by file order.
"""
from collections import Counter
from datetime import date
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------------------------------------------
REPO_URL = "https://github.com/liamkunert/BT4222-group17-jdsearch-submission.git"
WORK_FOLDER = "BT4222-G17-JDsearch"  # working folder in the runner's Google Drive (Colab)
# Public, read-only Google Drive folder with the raw data and every artefact of our reference run. Missing inputs
# are downloaded from it by file ID (path relative to the working folder -> Drive file ID): every file a notebook
# reads, so each notebook can also run on its own.
PUBLIC_FOLDER = "https://drive.google.com/drive/folders/1PGXci7tvDOh4Wh2HX0VPUO5cl_pAJNSN"
PUBLIC_FILES = {
    "data/user_behavior_data.txt": "1Vefw_BZ9qvVVEsveBdnfb7o3m3NRX7oH",
    "data/product_meta_data.txt": "1BYpLJIFCf3zpr1bd5UCqkSAoLDu6xwfn",
    "processed/search_table.parquet": "1GujuMtxsFwWhzRp_t2AnuoebNmyVys76",
    "processed/split_v1.csv": "1nuRIItA_23jQOOwORB9uoz3zs9sKm5Y4",
    "processed/features_v3_meta.json": "1JypWkB4HgJflJS8CFpZopuLQXUBNHanL",
    "processed/features_v3_train_dev.parquet": "1GyTzsb21EacpEhDYExsHb6wAhOVhKnu0",
    "processed/features_v3_train_rest.parquet": "1JHq6DuJJc8CU9b_oFji1NL18Ki2y-y8V",
    "processed/features_v3_validation.parquet": "11y5t-OD53UWPrnY-7YApMThUXaBZLkJq",
    "processed/features_v3_test.parquet": "1xFpA7zRkqqDwBC0RYCMFk2qtwgSdqwxB",
    "processed/features_v3_minage0.00694444d_meta.json": "1zsA26-IbYfXAdCjNzvj3IZHMetz7FZST",
    "processed/features_v3_minage0.00694444d_train_dev.parquet": "1_z9c4Vvy7dRHohHH2aaf_XzLHk2P7vCr",
    "processed/features_v3_minage0.00694444d_validation.parquet": "1C7LGPhEhtjWffcbrOxyOyF3s1oJ8XW8e",
    "processed/cf/cf1_train_dev.parquet": "1spjJ_5-XdvCUl2MpK-aTSzpqBqjiD87S",
    "processed/cf/cf1_train_rest.parquet": "1vYrP7VyrASap0k-Jh2_wp6JFoNqxFuck",
    "processed/cf/cf1_validation.parquet": "1cAFbD1k7SogOGiigxgDV35I6aUvsURXp",
    "processed/cf/cf1_test.parquet": "1zn4AR8SKra6h-DEHWPHJbRmdg9piPDCM",
    "processed/team3/train_dev.npz": "16xMjklOJB4URU3BrulBbV1sECLnitimm",
    "processed/team3/validation.npz": "1-769HZlJfh0J0qh-jsztcHK8FnEI4ebz",
    "processed/team3/lookups.npz": "1n6LjML35vX2xeIjff9gT4khybQEc1kwR",
    "processed/team3/meta.json": "13YfQujy1zH12tQ7VhOqHHDK6cMbYtH24",
    "processed/neural/lambdamart_validation_scores.parquet": "1kFvuyI0s9swtPow7DDiSpI22KEkpnIAq",
    "processed/test/lambdamart_test_scores.parquet": "1HqCk0X12WxmIWJVfKTfTaKyPZg8uG25m",
    "processed/test/neural_test_scores.parquet": "1Bq7WwAzRHghgZoaolqDAOyOPYxs7ho5k",
    "results.csv": "1Qe7ivomV2ZRMc7DEW8H23LTm9n35iD4V",
}
IN_COLAB = "google.colab" in sys.modules
REPO_ROOT = Path(__file__).resolve().parent


class Paths:
    """Folders of one run: ``data`` (raw files), ``processed`` (features, models, scores), ``outputs`` (tables,
    figures) and the results table."""

    def __init__(self, work):
        self.work = Path(work)
        self.data = Path(os.environ.get("JDSEARCH_DATA", self.work / "data"))  # raw files elsewhere (local runs)
        self.processed = Path(os.environ.get("JDSEARCH_PROCESSED", self.work / "processed"))
        self.outputs = self.work / "outputs"
        self.results = self.work / "results.csv"
        self.behaviour = self.data / "user_behavior_data.txt"
        self.catalogue = self.data / "product_meta_data.txt"
        for folder in (self.data, self.processed, self.outputs):
            folder.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return f"Paths(work={self.work})"


def setup(packages=None):
    """Prepare a notebook run and return its ``Paths``.

    On Colab the runner's Google Drive is mounted and ``MyDrive/<WORK_FOLDER>`` is the working folder, so files
    written by one notebook are there for the next (each Colab notebook runs on its own machine). Locally the
    working folder is ``./run`` next to this file, or the folder in the ``JDSEARCH_WORK`` environment variable.
    ``packages`` maps package names to exact versions that are installed if a different version is present.
    """
    for name, version in (packages or {}).items():
        try:
            from importlib.metadata import version as installed
            current = installed(name)
        except Exception:
            current = None
        if current != version:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"{name}=={version}"], check=True)
    if "JDSEARCH_WORK" in os.environ:
        work = Path(os.environ["JDSEARCH_WORK"])
    elif IN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive")
        work = Path("/content/drive/MyDrive") / WORK_FOLDER
    else:
        work = REPO_ROOT / "run"
    paths = Paths(work)
    print(f"working folder: {paths.work}")
    return paths


def require(paths, *relative):
    """Make sure input files exist in the working folder; download missing ones from the public folder."""
    for name in relative:
        folder, _, rest = name.partition("/")
        target = {"data": paths.data, "processed": paths.processed}.get(folder, paths.work / folder) / rest
        if target.is_file():
            continue
        if name not in PUBLIC_FILES:
            raise FileNotFoundError(f"{target} is missing: run the notebook that writes it, or download it from the "
                                    f"public data folder {PUBLIC_FOLDER or '(link in the README)'}")
        import gdown
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {name} from the public data folder")
        gdown.download(id=PUBLIC_FILES[name], output=str(target), quiet=True)


def commit():
    """Short commit hash of the code that produced a result (``unknown`` outside a git checkout)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# ---------------------------------------------------------------------------------------------------------------
# 2. Reading the raw files
# ---------------------------------------------------------------------------------------------------------------
HEADER = ["query", "candidate_wid_list", "candidate_label_list", "history_qry_list",
          "history_wid_list", "history_type_list", "history_time_list"]
SECONDS_PER_DAY = 86400
EXPLORED_SIZE, EXPLORED_SEED = 5000, 4222  # the sample explored before the split (interim report)


def iter_records(path):
    """Yield (session_id, fields) for every behaviour record."""
    with Path(path).open(encoding="utf-8") as handle:
        assert next(handle).rstrip("\r\n").split("\t") == HEADER
        for session_id, line in enumerate(handle):
            fields = line.rstrip("\r\n").split("\t")
            assert len(fields) == len(HEADER)
            yield session_id, fields


def elapsed_days(gaps):
    """Days from each past action to the search; ``gaps`` = [0, g1, ..., gn, gap to the search] in seconds."""
    gaps = np.asarray(gaps, dtype=np.int64)
    return np.cumsum(gaps[::-1])[::-1][1:] / SECONDS_PER_DAY


def explored_ids(n_records, size=EXPLORED_SIZE, seed=EXPLORED_SEED):
    """Session IDs of the sample explored before the split (kept out of validation and test)."""
    return np.sort(np.random.default_rng(seed).choice(n_records, size, replace=False))


def search_table(path):
    """One row per search: list size, label counts and history context used for splits and segments."""
    rows = []
    for session_id, fields in iter_records(path):
        labels = np.array([int(float(v)) for v in fields[2].split("_")])
        types = np.array(fields[5].split("_"))
        gaps = np.array(fields[6].split("_"), dtype=np.int64)
        assert len(gaps) == len(types) + 1 and len(labels) == fields[1].count("_") + 1
        elapsed = elapsed_days(gaps)
        rows.append((session_id, len(labels), int((labels >= 1).sum()), int((labels == 3).sum()), len(types),
                     int((types == "CLICK").sum()), int((types == "CART").sum()), int((types == "ORD").sum()),
                     int((types == "FLW").sum()), int((elapsed < 7).sum()), gaps[-1] / SECONDS_PER_DAY, int(gaps[-1])))
    table = pd.DataFrame(rows, columns=[
        "session_id", "n_candidates", "n_engaged", "n_purchased", "history_len", "n_click", "n_cart", "n_ord",
        "n_flw", "actions_last_7d", "days_since_last_action", "final_gap_s"]).set_index("session_id")
    # The final gap is never 0 but exactly 1 s in 5.3% of records: most likely gaps <= 0 clamped to 1 s, i.e. the
    # history contains actions from during or after the final search. Flagged here and excluded downstream.
    table["final_gap_clamped"] = table.final_gap_s == 1
    return table


def load_metadata(path, wids):
    """Catalogue rows for the given products: wid -> (brand, shop, cate_1, cate_2, cate_3, cate_4).

    Empty or -1 values are None. Products without a catalogue row are absent; ``metadata_keys`` handles them.
    """
    wids, metadata = set(wids), {}
    with Path(path).open(encoding="utf-8") as handle:
        assert next(handle).split("\t", 1)[0] == "wid"
        for line in handle:
            wid = line.split("\t", 1)[0]
            if wid in wids:
                fields = line.rstrip("\r\n").split("\t")
                metadata[wid] = tuple(int(fields[j]) if fields[j] not in ("", "-1") else None
                                      for j in (2, 12, 4, 6, 8, 10))
    return metadata


def load_candidates(path, session_ids, shuffle_seed=EXPLORED_SEED):
    """Long table (session_id, wid, label) for the given searches; one row per candidate occurrence.

    Candidates are shuffled within each search (seeded), because the file stores engaged candidates first: a model
    must never see that order, not even through row order during training. ``shuffle_seed=None`` keeps file order
    (only for checks).
    """
    selected = set(np.asarray(session_ids).tolist())
    rng = None if shuffle_seed is None else np.random.default_rng(shuffle_seed)
    parts = []
    for session_id, fields in iter_records(path):
        if session_id in selected:
            wids = np.array(fields[1].split("_"))
            labels = np.array([int(float(v)) for v in fields[2].split("_")])
            if rng is not None:
                order = rng.permutation(len(wids))
                wids, labels = wids[order], labels[order]
            parts.append(pd.DataFrame({"session_id": session_id, "wid": wids, "label": labels}))
    frame = pd.concat(parts, ignore_index=True)
    frame["label"] = frame["label"].astype(np.int8)
    assert frame.session_id.nunique() == len(selected)
    return frame


# ---------------------------------------------------------------------------------------------------------------
# 3. Text and history matching
# ---------------------------------------------------------------------------------------------------------------
TERM_SEPARATOR = "\x18"  # anonymised term IDs are separated by the control character ^X
LEVELS = ("product", "brand", "shop", "L1", "L2", "L3", "L4")
SOURCES = ("all", "search", "queryless")


def tokenize(text):
    """Ordered term IDs; the query-less marker -1 and empty metadata are not terms."""
    if text is None or text in ("", "-1"):
        return ()
    terms = tuple(text.split(TERM_SEPARATOR))
    if not all(term.isdecimal() for term in terms):
        raise ValueError(f"Invalid anonymised term sequence: {text!r}")
    return terms


def token_set(text):
    return frozenset(tokenize(text))


def overlap_features(query_terms, document_terms):
    """(shared terms, share of the query covered, Jaccard) between two term sets."""
    shared = len(query_terms & document_terms)
    union = len(query_terms | document_terms)
    return shared, shared / len(query_terms) if query_terms else 0.0, shared / union if union else 0.0


def metadata_keys(wid, metadata):
    """Match keys of a product: (wid, brand, shop, L1-L4 category paths).

    Categories use the complete ancestor path, because some category IDs have several parents. Missing metadata
    never matches other missing metadata; product identity is known even without a catalogue row.
    """
    if wid not in metadata:
        return (wid, None, None, None, None, None, None)
    brand, shop, *categories = metadata[wid]
    paths = []
    for depth in range(1, 5):
        path = tuple(categories[:depth])
        paths.append(path if all(v is not None for v in path) else None)
    return (wid, brand, shop, *paths)


def match_counts(candidates, past_items, has_query, metadata):
    """History match counts per candidate and level, for all history and split by search-linked / query-less."""
    candidate_keys = [metadata_keys(wid, metadata) for wid in candidates]
    counters = {source: [Counter() for _ in LEVELS] for source in SOURCES}
    eligible_history = {source: np.zeros(len(LEVELS), dtype=int) for source in SOURCES}
    for wid, from_search in zip(past_items, has_query):
        keys = metadata_keys(wid, metadata)
        source = "search" if from_search else "queryless"
        for j, key in enumerate(keys):
            if key is not None:
                for target in ("all", source):
                    counters[target][j][key] += 1
                    eligible_history[target][j] += 1
    scores = {source: np.asarray([[counters[source][j][key] if key is not None else 0
                                  for j, key in enumerate(keys)] for keys in candidate_keys])
              for source in SOURCES}
    known = np.asarray([[key is not None for key in keys] for keys in candidate_keys])
    assert np.array_equal(scores["all"], scores["search"] + scores["queryless"])
    assert np.all(scores["all"][:, 4:] <= scores["all"][:, 3:-1])  # a deeper path match implies its ancestor match
    return scores, known, candidate_keys, eligible_history


# ---------------------------------------------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------------------------------------------
K = 10
RESULT_COLUMNS = ["date", "notebook", "step", "model", "split", "searches", "ndcg_at_10", "ndcg_ci_low",
                  "ndcg_ci_high", "recall_at_10", "mrr", "commit", "notes"]


def _tie_blocks(scores):
    """Sort by descending score; return the order and the [start, end) positions of each tie group."""
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    return order, starts, np.r_[starts[1:], len(scores)]


def _expected_reciprocal_rank(start, size, engaged):
    """E[1 / rank of the first engaged item] when `engaged` of `size` tied items fill positions start.. ."""
    j = np.arange(size - engaged + 1)
    # P(first engaged offset >= j) = C(size - j, engaged) / C(size, engaged), as a running product.
    survival = np.r_[1.0, np.cumprod((size - engaged - j[:-1]) / (size - j[:-1]))] if len(j) > 1 else np.ones(1)
    probability = survival - np.r_[survival[1:], 0.0]
    return float(np.dot(probability, 1.0 / (start + j + 1)))


def expected_metrics(scores, labels, k=K):
    """(NDCG@k, Recall@k, MRR) for one search, exact under random tie-breaking; NaNs if nothing is engaged.

    Gains are the labels 0-3 (linear); "engaged" means label >= 1. Within a group of tied scores every order is
    equally likely, so each metric is the exact expectation over those orders.
    """
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    if len(scores) != len(labels) or not np.all(np.isfinite(scores)):
        raise ValueError("Scores and labels must be aligned and finite")
    engaged = labels >= 1
    if not engaged.any():
        return np.nan, np.nan, np.nan
    discount = 1 / np.log2(np.arange(2, min(k, len(labels)) + 2))
    ideal = np.dot(np.sort(labels)[::-1][:len(discount)], discount)
    order, starts, ends = _tie_blocks(scores)
    gains, hits = labels[order], engaged[order]
    dcg = found = 0.0
    mrr = None
    for start, end in zip(starts, ends):
        size, block_hits = end - start, hits[start:end].sum()
        if start < len(discount):
            slots = discount[start:min(end, len(discount))]
            dcg += gains[start:end].mean() * slots.sum()
            found += block_hits / size * len(slots)
        if mrr is None and block_hits:
            mrr = _expected_reciprocal_rank(start, size, int(block_hits))
        if start >= len(discount) and mrr is not None:
            break
    return dcg / ideal, found / engaged.sum(), mrr


def check_complete(frame, expected_sizes):
    """Fail unless `frame` holds exactly the expected searches, each with its complete candidate list."""
    sizes = frame.groupby("session_id").size()
    expected = pd.Series(expected_sizes)
    missing, extra = expected.index.difference(sizes.index), sizes.index.difference(expected.index)
    if len(missing) or len(extra):
        raise ValueError(f"{len(missing)} expected searches missing, {len(extra)} unexpected searches present")
    wrong = (sizes.reindex(expected.index) != expected).sum()
    if wrong:
        raise ValueError(f"{wrong} searches have incomplete or extra candidates")


def evaluate(frame, score_col="score", k=K, expected_sizes=None):
    """Per-search metrics for a long table with columns session_id, label and `score_col`.

    Searches without an engaged candidate are dropped (every order scores the same). Pass `expected_sizes`
    (session_id -> list length) to check first that every list is complete.
    """
    if not frame.label.isin([0, 1, 2, 3]).all():
        raise ValueError("Labels must be 0, 1, 2 or 3")
    if not np.isfinite(frame[score_col].to_numpy(dtype=float)).all():
        raise ValueError(f"Scores in {score_col!r} must be finite")
    if expected_sizes is not None:
        check_complete(frame, expected_sizes)
    frame = frame.sort_values("session_id", kind="stable")
    sessions = frame.session_id.to_numpy()
    scores, labels = frame[score_col].to_numpy(dtype=float), frame.label.to_numpy()
    bounds = np.r_[0, np.flatnonzero(np.diff(sessions)) + 1, len(sessions)]
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        ndcg, recall, mrr = expected_metrics(scores[lo:hi], labels[lo:hi], k)
        rows.append((sessions[lo], hi - lo, int((labels[lo:hi] >= 1).sum()), ndcg, recall, mrr))
    per_search = pd.DataFrame(rows, columns=["session_id", "n_candidates", "n_engaged", "ndcg", "recall", "mrr"])
    return per_search[per_search.n_engaged > 0].set_index("session_id")


def bootstrap_interval(values, seed=4222, repeats=1000):
    """Mean and 95% percentile interval over resampled searches."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(repeats)])
    return values.mean(), *np.quantile(means, [0.025, 0.975])


def summarise(per_search, by=None):
    """Mean metrics with a bootstrap interval for NDCG; `by` is a Series of segment labels per search."""
    def one(part):
        mean, low, high = bootstrap_interval(part.ndcg)
        return pd.Series({"searches": len(part), "ndcg_at_10": mean, "ndcg_ci_low": low, "ndcg_ci_high": high,
                          "recall_at_10": part.recall.mean(), "mrr": part.mrr.mean()})
    if by is None:
        return one(per_search).to_frame("all").T
    return per_search.groupby(by.reindex(per_search.index), observed=True).apply(one)


def paired_difference(per_a, per_b, metric="ndcg"):
    """Mean of A minus B over the same searches, with a bootstrap interval; fails if the searches differ."""
    if not per_a.index.sort_values().equals(per_b.index.sort_values()):
        raise ValueError("Paired comparison needs exactly the same searches in both results")
    a, b = per_a[metric].align(per_b[metric])
    return bootstrap_interval(a - b)


def log_result(paths, notebook, step, model, split, per_search, notes=""):
    """Write one evaluated ranking to the run's results table (replacing an earlier row with the same key)."""
    summary = summarise(per_search).iloc[0]
    row = {"date": date.today().isoformat(), "notebook": notebook, "step": step, "model": model, "split": split,
           "searches": int(summary.searches), **{c: round(float(summary[c]), 5) for c in RESULT_COLUMNS[6:11]},
           "commit": commit(), "notes": notes}
    table = read_results(paths)
    key = (table.notebook == notebook) & (table.model == model) & (table.split == split)
    new = pd.DataFrame([row], columns=RESULT_COLUMNS)
    table = new if not (~key).any() else pd.concat([table[~key], new], ignore_index=True)
    table.to_csv(paths.results, index=False)
    return pd.Series(row)


def read_results(paths):
    """The run's results table (empty if nothing is logged yet)."""
    if Path(paths.results).is_file():
        return pd.read_csv(paths.results)
    return pd.DataFrame(columns=RESULT_COLUMNS)


def logged(paths, model, split="validation_clean"):
    """NDCG@10 of one logged ranking."""
    table = read_results(paths)
    rows = table[(table.model == model) & (table.split == split)]
    if len(rows) != 1:
        raise KeyError(f"{model!r} ({split}) is logged {len(rows)} times")
    return float(rows.ndcg_at_10.iloc[0])


# ---------------------------------------------------------------------------------------------------------------
# Self-checks (run by notebook 02)
# ---------------------------------------------------------------------------------------------------------------
def validate():
    """Hand-checked cases for the evaluator and the matching helpers; raises AssertionError on any change."""
    # Evaluation: perfect, worst and tied orders, Recall and MRR under ties.
    assert expected_metrics([1, 0], [3, 0], 1)[0] == 1.0
    assert expected_metrics([0, 1], [3, 0], 1)[0] == 0.0
    assert expected_metrics([0, 0], [3, 0], 1)[0] == 0.5
    assert np.isnan(expected_metrics([0, 0], [0, 0])[0])
    assert np.isclose(expected_metrics([0, 0, 0], [1, 0, 0], 1)[1], 1 / 3)
    assert np.isclose(expected_metrics([0, 0], [1, 0])[2], 0.75)
    assert expected_metrics([1, 1, 0], [3, 0, 1]) == expected_metrics([0, 1, 1], [1, 0, 3])  # permutation-invariant
    # Text: partial digit matches, order and placeholders.
    assert tokenize("12\x1832\x1856") == ("12", "32", "56")
    assert token_set("12\x1832\x1812") == token_set("32\x1812")
    assert not token_set("-1") and not token_set("")
    assert not token_set("123456789") & token_set("23456789")
    assert overlap_features(token_set("12\x1832"), token_set("32\x1856")) == (1, 0.5, 1 / 3)
    for bad in ("12_32", "12 32", "12\x18", "-1\x1812"):
        try:
            tokenize(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted malformed text: {bad!r}")
    # History matches: category paths, missing metadata, search-linked vs query-less history.
    meta = {"a": (10, 20, 1, 2, 3, 4), "b": (10, 21, 1, 2, 3, 5),
            "c": (11, 22, 1, 7, 3, 4), "d": (None, None, 1, 2, 3, None)}
    scores, known, _, _ = match_counts(["a", "b", "c", "d", "missing"],
                                      ["a", "a", "b", "missing"], [True, False, False, True], meta)
    assert scores["all"][0].tolist() == [2, 3, 2, 3, 3, 3, 2]
    assert scores["search"][0, 0] == 1 and scores["queryless"][0, 0] == 1
    assert scores["all"][2, 5] == 0  # same child ID, different ancestor: no L3 path match
    assert scores["all"][3, 6] == 0 and not known[3, 6]  # missing L4
    assert scores["all"][4, 0] == 1 and not known[4, 1:].any()
    return True
