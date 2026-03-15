"""
Open LLM Leaderboard: Model Scores (Multithreaded)

Downloads per-instance model scores from HuggingFace's Open LLM Leaderboard
and saves them as dense NumPy matrices with JSON metadata.

Uses load_dataset per config with 8-thread concurrency and retry logic
with exponential backoff on 429 rate limit errors.
"""

import json
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent

OWNER = "open-llm-leaderboard"
SPLIT = "latest"
MAX_WORKERS = 8
MAX_RETRIES = 5
INITIAL_BACKOFF = 2
CORRECTNESS_COLS = ["acc", "acc_norm", "exact_match"]


def setup():
    warnings.filterwarnings(
        "ignore",
        message=".*huggingface_hub.*cache-system uses symlinks.*",
    )
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"


def download_model_subset(owner, model, model_idx, benchmark, subset, split):
    """Download scores for a single model/subset combination.

    Returns a list of (doc_hash, model_idx, score) tuples.
    """
    repo_id = f"{owner}/{model}-details"
    config = f"{model}__leaderboard_{benchmark}_{subset}"

    for attempt in range(MAX_RETRIES):
        try:
            df = load_dataset(repo_id, config, split=split).to_pandas()
            break
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                wait = INITIAL_BACKOFF * (2**attempt)
                print(f"  Rate limited on {model}/{subset}, retrying in {wait}s...")
                time.sleep(wait)
                continue
            print(f"  Skipping {model}/{subset}: {e}")
            return []

    col = next((c for c in CORRECTNESS_COLS if c in df.columns), None)
    if col is None:
        print(f"  No correctness column in {model}/{subset}: {list(df.columns)}")
        return []

    results = []
    for _, row in df[["doc_hash", col]].iterrows():
        results.append((row["doc_hash"], model_idx, float(row[col])))
    return results


def download_benchmark_scores(benchmark, subsets, models):
    """Download and save all model scores for a single benchmark."""
    data_dir = BASE_DIR / "data" / benchmark
    scores_path = data_dir / "model_scores.npy"
    meta_path = data_dir / "model_scores_metadata.json"

    if scores_path.exists() and meta_path.exists():
        print(f"Skipping {benchmark} — already exists")
        return

    data_dir.mkdir(parents=True, exist_ok=True)

    # Submit all model×subset tasks concurrently
    futures = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for model_idx, model in enumerate(models):
            for subset in subsets:
                future = executor.submit(
                    download_model_subset,
                    OWNER,
                    model,
                    model_idx,
                    benchmark,
                    subset,
                    SPLIT,
                )
                futures[future] = (model_idx, model, subset)

        # Collect results with a progress bar
        instance_scores: dict[str, dict[int, float]] = {}
        total = len(futures)
        with tqdm(total=total, desc=f"{benchmark}", unit="task") as pbar:
            for future in as_completed(futures):
                for doc_hash, model_idx, score in future.result():
                    if doc_hash not in instance_scores:
                        instance_scores[doc_hash] = {}
                    instance_scores[doc_hash][model_idx] = score
                pbar.update(1)

    # Build dense float32 matrix; NaN = no result for that model/instance
    doc_hashes = list(instance_scores.keys())
    matrix = np.full((len(doc_hashes), len(models)), np.nan, dtype=np.float32)
    for row_idx, doc_hash in enumerate(doc_hashes):
        for col_idx, score in instance_scores[doc_hash].items():
            matrix[row_idx, col_idx] = score

    np.save(scores_path, matrix)
    with open(meta_path, "w") as f:
        json.dump({"doc_hashes": doc_hashes, "models": models}, f)

    print(
        f"Saved {len(doc_hashes)} instances × {len(models)} models → {scores_path}, {meta_path}"
    )


def display_scores(benchmark="bbh"):
    """Load and display the per-instance model scores for a benchmark."""
    data_dir = BASE_DIR / "data" / benchmark
    scores_path = data_dir / "model_scores.npy"
    meta_path = data_dir / "model_scores_metadata.json"

    if not scores_path.exists() or not meta_path.exists():
        print(f"No saved scores found for {benchmark}")
        return

    matrix = np.load(scores_path)
    meta = json.load(open(meta_path))

    df = pd.DataFrame(matrix, columns=meta["models"])
    df.insert(0, "doc_hash", meta["doc_hashes"])

    print(f"Shape: {df.shape}")
    print(df.head())


def main():
    setup()

    models_path = BASE_DIR / "resources" / "models.json"
    models = json.load(open(models_path))
    print(f"Number of models: {len(models)}")

    num_preview = 5
    for i, model in enumerate(models[:num_preview]):
        print(f"Model {i+1}: {model}")

    benchmarks_path = BASE_DIR / "resources" / "benchmarks.json"
    benchmarks = json.load(open(benchmarks_path))
    print(benchmarks)

    for benchmark, subsets in benchmarks.items():
        if not subsets:
            print(f"Skipping {benchmark} — no subsets")
            continue
        download_benchmark_scores(benchmark, subsets, models)

    display_scores("bbh")


if __name__ == "__main__":
    main()
