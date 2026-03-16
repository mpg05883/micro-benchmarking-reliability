"""
Verify local model scores against HuggingFace.

For each benchmark, loads the local .npy score matrix and metadata,
selects k random models, re-downloads their results from HuggingFace,
and compares per-instance scores. Writes a log summarizing any differences.
"""

import argparse
import json
import math
import os
import random
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent

OWNER = "open-llm-leaderboard"
SPLIT = "latest"
CORRECTNESS_COLS = ["acc", "acc_norm", "exact_match"]
SKIP_BENCHMARKS = {"ifeval", "mmlu"}
MAX_RETRIES = 5
INITIAL_BACKOFF = 2


def setup():
    warnings.filterwarnings(
        "ignore",
        message=".*huggingface_hub.*cache-system uses symlinks.*",
    )
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"


def load_hf_dataset(repo_id, config, split, cache_dir):
    """Load a HuggingFace dataset with retry and exponential backoff."""
    attempt = 0
    while True:
        try:
            return load_dataset(
                repo_id, config, split=split, cache_dir=cache_dir
            ).to_pandas()
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES:
                wait = INITIAL_BACKOFF * (2 ** min(attempt, 6))
                print(f"  Rate limited, retrying in {wait}s...")
                time.sleep(wait)
                attempt += 1
                continue
            raise


def verify_benchmark(benchmark, subsets, selected_models, model_indices, cache_dir):
    """Verify scores for a single benchmark. Returns (checked, mismatches_list)."""
    data_dir = BASE_DIR / "data" / benchmark
    scores_path = data_dir / "model_scores.npy"
    meta_path = data_dir / "model_scores_metadata.json"

    if not scores_path.exists() or not meta_path.exists():
        print(f"  Skipping {benchmark} — missing data files")
        return 0, []

    matrix = np.load(scores_path)
    with open(meta_path) as f:
        meta = json.load(f)

    doc_hashes = meta["doc_hashes"]
    models_in_meta = meta["models"]

    # Build doc_hash → row index lookup
    hash_to_row = {h: i for i, h in enumerate(doc_hashes)}

    # Build model name → column index lookup
    model_to_col = {m: i for i, m in enumerate(models_in_meta)}

    checked = 0
    mismatches = []

    for model in tqdm(selected_models, desc=f"  {benchmark}", unit="model"):
        col_idx = model_to_col.get(model)
        if col_idx is None:
            print(f"    Model {model} not in {benchmark} metadata, skipping")
            continue

        for subset in subsets:
            repo_id = f"{OWNER}/{model}-details"
            config = f"{model}__leaderboard_{benchmark}_{subset}"

            try:
                df = load_hf_dataset(repo_id, config, SPLIT, cache_dir)
            except Exception as e:
                print(f"    Failed to load {model}/{subset}: {e}")
                mismatches.append({
                    "benchmark": benchmark,
                    "subset": subset,
                    "model": model,
                    "doc_hash": "N/A",
                    "local_score": "N/A",
                    "hf_score": "N/A",
                    "issue": f"Download failed: {e}",
                })
                continue

            col = next((c for c in CORRECTNESS_COLS if c in df.columns), None)
            if col is None:
                print(f"    No correctness column in {model}/{subset}")
                continue

            for _, row in df[["doc_hash", col]].iterrows():
                doc_hash = row["doc_hash"]
                hf_score = float(row[col])

                row_idx = hash_to_row.get(doc_hash)
                if row_idx is None:
                    mismatches.append({
                        "benchmark": benchmark,
                        "subset": subset,
                        "model": model,
                        "doc_hash": doc_hash,
                        "local_score": "MISSING",
                        "hf_score": hf_score,
                        "issue": "doc_hash not found in local data",
                    })
                    checked += 1
                    continue

                local_score = float(matrix[row_idx, col_idx])
                checked += 1

                # NaN-aware comparison
                if math.isnan(local_score) and math.isnan(hf_score):
                    continue
                if math.isnan(local_score) or math.isnan(hf_score):
                    mismatches.append({
                        "benchmark": benchmark,
                        "subset": subset,
                        "model": model,
                        "doc_hash": doc_hash,
                        "local_score": local_score,
                        "hf_score": hf_score,
                        "issue": "NaN mismatch",
                    })
                elif local_score != hf_score:
                    mismatches.append({
                        "benchmark": benchmark,
                        "subset": subset,
                        "model": model,
                        "doc_hash": doc_hash,
                        "local_score": local_score,
                        "hf_score": hf_score,
                        "issue": "Score mismatch",
                    })

    return checked, mismatches


def write_log(log_path, k, seed, benchmark_results, total_checked, total_mismatches):
    """Write verification results to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Model Scores Verification Log\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"k (models per benchmark): {k}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Benchmarks verified: {', '.join(benchmark_results.keys())}\n")
        f.write("\n")

        for benchmark, (checked, mismatches) in benchmark_results.items():
            f.write("-" * 70 + "\n")
            f.write(f"Benchmark: {benchmark}\n")
            f.write(f"  Instances checked: {checked}\n")
            f.write(f"  Mismatches: {len(mismatches)}\n")

            if mismatches:
                f.write("\n  Details:\n")
                for m in mismatches:
                    f.write(
                        f"    [{m['subset']}] {m['model']} | "
                        f"doc_hash={m['doc_hash'][:16]}... | "
                        f"local={m['local_score']} | hf={m['hf_score']} | "
                        f"{m['issue']}\n"
                    )
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write(f"TOTAL instances checked: {total_checked}\n")
        f.write(f"TOTAL mismatches: {total_mismatches}\n")
        f.write(f"Result: {'PASS' if total_mismatches == 0 else 'FAIL'}\n")
        f.write("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify local model scores against HuggingFace"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of random models to verify (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    setup()

    # Load benchmarks and models
    with open(BASE_DIR / "resources" / "benchmarks.json") as f:
        benchmarks = json.load(f)

    # Determine which models are available across all active benchmarks
    # by intersecting the models lists from each benchmark's metadata
    active_benchmarks = {
        b: subsets for b, subsets in benchmarks.items()
        if subsets and b not in SKIP_BENCHMARKS
    }

    models_per_benchmark = {}
    for benchmark in active_benchmarks:
        meta_path = BASE_DIR / "data" / benchmark / "model_scores_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            models_per_benchmark[benchmark] = set(meta["models"])

    # Use models present in all benchmarks for consistent sampling
    common_models = sorted(set.intersection(*models_per_benchmark.values()))
    print(f"Models common to all benchmarks: {len(common_models)}")

    # Select k random models
    rng = random.Random(args.seed)
    k = min(args.k, len(common_models))
    selected_models = rng.sample(common_models, k)

    print(f"Selected {k} models to verify:")
    for model in selected_models:
        print(f"  {model}")

    # Verify each benchmark
    cache_dir = BASE_DIR / "cache"
    benchmark_results = {}
    total_checked = 0
    total_mismatches = 0

    for benchmark, subsets in active_benchmarks.items():
        print(f"\nVerifying {benchmark} ({len(subsets)} subsets)...")

        cache_dir.mkdir(exist_ok=True)

        checked, mismatches = verify_benchmark(
            benchmark, subsets, selected_models, None, cache_dir
        )

        benchmark_results[benchmark] = (checked, mismatches)
        total_checked += checked
        total_mismatches += len(mismatches)

        print(f"  Checked: {checked}, Mismatches: {len(mismatches)}")

        # Clear cache after each benchmark
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print(f"  Cleared cache at {cache_dir}")

    # Write log file
    log_path = BASE_DIR / "logs" / "verify_scores.log"
    write_log(log_path, k, args.seed, benchmark_results, total_checked, total_mismatches)
    print(f"\nLog written to {log_path}")

    # Print summary
    print(f"\nTotal instances checked: {total_checked}")
    print(f"Total mismatches: {total_mismatches}")
    print(f"Result: {'PASS' if total_mismatches == 0 else 'FAIL'}")


if __name__ == "__main__":
    main()
