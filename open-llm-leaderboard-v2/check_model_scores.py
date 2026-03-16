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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

PARENT_DIR = Path(__file__).resolve().parent

CORRECTNESS_COLUMNS = ["acc", "acc_norm", "exact_match"]

NON_TRANSIENT_ERRORS = [
    "DatasetGenerationError",
    "SchemaInferenceError",
    "Please pass `features`",
    "does not exist",
    "404",
    "BuilderConfig",
    "__leaderboard_",
]


@dataclass
class Mismatch:
    """A single score mismatch between local and HuggingFace data."""

    benchmark: str
    subset: str
    model: str
    doc_hash: str
    local_score: float | str
    hf_score: float | str
    issue: str


@dataclass
class BenchmarkResult:
    """Verification results for a single benchmark."""

    checked: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)


def suppress_huggingface() -> None:
    """Suppress Hugging Face warnings and progress bars."""
    warnings.filterwarnings(
        "ignore",
        message=".*huggingface_hub.*cache-system uses symlinks.*",
    )

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["DATASETS_VERBOSITY"] = "error"

    from datasets import disable_progress_bars as disable_datasets_bars
    from huggingface_hub.utils import disable_progress_bars as disable_hub_bars

    disable_datasets_bars()
    disable_hub_bars()

    import datasets.utils.logging as datasets_logging

    datasets_logging.set_verbosity_error()


def load_hf_scores(
    model: str,
    hf_benchmark: str,
    subset: str,
    cache_dir: Path,
    owner: str,
    split: str,
    initial_backoff: int,
    max_retries: int,
) -> dict[str, float]:
    """Download scores for a single model/subset from HuggingFace.

    ``hf_benchmark`` is the benchmark name used in HuggingFace config strings
    (e.g. ``gpqa``, ``bbh``), not the directory name.

    Returns a dict mapping doc_hash to score, or an empty dict on failure.
    """
    hf_model = model.replace("/", "__")
    repo_id = f"{owner}/{hf_model}-details"
    config = f"{hf_model}__leaderboard_{hf_benchmark}_{subset}"

    attempt = 0
    while True:
        try:
            df = load_dataset(
                repo_id, config, split=split, cache_dir=cache_dir
            ).to_pandas()
            break
        except Exception as e:
            error_str = str(e)

            if any(msg in error_str for msg in NON_TRANSIENT_ERRORS):
                return {}

            if "429" in error_str or attempt < max_retries:
                wait = initial_backoff * (2 ** min(attempt, 6))
                reason = "Rate limited" if "429" in error_str else "Error"
                print(
                    f"  {reason} on {model}/{subset}, "
                    f"retrying in {wait}s (attempt {attempt + 1})..."
                )
                time.sleep(wait)
                attempt += 1
                continue
            raise

    column = next((c for c in CORRECTNESS_COLUMNS if c in df.columns), None)
    if column is None:
        return {}

    return {
        row["doc_hash"]: float(row[column])
        for _, row in df[["doc_hash", column]].iterrows()
    }


def _compare_scores(
    local_score: float,
    hf_score: float,
) -> str | None:
    """Compare two scores with NaN awareness.

    Returns the issue description, or None if the scores match.
    """
    if math.isnan(local_score) and math.isnan(hf_score):
        return None
    if math.isnan(local_score) or math.isnan(hf_score):
        return "NaN mismatch"
    if local_score != hf_score:
        return "Score mismatch"
    return None


def verify_benchmark(
    benchmark: str,
    hf_benchmark: str,
    subsets: list[str],
    selected_models: list[str],
    matrix: np.ndarray,
    doc_hashes: list[str],
    model_names: list[str],
    cache_dir: Path,
    owner: str,
    split: str,
    initial_backoff: int,
    max_retries: int,
    cache_clear_interval: int = 50,
) -> BenchmarkResult:
    """Verify scores for a single benchmark against HuggingFace.

    ``benchmark`` is the directory name (e.g. ``gpqa_diamond``).
    ``hf_benchmark`` is the HuggingFace benchmark name (e.g. ``gpqa``).
    """
    hash_to_row = {h: i for i, h in enumerate(doc_hashes)}
    model_to_col = {m: i for i, m in enumerate(model_names)}
    result = BenchmarkResult()
    downloads_since_clear = 0

    for model in tqdm(selected_models, desc=f"  {benchmark}", unit="model"):
        col_idx = model_to_col.get(model)
        if col_idx is None:
            print(f"    Model {model} not in {benchmark} metadata, skipping")
            continue

        for subset in subsets:
            hf_scores = load_hf_scores(
                model, hf_benchmark, subset, cache_dir,
                owner, split, initial_backoff, max_retries,
            )
            downloads_since_clear += 1

            if not hf_scores:
                result.mismatches.append(Mismatch(
                    benchmark=benchmark,
                    subset=subset,
                    model=model,
                    doc_hash="N/A",
                    local_score="N/A",
                    hf_score="N/A",
                    issue="Download failed or no data on HuggingFace",
                ))
                continue

            for doc_hash, hf_score in hf_scores.items():
                row_idx = hash_to_row.get(doc_hash)
                if row_idx is None:
                    # Doc hash from HF not in local data — expected for
                    # models evaluated on older dataset versions
                    continue

                local_score = float(matrix[row_idx, col_idx])
                result.checked += 1

                issue = _compare_scores(local_score, hf_score)
                if issue is not None:
                    result.mismatches.append(Mismatch(
                        benchmark=benchmark,
                        subset=subset,
                        model=model,
                        doc_hash=doc_hash,
                        local_score=local_score,
                        hf_score=hf_score,
                        issue=issue,
                    ))

            # Periodically clear cache to free disk space
            if downloads_since_clear >= cache_clear_interval:
                _clear_cache(cache_dir)
                downloads_since_clear = 0

    return result


def write_log(
    log_path: Path,
    k: int,
    seed: int | None,
    benchmark_results: dict[str, BenchmarkResult],
    total_checked: int,
    total_mismatches: int,
) -> None:
    """Write verification results to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("Model Scores Verification Log\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Models per benchmark: {k}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Benchmarks verified: {', '.join(benchmark_results.keys())}\n")
        f.write("\n")

        for benchmark, result in benchmark_results.items():
            f.write("-" * 70 + "\n")
            f.write(f"Benchmark: {benchmark}\n")
            f.write(f"  Instances checked: {result.checked}\n")
            f.write(f"  Mismatches: {len(result.mismatches)}\n")

            if result.mismatches:
                f.write("\n  Details:\n")
                for m in result.mismatches:
                    doc_hash_str = (
                        m.doc_hash[:16] + "..."
                        if len(m.doc_hash) > 16
                        else m.doc_hash
                    )
                    f.write(
                        f"    [{m.subset}] {m.model} | "
                        f"doc_hash={doc_hash_str} | "
                        f"local={m.local_score} | hf={m.hf_score} | "
                        f"{m.issue}\n"
                    )
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write(f"TOTAL instances checked: {total_checked}\n")
        f.write(f"TOTAL mismatches: {total_mismatches}\n")
        f.write(f"Result: {'PASS' if total_mismatches == 0 else 'FAIL'}\n")
        f.write("=" * 70 + "\n")


def _clear_cache(cache_dir: Path) -> None:
    """Remove and recreate the cache directory."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _select_models(
    dir_names: list[str],
    data_dir: Path,
    metadata_file_name: str,
    k: int | None,
    seed: int | None,
) -> list[str]:
    """Select models to verify.

    If ``k`` is None, all models common to all benchmarks are returned.
    Otherwise, ``k`` random models are sampled.
    """
    models_per_benchmark: list[set[str]] = []
    for dir_name in dir_names:
        metadata_path = data_dir / dir_name / metadata_file_name
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            models_per_benchmark.append(set(metadata["models"]))

    if not models_per_benchmark:
        return []

    common_models = sorted(set.intersection(*models_per_benchmark))
    print(f"Models common to all benchmarks: {len(common_models)}")

    if not common_models:
        return []

    if k is None:
        print(f"Verifying all {len(common_models)} models")
        return common_models

    rng = random.Random(seed)
    k = min(k, len(common_models))
    selected = rng.sample(common_models, k)

    print(f"Selected {k} models to verify:")
    for model in selected:
        print(f"  {model}")

    return selected


def main(args: argparse.Namespace) -> None:
    if not args.verbose:
        suppress_huggingface()

    with open(PARENT_DIR / args.configs_dir / args.benchmarks_file, "r") as f:
        all_benchmarks: dict[str, list[str]] = json.load(f)

    # Expand benchmarks so each directory has its own entry.
    # GPQA subsets become separate entries (gpqa_diamond, gpqa_extended, etc.).
    # Other benchmarks with directory names differing from their HF config name
    # (e.g. math → math_hard) are remapped.
    dir_name_overrides = {"math": "math_hard", "mmlu": "mmlu_pro"}
    expanded: list[tuple[str, str, list[str]]] = []  # (dir_name, hf_benchmark, subsets)

    for benchmark, subsets in all_benchmarks.items():
        if not subsets:
            continue
        if benchmark == "gpqa":
            continue
            # for subset in subsets:
            #     expanded.append((f"gpqa_{subset}", "gpqa", [subset]))
        else:
            dir_name = dir_name_overrides.get(benchmark, benchmark)
            expanded.append((dir_name, benchmark, subsets))

    data_dir = PARENT_DIR / args.data_dir
    dir_names = [dir_name for dir_name, _, _ in expanded]
    k = None if args.all else args.k
    selected_models = _select_models(
        dir_names, data_dir, args.metadata_file_name, k, args.seed,
    )
    if not selected_models:
        print("No models to verify.")
        return

    cache_dir = PARENT_DIR / args.cache_dir
    _clear_cache(cache_dir)

    benchmark_results: dict[str, BenchmarkResult] = {}
    total_checked = 0
    total_mismatches = 0

    for dir_name, hf_benchmark, subsets in expanded:
        scores_path = data_dir / dir_name / args.scores_file_name
        metadata_path = data_dir / dir_name / args.metadata_file_name

        if not scores_path.exists() or not metadata_path.exists():
            print(f"\nSkipping {dir_name} — missing data files")
            continue

        print(f"\nVerifying {dir_name} ({len(subsets)} subsets)...")

        matrix = np.load(scores_path)
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        result = verify_benchmark(
            benchmark=dir_name,
            hf_benchmark=hf_benchmark,
            subsets=subsets,
            selected_models=selected_models,
            matrix=matrix,
            doc_hashes=metadata["doc_hashes"],
            model_names=metadata["models"],
            cache_dir=cache_dir,
            owner=args.owner,
            split=args.split,
            initial_backoff=args.initial_backoff,
            max_retries=args.max_retries,
            cache_clear_interval=args.cache_clear_interval,
        )

        benchmark_results[dir_name] = result
        total_checked += result.checked
        total_mismatches += len(result.mismatches)

        print(f"  Checked: {result.checked}, Mismatches: {len(result.mismatches)}")

        _clear_cache(cache_dir)

    # Final cache cleanup
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    # Write log
    log_path = PARENT_DIR / "logs" / "check_model_scores.log"
    num_models = len(selected_models)
    write_log(
        log_path, num_models, args.seed,
        benchmark_results, total_checked, total_mismatches,
    )
    print(f"\nLog written to {log_path}")
    print(f"\nTotal instances checked: {total_checked}")
    print(f"Total mismatches: {total_mismatches}")
    print(f"Result: {'PASS' if total_mismatches == 0 else 'FAIL'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify local model scores against HuggingFace"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of random models to verify per benchmark (default: 5)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Verify all models instead of a random sample of k",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--cache_clear_interval", type=int, default=50,
        help="Clear cache every N downloads (default: 50)",
    )
    parser.add_argument("--configs_dir", type=str, default="configs")
    parser.add_argument("--benchmarks_file", type=str, default="benchmarks.json")
    parser.add_argument("--metadata_file_name", type=str, default="model_scores_metadata.json")
    parser.add_argument("--scores_file_name", type=str, default="model_scores.npy")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--owner", type=str, default="open-llm-leaderboard")
    parser.add_argument("--split", type=str, default="latest")
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--initial_backoff", type=int, default=2)
    args = parser.parse_args()
    main(args)
