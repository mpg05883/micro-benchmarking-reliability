import argparse
import json
import os
import shutil
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

PARENT_DIR = Path(__file__).resolve().parent


@dataclass
class Score:
    doc_hash: str
    model_idx: int
    score: float


def suppress_huggingface() -> None:
    """Suppress Hugging Face warnings and progress bars."""
    warnings.filterwarnings(
        "ignore",
        message=".*huggingface_hub.*cache-system uses symlinks.*",
    )

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"


def download_model_subset(
    model: str,
    model_idx: int,
    benchmark: str,
    subset: str,
    cache_dir: Path,
    split: str = "latest",
    owner: str = "open-llm-leaderboard",
    initial_backoff: int = 2,
    max_retries: int = 5,
) -> list[Score]:
    """Download scores for a single model/subset combination."""
    # HuggingFace detail repos use "__" instead of "/" in model names
    hf_model = model.replace("/", "__")
    repo_id = f"{owner}/{hf_model}-details"
    config = f"{hf_model}__leaderboard_{benchmark}_{subset}"

    attempt = 0
    while True:
        try:
            df = load_dataset(
                repo_id,
                config,
                split=split,
                cache_dir=cache_dir,
            ).to_pandas()
            break
        except Exception as e:
            is_rate_limited = "429" in str(e)
            is_retryable = attempt < max_retries

            if is_rate_limited or is_retryable:
                wait_seconds = initial_backoff * (2 ** min(attempt, 6))
                reason = "Rate limited" if is_rate_limited else "Error"
                print(
                    f"  {reason} on {model}/{subset}, "
                    f"retrying in {wait_seconds}s (attempt {attempt + 1})..."
                )
                time.sleep(wait_seconds)
                attempt += 1
                continue
            print(f"  Skipping {model}/{subset} after {attempt} retries: {e}")
            return []

    correctness_columns = ["acc", "acc_norm", "exact_match"]
    column = next((c for c in correctness_columns if c in df.columns), None)
    if column is None:
        print(f"  No correctness column in {model}/{subset}: {list(df.columns)}")
        return []

    # Each element contains the score for a single model/instance combination
    scores = [
        Score(
            doc_hash=row["doc_hash"],
            model_idx=model_idx,
            score=float(row[column]),
        )
        for _, row in df[["doc_hash", column]].iterrows()
    ]

    return scores


def _load_dataset_parquet(
    benchmark: str,
    subsets: list[str],
    data_dir: str,
) -> pd.DataFrame | None:
    """Load the benchmark's dataset parquet file(s) as a DataFrame.

    Returns a DataFrame with at least a ``doc_hash`` column, or None if the
    parquet files are missing. GPQA has separate parquets per subset; others
    are combined into one file.
    """
    data_path = PARENT_DIR / data_dir

    if benchmark == "gpqa":
        dfs = []
        for subset in subsets:
            parquet_path = data_path / f"gpqa_{subset}" / "dataset.parquet"
            if not parquet_path.exists():
                return None
            dfs.append(pd.read_parquet(parquet_path))
        return pd.concat(dfs, ignore_index=True)

    parquet_dir_names = {
        "math": "math_hard",
        "mmlu": "mmlu_pro",
    }

    dir_name = parquet_dir_names.get(benchmark, benchmark)
    parquet_path = data_path / dir_name / "dataset.parquet"
    if not parquet_path.exists():
        return None
    return pd.read_parquet(parquet_path)


def _verify_model_scores(
    benchmark: str,
    subsets: list[str],
    models: list[str],
    scores_path: Path,
    metadata_path: Path,
    data_dir: str,
) -> bool:
    """Verify existing model scores against the dataset parquet.

    Loads the dataset parquet, builds a DataFrame from the saved model scores
    and metadata, and merges on ``doc_hash`` to check for missing scores.

    Returns True if verification passes (no re-download needed), False otherwise.
    """
    matrix = np.load(scores_path)
    num_instances, num_models = matrix.shape

    # Check model count
    if num_models != len(models):
        print(
            f"Re-downloading {benchmark} — model count mismatch "
            f"(saved: {num_models}, expected: {len(models)})"
        )
        return False

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    doc_hashes = metadata["doc_hashes"]
    model_names = metadata["models"]

    # Build model scores DataFrame: one row per doc_hash, one column per model
    scores_df = pd.DataFrame(matrix, columns=model_names)
    scores_df.insert(0, "doc_hash", doc_hashes)

    # Load dataset parquet
    dataset_df = _load_dataset_parquet(benchmark, subsets, data_dir)
    if dataset_df is None:
        print(
            f"Skipping instance verification for {benchmark} — "
            f"no dataset parquet found"
        )
        print(
            f"Skipping {benchmark} — already exists "
            f"({num_instances} instances × {num_models} models)"
        )
        return True

    # Merge on doc_hash to find missing scores
    merged_df = (
        dataset_df[["doc_hash"]]
        .drop_duplicates()
        .merge(
            scores_df,
            on="doc_hash",
            how="left",
            indicator=True,
        )
    )

    missing_scores = merged_df[merged_df["_merge"] == "left_only"]
    extra_scores = scores_df[
        ~scores_df["doc_hash"].isin(dataset_df["doc_hash"].unique())
    ]

    if not missing_scores.empty:
        print(
            f"Re-downloading {benchmark} — {len(missing_scores)} dataset instances "
            f"have no model scores"
        )
        return False

    if not extra_scores.empty:
        print(
            f"Warning: {benchmark} has {len(extra_scores)} model score rows "
            f"with doc_hashes not in the dataset parquet"
        )

    print(
        f"Skipping {benchmark} — already exists "
        f"({num_instances} instances × {num_models} models, "
        f"all {len(merged_df)} dataset instances have scores)"
    )
    return True


def download_model_scores(
    benchmark: str,
    subsets: list[str],
    models: list[str],
    owner: str = "open-llm-leaderboard",
    split: str = "latest",
    max_workers: int = 8,
    initial_backoff: int = 2,
    max_retries: int = 5,
    data_dir: str = "data",
    cache_dir: str = "cache",
    scores_file_name: str = "model_scores.npy",
    metadata_file_name: str = "model_scores_metadata.json",
    cache_clear_interval: int = 100,
) -> None:
    """Download and save all model scores for a single benchmark."""
    scores_path = PARENT_DIR / data_dir / benchmark / scores_file_name
    metadata_path = PARENT_DIR / data_dir / benchmark / metadata_file_name

    if scores_path.exists() and metadata_path.exists():
        is_valid = _verify_model_scores(
            benchmark=benchmark,
            subsets=subsets,
            models=models,
            scores_path=scores_path,
            metadata_path=metadata_path,
            data_dir=data_dir,
        )
        if is_valid:
            return

    cache_dir = PARENT_DIR / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Submit all model × subset tasks concurrently
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for model_idx, model in enumerate(models):
            for subset in subsets:
                future = executor.submit(
                    download_model_subset,
                    model=model,
                    model_idx=model_idx,
                    benchmark=benchmark,
                    subset=subset,
                    cache_dir=cache_dir,
                    split=split,
                    owner=owner,
                    initial_backoff=initial_backoff,
                    max_retries=max_retries,
                )
                futures[future] = (model_idx, model, subset)

        # Collect results, clearing the cache periodically to limit disk usage.
        # Completed tasks already have their data in memory, so deleting cached
        # files is safe. In-flight tasks that lose their cache will redownload.
        instance_scores: dict[str, dict[int, float]] = {}
        total = len(futures)
        completed = 0

        kwargs = {
            "desc": f"Downloading {benchmark} model scores",
            "unit": "task",
            "total": total,
        }

        with tqdm(**kwargs) as pbar:
            for future in as_completed(futures):
                for result in future.result():
                    scores = instance_scores.setdefault(result.doc_hash, {})
                    scores[result.model_idx] = result.score
                completed += 1
                pbar.update(1)

                if completed % cache_clear_interval == 0 and cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build a dense matrix where `NaN` means no result for that model/instance
    doc_hashes = list(instance_scores.keys())
    shape = (len(doc_hashes), len(models))
    matrix = np.full(
        shape,
        np.nan,
        dtype=np.float32,
    )

    for doc_idx, doc_hash in enumerate(doc_hashes):
        for model_idx, score in instance_scores[doc_hash].items():
            matrix[doc_idx, model_idx] = score

    scores_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(scores_path, matrix)
    with open(metadata_path, "w") as f:
        json.dump({"doc_hashes": doc_hashes, "models": models}, f)

    print(
        f"Saved {len(doc_hashes)} instances × {len(models)} models "
        f"to {scores_path} and {metadata_path}"
    )

    # Final cache cleanup for any remaining files
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        print(f"Cleared cache at {cache_dir}")


def main(args: argparse.Namespace) -> None:
    if not args.verbose:
        suppress_huggingface()

    with open(PARENT_DIR / args.configs_dir / args.benchmarks_file, "r") as f:
        benchmarks = json.load(f)

    with open(PARENT_DIR / args.configs_dir / args.models_file, "r") as f:
        models = json.load(f)

    # Only use gpqa_diamond for testing
    benchmarks = {"gpqa": ["diamond"]}

    for benchmark, subsets in benchmarks.items():
        if not subsets:
            print(f"Skipping {benchmark} — no subsets")
            continue
        print(f"Downloading model scores for {benchmark}")
        download_model_scores(
            benchmark,
            subsets,
            models,
            owner=args.owner,
            split=args.split,
            max_workers=args.max_workers,
            initial_backoff=args.initial_backoff,
            max_retries=args.max_retries,
            data_dir=args.data_dir,
            cache_dir=args.cache_dir,
            scores_file_name=args.scores_file_name,
            metadata_file_name=args.metadata_file_name,
            cache_clear_interval=args.cache_clear_interval,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--configs_dir", type=str, default="configs")
    parser.add_argument("--benchmarks_file", type=str, default="benchmarks.json")
    parser.add_argument("--models_file", type=str, default="models.json")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--scores_file_name", type=str, default="model_scores.npy")
    parser.add_argument(
        "--metadata_file_name", type=str, default="model_scores_metadata.json"
    )
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--cache_clear_interval", type=int, default=100)
    parser.add_argument("--owner", type=str, default="open-llm-leaderboard")
    parser.add_argument("--split", type=str, default="latest")
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--initial_backoff", type=int, default=2)
    args = parser.parse_args()
    main(args)
