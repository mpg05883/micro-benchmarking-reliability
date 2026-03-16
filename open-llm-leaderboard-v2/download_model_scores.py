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
    os.environ["DATASETS_VERBOSITY"] = "error"

    # Programmatically disable progress bars in case env vars are read too late
    from datasets import disable_progress_bars as disable_datasets_bars
    from huggingface_hub.utils import disable_progress_bars as disable_hub_bars

    disable_datasets_bars()
    disable_hub_bars()

    import datasets.utils.logging as datasets_logging

    datasets_logging.set_verbosity_error()


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
            error_str = str(e)

            # Non-transient errors: the dataset is empty, broken, or the
            # config doesn't exist on HuggingFace. Retrying won't help.
            non_transient_errors = [
                "DatasetGenerationError",
                "SchemaInferenceError",
                "Please pass `features`",
                "does not exist",
                "404",
                "BuilderConfig",
                "__leaderboard_",
            ]
            if any(msg in error_str for msg in non_transient_errors):
                print(f"  Skipping {model}/{subset} (no data)")
                return []

            is_rate_limited = "429" in error_str
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
    data_dir: str,
) -> pd.DataFrame | None:
    """Load the benchmark's dataset parquet file as a DataFrame.

    Returns a DataFrame with at least a ``doc_hash`` column, or None if the
    parquet file is missing.

    The ``benchmark`` parameter is the directory name (e.g. ``gpqa_diamond``,
    ``math_hard``, ``mmlu_pro``, ``bbh``, ``musr``).
    """
    data_path = PARENT_DIR / data_dir
    parquet_path = data_path / benchmark / "dataset.parquet"
    if not parquet_path.exists():
        return None
    return pd.read_parquet(parquet_path)


def _verify_model_scores(
    benchmark: str,
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
    dataset_df = _load_dataset_parquet(benchmark, data_dir)
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


def _build_and_save_matrix(
    instance_scores: dict[str, dict[int, float]],
    models: list[str],
    scores_path: Path,
    metadata_path: Path,
    drop_nan_models: bool = False,
) -> int:
    """Build a dense matrix from instance_scores and save to disk.

    If ``drop_nan_models`` is True, columns (models) whose scores are
    entirely NaN are removed before saving. This should only be used on
    the final save — not on checkpoints — because removing columns shifts
    model indices and would break resume logic.

    Returns the number of models in the saved matrix.
    """
    doc_hashes = list(instance_scores.keys())
    shape = (len(doc_hashes), len(models))
    matrix = np.full(shape, np.nan, dtype=np.float32)

    for doc_idx, doc_hash in enumerate(doc_hashes):
        for model_idx, score in instance_scores[doc_hash].items():
            matrix[doc_idx, model_idx] = score

    if drop_nan_models:
        has_scores = ~np.all(np.isnan(matrix), axis=0)
        num_dropped = int(np.sum(~has_scores))
        if num_dropped > 0:
            matrix = matrix[:, has_scores]
            models = [m for m, keep in zip(models, has_scores) if keep]
            print(
                f"Removed {num_dropped} models with all-NaN scores "
                f"({len(models)} models remaining)"
            )

    scores_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp files first, then rename — prevents corruption if
    # the process is interrupted mid-write.
    tmp_scores = scores_path.parent / "model_scores_tmp.npy"
    tmp_metadata = metadata_path.parent / "model_scores_metadata_tmp.json"
    np.save(tmp_scores, matrix)
    with open(tmp_metadata, "w") as f:
        json.dump({"doc_hashes": doc_hashes, "models": models}, f)
    tmp_scores.replace(scores_path)
    tmp_metadata.replace(metadata_path)

    return len(models)


def _save_progress(
    progress_path: Path,
    completed_tasks: set[tuple[int, str]],
) -> None:
    """Save the set of completed (model_idx, subset) pairs to disk."""
    serializable = [
        {"model_idx": model_idx, "subset": subset}
        for model_idx, subset in completed_tasks
    ]
    with open(progress_path, "w") as f:
        json.dump(serializable, f)


def _load_progress(
    progress_path: Path,
) -> set[tuple[int, str]]:
    """Load previously completed (model_idx, subset) pairs from disk."""
    if not progress_path.exists():
        return set()

    with open(progress_path, "r") as f:
        entries = json.load(f)

    return {(entry["model_idx"], entry["subset"]) for entry in entries}


def _load_existing_scores(
    scores_path: Path,
    metadata_path: Path,
    models: list[str],
) -> tuple[dict[str, dict[int, float]], set[int], set[int]]:
    """Load existing scores and determine which models need (re-)downloading.

    A model is considered "completed" only if it has at least one non-NaN
    score. Models whose entire column is NaN are returned in
    ``nan_model_indices`` so they can be retried even if the progress file
    previously marked them as done.

    Returns:
        instance_scores: dict mapping doc_hash → {model_idx: score}
        completed_model_indices: models with at least one non-NaN score
        nan_model_indices: models present in saved data but entirely NaN
    """
    if not scores_path.exists() or not metadata_path.exists():
        return {}, set(), set()

    try:
        matrix = np.load(scores_path)
    except ValueError as e:
        print(f"Warning: corrupted scores file ({e}). Starting fresh.")
        scores_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return {}, set(), set()
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    doc_hashes = metadata["doc_hashes"]
    saved_models = metadata["models"]

    # Build a mapping from saved model names to their column indices
    # in the saved matrix, then map to indices in the current models list
    current_model_to_idx = {m: i for i, m in enumerate(models)}
    saved_model_to_col = {m: i for i, m in enumerate(saved_models)}

    instance_scores: dict[str, dict[int, float]] = {}
    completed_model_indices: set[int] = set()
    nan_model_indices: set[int] = set()

    # Classify each model as completed (has scores) or all-NaN (needs retry)
    for model in models:
        current_idx = current_model_to_idx[model]
        saved_col = saved_model_to_col.get(model)
        if saved_col is None:
            continue

        column = matrix[:, saved_col]
        if np.any(~np.isnan(column)):
            completed_model_indices.add(current_idx)
        else:
            nan_model_indices.add(current_idx)

    # Reconstruct instance_scores, keeping only non-NaN values
    for doc_idx, doc_hash in enumerate(doc_hashes):
        scores = {}
        for model in models:
            current_idx = current_model_to_idx[model]
            saved_col = saved_model_to_col.get(model)
            if saved_col is None:
                continue
            value = float(matrix[doc_idx, saved_col])
            if not np.isnan(value):
                scores[current_idx] = value
        if scores:
            instance_scores[doc_hash] = scores

    return instance_scores, completed_model_indices, nan_model_indices


def _clear_cache(cache_dir: Path) -> None:
    """Clear the cache directory, ignoring errors from files still in use.

    On Windows, concurrent threads may hold open file handles, preventing
    deletion of individual files or the directory itself. This deletes what
    it can and silently skips the rest.
    """
    if not cache_dir.exists():
        return

    shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


def download_model_scores(
    benchmark: str,
    subsets: list[str],
    models: list[str],
    hf_benchmark: str | None = None,
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
    save_interval: int = 50,
) -> None:
    """Download and save all model scores for a single benchmark.

    ``benchmark`` is the directory name for storing results (e.g.
    ``gpqa_diamond``).  ``hf_benchmark`` is the benchmark name used in
    HuggingFace config strings (e.g. ``gpqa``).  If ``hf_benchmark`` is
    None it defaults to ``benchmark``.

    Supports resuming: tracks completed (model_idx, subset) pairs in a progress
    file and reloads existing scores from the npy/metadata files on restart.
    """
    if hf_benchmark is None:
        hf_benchmark = benchmark
    benchmark_dir = PARENT_DIR / data_dir / benchmark
    scores_path = benchmark_dir / scores_file_name
    metadata_path = benchmark_dir / metadata_file_name
    progress_path = benchmark_dir / "progress.json"

    # Load valid doc_hashes from the dataset parquet so we can filter out
    # stale or mismatched hashes returned by HuggingFace model detail repos.
    # Different models may have been evaluated against different versions of
    # the dataset, producing doc_hashes that no longer exist in the current
    # dataset snapshot.
    dataset_df = _load_dataset_parquet(benchmark, data_dir)
    valid_doc_hashes: set[str] | None = None
    if dataset_df is not None:
        valid_doc_hashes = set(dataset_df["doc_hash"].unique())
        print(
            f"{benchmark}: {len(valid_doc_hashes)} valid doc_hashes "
            f"from dataset parquet"
        )

    # Check if the download is already fully complete and verified
    if scores_path.exists() and metadata_path.exists() and not progress_path.exists():
        is_valid = _verify_model_scores(
            benchmark=benchmark,
            models=models,
            scores_path=scores_path,
            metadata_path=metadata_path,
            data_dir=data_dir,
        )
        if is_valid:
            return

    # Load any previously completed work
    completed_tasks = _load_progress(progress_path)
    instance_scores, completed_model_indices, nan_model_indices = _load_existing_scores(
        scores_path, metadata_path, models
    )

    # Filter out stale doc_hashes from any previous run
    if valid_doc_hashes is not None:
        stale_hashes = set(instance_scores.keys()) - valid_doc_hashes
        if stale_hashes:
            for h in stale_hashes:
                del instance_scores[h]
            print(
                f"Removed {len(stale_hashes)} stale doc_hashes "
                f"from existing {benchmark} scores"
            )

    # Models with actual scores don't need re-downloading.
    for model_idx in completed_model_indices:
        for subset in subsets:
            completed_tasks.add((model_idx, subset))

    # Models whose entire column is NaN need to be retried, even if the
    # progress file previously marked them as done.
    for model_idx in nan_model_indices:
        for subset in subsets:
            completed_tasks.discard((model_idx, subset))

    if nan_model_indices:
        print(
            f"Retrying {len(nan_model_indices)} models with all-NaN scores "
            f"in {benchmark}"
        )

    if completed_tasks:
        num_models_done = len({idx for idx, _ in completed_tasks})
        print(
            f"Resuming {benchmark} — {num_models_done}/{len(models)} models "
            f"already completed ({len(completed_tasks)} tasks)"
        )

    # Determine which (model_idx, subset) pairs still need downloading
    all_tasks = [
        (model_idx, model, subset)
        for model_idx, model in enumerate(models)
        for subset in subsets
        if (model_idx, subset) not in completed_tasks
    ]

    if not all_tasks:
        print(f"All tasks for {benchmark} already completed, finalizing...")
        _build_and_save_matrix(instance_scores, models, scores_path, metadata_path)
        progress_path.unlink(missing_ok=True)
        return

    cache_dir = PARENT_DIR / cache_dir
    _clear_cache(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Submit remaining tasks concurrently
    futures = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for model_idx, model, subset in all_tasks:
            future = executor.submit(
                download_model_subset,
                model=model,
                model_idx=model_idx,
                benchmark=hf_benchmark,
                subset=subset,
                cache_dir=cache_dir,
                split=split,
                owner=owner,
                initial_backoff=initial_backoff,
                max_retries=max_retries,
            )
            futures[future] = (model_idx, model, subset)

        # Collect results, periodically saving checkpoints and clearing cache.
        total_remaining = len(futures)
        completed_since_save = 0
        completed_since_clear = 0

        kwargs = {
            "desc": f"Downloading {benchmark} model scores",
            "unit": "task",
            "total": total_remaining,
        }

        with tqdm(**kwargs) as pbar:
            for future in as_completed(futures):
                model_idx, model, subset = futures[future]

                for result in future.result():
                    # Skip doc_hashes not in the current dataset parquet
                    if (
                        valid_doc_hashes is not None
                        and result.doc_hash not in valid_doc_hashes
                    ):
                        continue
                    scores = instance_scores.setdefault(result.doc_hash, {})
                    scores[result.model_idx] = result.score

                completed_tasks.add((model_idx, subset))
                completed_since_save += 1
                completed_since_clear += 1
                pbar.update(1)

                # Periodically save checkpoint
                if completed_since_save >= save_interval:
                    _build_and_save_matrix(
                        instance_scores, models, scores_path, metadata_path
                    )
                    _save_progress(progress_path, completed_tasks)
                    completed_since_save = 0

                # Periodically clear cache
                if completed_since_clear >= cache_clear_interval:
                    _clear_cache(cache_dir)
                    completed_since_clear = 0

    # Final save — drop models with no scores
    num_models_saved = _build_and_save_matrix(
        instance_scores, models, scores_path, metadata_path, drop_nan_models=True
    )

    # Remove progress file to indicate download is complete
    progress_path.unlink(missing_ok=True)

    print(
        f"Saved {len(instance_scores)} instances × {num_models_saved} models "
        f"to {scores_path} and {metadata_path}"
    )

    # Final cache cleanup
    _clear_cache(cache_dir)


def main(args: argparse.Namespace) -> None:
    # Redirect the huggingface_hub download cache so raw files don't
    # accumulate in the default ~/.cache/huggingface/hub directory.
    os.environ["HF_HUB_CACHE"] = str(PARENT_DIR / args.cache_dir / "hub")

    if not args.verbose:
        suppress_huggingface()

    with open(PARENT_DIR / args.configs_dir / args.benchmarks_file, "r") as f:
        benchmarks = json.load(f)

    with open(PARENT_DIR / args.configs_dir / args.models_file, "r") as f:
        models = json.load(f)

    # Expand benchmarks so each gets its own directory and scores matrix.
    # GPQA subsets are split into separate entries (gpqa_diamond, gpqa_extended,
    # gpqa_main) since each has its own dataset parquet. Other benchmarks with
    # directory names that differ from their HF config name (e.g. math → math_hard)
    # are also remapped here.
    dir_name_overrides = {"math": "math_hard", "mmlu": "mmlu_pro"}
    expanded: list[tuple[str, str, list[str]]] = []  # (dir_name, hf_benchmark, subsets)

    for benchmark, subsets in benchmarks.items():
        if not subsets:
            print(f"Skipping {benchmark} — no subsets")
            continue
        if benchmark == "gpqa":
            for subset in subsets:
                if subset in ("main", "extended"):
                    print(f"Skipping gpqa_{subset}")
                    continue
                expanded.append((f"gpqa_{subset}", "gpqa", [subset]))
        else:
            dir_name = dir_name_overrides.get(benchmark, benchmark)
            expanded.append((dir_name, benchmark, subsets))

    for dir_name, hf_benchmark, subsets in expanded:
        download_model_scores(
            dir_name,
            subsets,
            models,
            hf_benchmark=hf_benchmark,
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
            save_interval=args.save_interval,
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
    parser.add_argument("--save_interval", type=int, default=50)
    parser.add_argument("--owner", type=str, default="open-llm-leaderboard")
    parser.add_argument("--split", type=str, default="latest")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--initial_backoff", type=int, default=2)
    args = parser.parse_args()
    main(args)
