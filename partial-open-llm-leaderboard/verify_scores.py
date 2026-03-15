#!/usr/bin/env python3
"""
Verify that local per-instance model scores in partial-open-llm-leaderboard/data/
match the authoritative scores on the Hugging Face Open LLM Leaderboard.

For each benchmark a random sample of models is chosen, and their per-instance
scores are compared against the HuggingFace source row-by-row.

Supported benchmarks (v2 leaderboard):
  BBH, GPQA, MMLU-Pro

Skipped (v1 leaderboard — different format):
  MMLU

Usage:
    python verify_scores.py [--n-models N] [--seed S] [--benchmarks bbh gpqa mmlu-pro mmlu]
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing from utils/ in this directory
sys.path.insert(0, str(Path(__file__).parent))

from utils.enums import Benchmark
from utils.path import resolve_model_scores_path

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"

from datasets import load_dataset

# ── Constants ─────────────────────────────────────────────────────────────────

SPLIT = "latest"
CORRECTNESS_COLS = ["acc", "acc_norm", "exact_match"]
HF_OWNER = "open-llm-leaderboard"

# Columns that are metadata, not model scores
META_COLS = {"subtask_range", "subset", "scores_subset"}

# How each benchmark maps to HuggingFace configs.
#
# per_subset=True  → one HF config per local subset, using the subset name with
#                    strip_prefix removed to form: {model}__leaderboard_{hf_bm}_{suffix}
# per_subset=False → one HF config covers the whole benchmark:
#                    {model}__leaderboard_{hf_bm}_{hf_suffix}
BENCHMARK_HF = {
    Benchmark.BBH: {
        "per_subset": True,
        "hf_benchmark": "bbh",
        "strip_prefix": "bbh_",
    },
    Benchmark.GPQA: {
        "per_subset": True,
        "hf_benchmark": "gpqa",
        "strip_prefix": "gpqa_",
    },
    Benchmark.MMLU_PRO: {
        "per_subset": False,
        "hf_benchmark": "mmlu",
        "hf_suffix": "pro",
    },
    Benchmark.MMLU: None,  # v1 leaderboard — skipped
}

# ── Helpers ───────────────────────────────────────────────────────────────────


def get_model_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def get_subset_col(df: pd.DataFrame) -> str:
    """Return the name of the subset column (MMLU-Pro uses 'scores_subset')."""
    return "scores_subset" if "scores_subset" in df.columns else "subset"


def load_hf_config(model: str, config: str) -> pd.Series | None:
    """
    Load a HuggingFace dataset config and return the correctness column as a
    Series (reset index). Returns None on any error.
    """
    repo_id = f"{HF_OWNER}/{model}-details"
    try:
        hf_df = load_dataset(repo_id, config, split=SPLIT).to_pandas()
    except Exception as exc:
        print(f"      [WARN] Cannot load {repo_id} / {config}: {exc}")
        return None

    col = next((c for c in CORRECTNESS_COLS if c in hf_df.columns), None)
    if col is None:
        print(
            f"      [WARN] No correctness column found in {config}. "
            f"Available: {list(hf_df.columns)}"
        )
        return None

    return hf_df[col].reset_index(drop=True)


def compare_scores(local: pd.Series, hf: pd.Series) -> dict:
    """
    Element-wise comparison between local and HF scores.
    Returns a result dict with match info and (capped) differing indices.
    """
    local_n = len(local)
    hf_n = len(hf)

    if local_n != hf_n:
        return {
            "size_mismatch": True,
            "local_n": local_n,
            "hf_n": hf_n,
            "n_diff": local_n,  # treat all as differing for summary counts
            "match_rate": 0.0,
            "diff_indices": [],
            "local_vals": [],
            "hf_vals": [],
        }

    local_arr = local.values.astype(float)
    hf_arr = hf.values.astype(float)
    diff_mask = local_arr != hf_arr
    diff_indices = list(np.where(diff_mask)[0])
    n_diff = len(diff_indices)

    # Cap displayed indices at 20
    cap = 20
    shown_idx = diff_indices[:cap]

    return {
        "size_mismatch": False,
        "local_n": local_n,
        "hf_n": hf_n,
        "n_diff": n_diff,
        "match_rate": 1.0 - n_diff / local_n if local_n else 1.0,
        "diff_indices": shown_idx,
        "local_vals": [int(local_arr[i]) for i in shown_idx],
        "hf_vals": [int(hf_arr[i]) for i in shown_idx],
    }


# ── Logging ───────────────────────────────────────────────────────────────────


def build_log_entry(model: str, benchmark: str, subset: str, result: dict) -> str:
    lines = []
    if result["size_mismatch"]:
        lines.append(
            f"[SIZE MISMATCH] model={model}  benchmark={benchmark}  subset={subset}"
        )
        lines.append(f"  local_n={result['local_n']}  hf_n={result['hf_n']}")
    else:
        lines.append(
            f"[MISMATCH] model={model}  benchmark={benchmark}  subset={subset}"
        )
        lines.append(
            f"  local_n={result['local_n']}  hf_n={result['hf_n']}"
            f"  n_diff={result['n_diff']}  match_rate={result['match_rate']:.1%}"
        )
        lines.append(f"  differing indices (first 20): {result['diff_indices']}")
        lines.append(f"  local values at those indices: {result['local_vals']}")
        lines.append(f"  hf values at those indices:    {result['hf_vals']}")
    return "\n".join(lines)


def write_log(mismatch_entries: list[str], logs_dir: Path, run_time: datetime) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = run_time.strftime("%H:%M:%S_%B-%d-%Y")
    log_path = logs_dir / f"{timestamp}.log"

    header = f"=== Run at {run_time.strftime('%H:%M:%S %B-%d-%Y')} ==="
    content = header + "\n\n" + "\n\n".join(mismatch_entries) + "\n"
    log_path.write_text(content, encoding="utf-8")
    return log_path


# ── Per-benchmark verification ────────────────────────────────────────────────


def verify_benchmark(
    benchmark: Benchmark,
    n_models: int,
    rng: np.random.Generator,
) -> tuple[list[dict], list[str]]:
    """
    Verify `n_models` randomly sampled models for `benchmark`.

    Returns:
        results  — list of per-comparison dicts (for the summary table)
        mismatch_entries — list of log strings for any failures
    """
    hf_cfg = BENCHMARK_HF[benchmark]
    if hf_cfg is None:
        print(f"\n{'─' * 60}")
        print(f"=== {benchmark.pretty_name} — SKIPPED (v1 leaderboard format) ===")
        return [], []

    scores_path = resolve_model_scores_path(benchmark)
    local_df = pd.read_csv(scores_path)
    model_cols = get_model_columns(local_df)
    subset_col = get_subset_col(local_df)

    # Sample models
    sampled = rng.choice(model_cols, size=min(n_models, len(model_cols)), replace=False)
    sampled = list(sampled)

    print(f"\n{'─' * 60}")
    print(f"=== {benchmark.pretty_name} — {len(sampled)} models sampled ===")
    print(
        f"{'Model':<45} {'Subset':<40} {'N':>6} {'Match':>6} {'Diff':>5} {'Match%':>7}"
    )
    print("─" * 115)

    results = []
    mismatch_entries = []

    # For MMLU-Pro: cache the full HF dataset per model (one config for all subsets)
    hf_whole_cache: dict[str, pd.Series | None] = {}

    for model in sampled:
        per_subset = hf_cfg["per_subset"]

        # --- whole-benchmark mode (MMLU-Pro) ---
        if not per_subset:
            hf_suffix = hf_cfg["hf_suffix"]
            hf_benchmark = hf_cfg["hf_benchmark"]
            config = f"{model}__leaderboard_{hf_benchmark}_{hf_suffix}"

            if model not in hf_whole_cache:
                print(f"  Loading HF config: {config} …")
                hf_whole_cache[model] = load_hf_config(model, config)

            hf_all = hf_whole_cache[model]

            # Iterate subsets from the enum and compare per local subset slice
            hf_cursor = 0  # track position within the whole HF series
            for subset in benchmark.subsets:
                local_rows = local_df[local_df[subset_col] == subset][model].reset_index(
                    drop=True
                )
                local_n = len(local_rows)

                if hf_all is None:
                    row = {
                        "model": model, "benchmark": str(benchmark), "subset": subset,
                        "size_mismatch": True, "local_n": local_n, "hf_n": 0,
                        "n_diff": local_n, "match_rate": 0.0,
                        "diff_indices": [], "local_vals": [], "hf_vals": [],
                    }
                    print(
                        f"  {model:<43} {subset:<40} {local_n:>6} {'N/A':>6} {'N/A':>5} {'N/A':>7}"
                    )
                    results.append(row)
                    continue

                hf_slice = hf_all.iloc[hf_cursor: hf_cursor + local_n].reset_index(drop=True)
                hf_cursor += local_n

                result = compare_scores(local_rows, hf_slice)
                result.update({"model": model, "benchmark": str(benchmark), "subset": subset})
                results.append(result)

                match_str = "✓" if result["n_diff"] == 0 and not result["size_mismatch"] else "✗"
                print(
                    f"  {model:<43} {subset:<40} {result['local_n']:>6} {match_str:>6}"
                    f" {result['n_diff']:>5} {result['match_rate']:>7.1%}"
                )

                if result["n_diff"] > 0 or result["size_mismatch"]:
                    mismatch_entries.append(
                        build_log_entry(model, str(benchmark), subset, result)
                    )

        # --- per-subset mode (BBH, GPQA) ---
        else:
            hf_benchmark = hf_cfg["hf_benchmark"]
            strip_prefix = hf_cfg["strip_prefix"]

            for subset in benchmark.subsets:
                local_rows = local_df[local_df[subset_col] == subset][model].reset_index(
                    drop=True
                )
                local_n = len(local_rows)

                hf_subset_suffix = subset.removeprefix(strip_prefix)
                config = f"{model}__leaderboard_{hf_benchmark}_{hf_subset_suffix}"
                hf_scores = load_hf_config(model, config)

                if hf_scores is None:
                    row = {
                        "model": model, "benchmark": str(benchmark), "subset": subset,
                        "size_mismatch": True, "local_n": local_n, "hf_n": 0,
                        "n_diff": local_n, "match_rate": 0.0,
                        "diff_indices": [], "local_vals": [], "hf_vals": [],
                    }
                    print(
                        f"  {model:<43} {subset:<40} {local_n:>6} {'N/A':>6} {'N/A':>5} {'N/A':>7}"
                    )
                    results.append(row)
                    continue

                result = compare_scores(local_rows, hf_scores)
                result.update({"model": model, "benchmark": str(benchmark), "subset": subset})
                results.append(result)

                match_str = "✓" if result["n_diff"] == 0 and not result["size_mismatch"] else "✗"
                print(
                    f"  {model:<43} {subset:<40} {result['local_n']:>6} {match_str:>6}"
                    f" {result['n_diff']:>5} {result['match_rate']:>7.1%}"
                )

                if result["n_diff"] > 0 or result["size_mismatch"]:
                    mismatch_entries.append(
                        build_log_entry(model, str(benchmark), subset, result)
                    )

    return results, mismatch_entries


# ── Main ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify local per-instance model scores against HuggingFace."
    )
    parser.add_argument(
        "--n-models",
        type=int,
        default=3,
        help="Number of models to sample per benchmark (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model sampling (default: 42).",
    )
    valid = [b.value for b in Benchmark]
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=valid,
        choices=valid,
        metavar="BENCHMARK",
        help=f"Benchmarks to check. Choices: {valid}. Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    run_time = datetime.now()
    logs_dir = Path(__file__).parent / "logs"

    all_results: list[dict] = []
    all_mismatch_entries: list[str] = []

    benchmarks_to_run = [Benchmark(b) for b in args.benchmarks]

    for benchmark in benchmarks_to_run:
        results, mismatches = verify_benchmark(benchmark, args.n_models, rng)
        all_results.extend(results)
        all_mismatch_entries.extend(mismatches)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("SUMMARY")
    print(f"{'═' * 60}")

    checked = [r for r in all_results if not (r.get("hf_n", 0) == 0 and r.get("size_mismatch"))]
    total = len(checked)
    passed = sum(1 for r in checked if r["n_diff"] == 0 and not r["size_mismatch"])

    skipped_bms = [b.pretty_name for b in benchmarks_to_run if BENCHMARK_HF[b] is None]

    skipped_str = f"  Skipped: {', '.join(skipped_bms)} (v1 format)" if skipped_bms else ""
    print(f"  Matched: {passed}/{total} subset comparisons")
    if skipped_str:
        print(skipped_str)

    if all_mismatch_entries:
        log_path = write_log(all_mismatch_entries, logs_dir, run_time)
        print(f"  Mismatches logged to: {log_path}")
    else:
        if total > 0:
            print("  All comparisons passed — no log written.")

    print()


if __name__ == "__main__":
    main()
