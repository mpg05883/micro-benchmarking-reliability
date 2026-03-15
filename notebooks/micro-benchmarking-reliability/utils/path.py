from pathlib import Path

from .enums import Benchmark


def resolve_data_dir():
    path = Path(__file__).parent.parent / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_root_dir():
    return Path(__file__).parent.parent.parent.parent


def resolve_leaderboard_results_dir():
    return resolve_root_dir() / "open-llm-leaderboard-results-combined"


def resolve_model_scores_path(benchmark: Benchmark):
    path = resolve_data_dir() / benchmark / "model_scores.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_dataset_path(benchmark: Benchmark):
    path = resolve_data_dir() / benchmark / "dataset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
