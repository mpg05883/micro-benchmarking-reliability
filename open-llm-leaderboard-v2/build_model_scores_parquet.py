"""Read each benchmark directory under data/, combine the model scores
metadata JSON and .npy matrix into a single DataFrame, drop all-NaN
columns, and save as model_scores.parquet.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PARENT_DIR = Path(__file__).resolve().parent
DATA_DIR = PARENT_DIR / "data"

SCORES_FILE = "model_scores.npy"
METADATA_FILE = "model_scores_metadata.json"
OUTPUT_FILE = "model_scores.parquet"


def build_parquet(benchmark_dir: Path) -> None:
    """Build a parquet file from the npy + metadata JSON in *benchmark_dir*."""
    scores_path = benchmark_dir / SCORES_FILE
    metadata_path = benchmark_dir / METADATA_FILE

    if not scores_path.exists() or not metadata_path.exists():
        print(f"Skipping {benchmark_dir.name} — missing scores or metadata file")
        return

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    doc_hashes: list[str] = metadata["doc_hashes"]
    models: list[str] = metadata["models"]

    # Deduplicate model names — parquet requires unique column names.
    # Append a suffix (_2, _3, …) to repeated names.
    seen: dict[str, int] = {}
    unique_models: list[str] = []
    for name in models:
        count = seen.get(name, 0) + 1
        seen[name] = count
        unique_models.append(name if count == 1 else f"{name}_{count}")

    # Load the score matrix (instances × models)
    matrix = np.load(scores_path)

    # Build DataFrame with doc_hash as the first column and one column per model
    df = pd.DataFrame(matrix, columns=unique_models)
    df.insert(0, "doc_hash", doc_hashes)

    # Drop model columns where every value is NaN
    before = len(df.columns)
    df = df.dropna(axis=1, how="all")
    dropped = before - len(df.columns)

    output_path = benchmark_dir / OUTPUT_FILE
    df.to_parquet(output_path, index=False)

    print(
        f"{benchmark_dir.name}: saved {len(df)} rows × {len(df.columns) - 1} models "
        f"to {output_path.name}"
        + (f" (dropped {dropped} all-NaN columns)" if dropped else "")
    )


def main() -> None:
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        return

    for child in sorted(DATA_DIR.iterdir()):
        if child.is_dir():
            build_parquet(child)


if __name__ == "__main__":
    main()
