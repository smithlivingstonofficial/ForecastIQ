"""Data loading utilities for ForecastIQ AI.

This module reads the campaign CSV files from a supplied data directory.
It avoids absolute paths and discovers files using platform keywords so the
pipeline can still work when the hackathon evaluator replaces data/ with a
held-out dataset using the same schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


PlatformFrames = Dict[str, pd.DataFrame]


PLATFORM_KEYWORDS = {
    "google": ("google", "google_ads", "google-ads"),
    "bing": ("bing", "microsoft", "ms_ads", "msads"),
    "meta": ("meta", "facebook", "fb_ads", "meta_ads"),
}


def _list_csv_files(data_dir: Path) -> List[Path]:
    """Return all CSV files inside data_dir sorted by filename."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {data_dir}")

    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in data directory: {data_dir}")
    return csv_files


def _find_platform_file(csv_files: Iterable[Path], keywords: Iterable[str]) -> Path:
    """Find the first CSV file whose filename contains any platform keyword."""
    keyword_tuple = tuple(keyword.lower() for keyword in keywords)

    for path in csv_files:
        name = path.name.lower()
        if any(keyword in name for keyword in keyword_tuple):
            return path

    available = ", ".join(path.name for path in csv_files)
    raise FileNotFoundError(
        "Could not identify a required platform file. "
        f"Expected filename keywords {keyword_tuple}. Available files: {available}"
    )


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file with basic error context."""
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive error context
        raise RuntimeError(f"Failed to read CSV file {path}: {exc}") from exc


def load_platform_datasets(data_dir: str | Path) -> PlatformFrames:
    """Load Google, Bing/Microsoft, and Meta campaign CSVs.

    Parameters
    ----------
    data_dir:
        Directory containing the input CSV files.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys are: google, bing, meta.
    """
    data_path = Path(data_dir)
    csv_files = _list_csv_files(data_path)

    platform_paths = {
        platform: _find_platform_file(csv_files, keywords)
        for platform, keywords in PLATFORM_KEYWORDS.items()
    }

    return {platform: read_csv(path) for platform, path in platform_paths.items()}


def summarize_loaded_datasets(frames: PlatformFrames) -> pd.DataFrame:
    """Create a small summary table for quick development checks."""
    rows = []
    for platform, df in frames.items():
        rows.append(
            {
                "platform": platform,
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": ", ".join(map(str, df.columns)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load campaign CSV datasets.")
    parser.add_argument("--data-dir", default="data", help="Folder containing campaign CSV files.")
    args = parser.parse_args()

    frames = load_platform_datasets(args.data_dir)
    summary = summarize_loaded_datasets(frames)

    print("Loaded campaign datasets successfully.\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
