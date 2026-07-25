#!/usr/bin/env python3
"""Prepare the small, tabular part of the manuscript's Zenodo data record."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_FILES = OrderedDict(
    [
        ("biological", "biological_results.csv"),
        ("unconstrained", "unconstrained_results.csv"),
        ("connection_pruned", "connection_pruned_results.csv"),
        ("binned", "binned_results.csv"),
        ("neuron_binned", "neuron_binned_results.csv"),
    ]
)

DISPLAY_NAMES = {
    "biological": "Biological",
    "unconstrained": "Unconstrained",
    "connection_pruned": "Connection-pruned",
    "binned": "Binned",
    "neuron_binned": "Neuron binned",
}

SHAPE_ACCURACY = {
    "biological": 64.0,
    "unconstrained": 70.0,
    "connection_pruned": 69.0,
    "binned": 63.0,
    "neuron_binned": 60.0,
}
SHAPE_TEST_IMAGES_PER_CLASS = 5_000
SHAPE_TEST_N = 2 * SHAPE_TEST_IMAGES_PER_CLASS


def shape_binomial_sem_percent(mean_accuracy_percent: float) -> float:
    """Estimate shape-task SEM across the 10,000 held-out binary trials."""
    p = mean_accuracy_percent / 100
    return float(np.sqrt(p * (1 - p) / (SHAPE_TEST_N - 1)) * 100)


def parse_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Add the variables encoded in each stimulus filename."""
    out = df.copy()
    image_name = out["Image"].astype(str).map(lambda value: Path(value).name)
    counts = image_name.str.extract(r"img_(\d+)_(\d+)_")
    out["yellow_count"] = pd.to_numeric(counts[0], errors="coerce")
    out["blue_count"] = pd.to_numeric(counts[1], errors="coerce")
    smaller = out[["yellow_count", "blue_count"]].min(axis=1)
    larger = out[["yellow_count", "blue_count"]].max(axis=1)
    out["weber_ratio"] = larger / smaller
    out["equalized"] = image_name.str.lower().str.contains("equalized")

    class_dir = out["True label"].map({0: "blue", 1: "yellow"}).fillna("unknown")
    out["Image"] = [
        f"images/one_to_ten/test/{label}/{name}"
        for label, name in zip(class_dir, image_name)
    ]
    return out


def write_prediction_tables(output_dir: Path) -> dict[str, pd.DataFrame]:
    """Write portable copies of the trial-level numerical-task predictions."""
    parsed: dict[str, pd.DataFrame] = {}
    for key, filename in PREDICTION_FILES.items():
        source = PROJECT_ROOT / "supplementary_data" / filename
        df = pd.read_csv(source)
        df["Prediction"] = pd.to_numeric(df["Prediction"], errors="raise").astype(int)
        df["True label"] = pd.to_numeric(df["True label"], errors="raise").astype(int)
        df["Is correct"] = pd.to_numeric(df["Is correct"], errors="raise").astype(int)
        df = parse_trials(df)
        destination = output_dir / f"predictions_numerical_{key}.csv"
        df.to_csv(destination, index=False, lineterminator="\n")
        parsed[key] = df
    return parsed


def write_weber_source_data(output_dir: Path, trials: dict[str, pd.DataFrame]) -> None:
    rows: list[dict[str, object]] = []
    for key, df in trials.items():
        selected = df[df["equalized"] & np.isfinite(df["weber_ratio"])].copy()
        selected = selected[selected["weber_ratio"] >= 1.2]
        for ratio, group in selected.groupby("weber_ratio", sort=True):
            accuracy = group["Is correct"].astype(float)
            rows.append(
                {
                    "network": DISPLAY_NAMES[key],
                    "weber_ratio": round(float(ratio), 6),
                    "n_trials": int(len(group)),
                    "n_correct": int(accuracy.sum()),
                    "mean_accuracy_percent": float(accuracy.mean() * 100),
                    "standard_deviation_percent": float(accuracy.std(ddof=1) * 100),
                    "standard_error_percent": float(accuracy.sem(ddof=1) * 100),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / "source_data_figure_3f_weber.csv",
        index=False,
        lineterminator="\n",
    )


def write_task_accuracy_source_data(
    output_dir: Path, trials: dict[str, pd.DataFrame]
) -> None:
    rows: list[dict[str, object]] = []
    for key in PREDICTION_FILES:
        rows.append(
            {
                "task": "Color discrimination",
                "network": DISPLAY_NAMES[key],
                "mean_accuracy_percent": 100.0,
                "error_type": "value used in figure script",
                "error_percent": 0.0,
                "n_trials": pd.NA,
                "source": "connectome/visualization/models_accuracy.py",
            }
        )

        shape_mean = SHAPE_ACCURACY[key]
        rows.append(
            {
                "task": "Shape recognition",
                "network": DISPLAY_NAMES[key],
                "mean_accuracy_percent": shape_mean,
                "error_type": "binomial standard error of the mean",
                "error_percent": shape_binomial_sem_percent(shape_mean),
                "n_trials": SHAPE_TEST_N,
                "source": (
                    "derived from the reported accuracy and the held-out "
                    "test-set size"
                ),
            }
        )

        numerical = trials[key]
        numerical = numerical[
            numerical["equalized"] & (numerical["weber_ratio"] >= 1.33)
        ]
        accuracy = numerical["Is correct"].astype(float)
        rows.append(
            {
                "task": "Numerical discrimination",
                "network": DISPLAY_NAMES[key],
                "mean_accuracy_percent": float(accuracy.mean() * 100),
                "error_type": "standard error of the mean",
                "error_percent": float(accuracy.sem(ddof=1) * 100),
                "n_trials": int(len(numerical)),
                "source": f"predictions_numerical_{key}.csv",
            }
        )

    pd.DataFrame(rows).to_csv(
        output_dir / "source_data_figure_3_task_accuracy.csv",
        index=False,
        lineterminator="\n",
    )


def write_transfer_source_data(output_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "split": "Train",
                "accuracy_percent": 63.0,
                "source": "paper_figures/figure_3.ipynb",
            },
            {
                "split": "Test",
                "accuracy_percent": 48.0,
                "source": "paper_figures/figure_3.ipynb",
            },
        ]
    ).to_csv(
        output_dir / "source_data_figure_3d_transfer.csv",
        index=False,
        lineterminator="\n",
    )


def write_software_versions(output_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "repository": "connectome",
                "url": "https://github.com/eudald-seeslab/connectome",
                "commit": "93476a27692cc5a2e8e4c0ea0f1ec398ab5ae50d",
                "package_version": "0.2.0",
            },
            {
                "repository": "train-your-fly",
                "url": "https://github.com/eudald-seeslab/train-your-fly",
                "commit": "733a8bdb80cb68089d63f8af803a6180c0c67405",
                "package_version": "0.1.0",
            },
            {
                "repository": "cogstim",
                "url": "https://github.com/eudald-seeslab/cogstim",
                "commit": "e61ebf1d59d929af0fee2bc5e670bbbaf5e89f1f",
                "package_version": "0.8.1",
            },
        ]
    ).to_csv(
        output_dir / "software_versions.csv",
        index=False,
        lineterminator="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trials = write_prediction_tables(args.output_dir)
    write_weber_source_data(args.output_dir, trials)
    write_task_accuracy_source_data(args.output_dir, trials)
    write_transfer_source_data(args.output_dir)
    write_software_versions(args.output_dir)


if __name__ == "__main__":
    main()
