#!/usr/bin/env python3
"""
Generate extended summary plots and analysis tables for SDF docking results produced
by process_sdf_docking_glide.py.

Expected input files, by default in ./sdf_docking_results:

    pose_scores.csv
    aggregated_by_zinc.csv
    aggregated_by_target_and_zinc.csv
    target_summary.csv

Example usage:

    python plot_sdf_docking_results_extended.py
    python plot_sdf_docking_results_extended.py --results-dir sdf_docking_results --out sdf_docking_results/plots
    python plot_sdf_docking_results_extended.py --top-n 25 --max-pose-rank 15 --formats png pdf

The script keeps the original plots and adds:

    1. Target-level mean/median/CI summaries from all poses.
    2. Pose-rank-level summaries, globally and per target.
    3. Compound-target pose-count summaries.
    4. Plots showing how docking scores differ across targets, pose ranks,
       and number of generated/scored poses.
    5. A Markdown analysis report explaining what was done previously,
       what was added, and why the added analysis is needed.

Important convention:
    Lower, more negative docking scores are treated as better.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


# Plot style defaults. STIX keeps math text consistent with scientific figures.
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "mathtext.fontset": "stix",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)


REQUIRED_FILES = {
    "pose_scores": "pose_scores.csv",
    "aggregated_by_zinc": "aggregated_by_zinc.csv",
    "aggregated_by_target_and_zinc": "aggregated_by_target_and_zinc.csv",
    "target_summary": "target_summary.csv",
}

POSE_COLUMN_CANDIDATES = (
    "pose_rank",
    "pose_index",
    "pose_idx",
    "pose_id",
    "pose_number",
    "pose",
    "rank",
)


# ---------------------------------------------------------------------------
# IO and validation
# ---------------------------------------------------------------------------


def read_csv_checked(path: Path) -> pd.DataFrame:
    """Read a CSV file and fail with a clear message if it is missing or empty."""
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    return df



def load_results(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all required docking-result CSV files."""
    return {
        key: read_csv_checked(results_dir / filename)
        for key, filename in REQUIRED_FILES.items()
    }



def ensure_columns(df: pd.DataFrame, required: Iterable[str], name: str) -> None:
    """Validate that a dataframe contains required columns."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )



def clean_scores(df: pd.DataFrame, score_columns: Iterable[str]) -> pd.DataFrame:
    """Convert score columns to numeric and drop rows lacking all requested scores."""
    df = df.copy()
    for col in score_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    existing = [col for col in score_columns if col in df.columns]
    if existing:
        df = df.dropna(subset=existing, how="all")
    return df



def sanitize_filename(value: object) -> str:
    """Create a filesystem-safe file-name fragment."""
    text = str(value)
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in text)
    return safe.strip("_") or "unknown"



def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    """Save one figure in one or more formats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



def write_csv(df: pd.DataFrame, out_dir: Path, filename: str) -> Path:
    """Write an analysis table and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Derived data used by the added analysis
# ---------------------------------------------------------------------------


def infer_pose_rank(pose_scores: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Add a numeric pose_rank column.

    If an explicit pose column exists, use it. If not, infer pose rank from the
    input row order within each target-compound group. This makes the plot usable
    even when the processing script did not store a pose index explicitly.
    """
    df = pose_scores.copy().reset_index(drop=False).rename(columns={"index": "_input_order"})

    for col in POSE_COLUMN_CANDIDATES:
        if col not in df.columns:
            continue

        # Handles numeric pose IDs and strings such as "pose_1" or "ligand_pose3".
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() == 0:
            extracted = df[col].astype(str).str.extract(r"(-?\d+)", expand=False)
            numeric = pd.to_numeric(extracted, errors="coerce")

        if numeric.notna().sum() > 0:
            df["pose_rank"] = numeric
            return df, col

    group_cols = [col for col in ["target", "zinc_id"] if col in df.columns]
    if not group_cols:
        group_cols = ["target"] if "target" in df.columns else []

    if group_cols:
        df = df.sort_values(group_cols + ["_input_order"])
        df["pose_rank"] = df.groupby(group_cols).cumcount() + 1
        df = df.sort_values("_input_order")
    else:
        df["pose_rank"] = np.arange(1, len(df) + 1)

    return df, "inferred_from_input_order"



def prepare_pose_scores(pose_scores: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Validate and normalize pose-level scores."""
    ensure_columns(pose_scores, ["target", "docking_score"], "pose_scores.csv")
    df = clean_scores(pose_scores, ["docking_score"])
    if "zinc_id" not in df.columns:
        df["zinc_id"] = "unknown"

    df["target"] = df["target"].astype(str)
    df["zinc_id"] = df["zinc_id"].astype(str)
    df, pose_rank_source = infer_pose_rank(df)
    df["pose_rank"] = pd.to_numeric(df["pose_rank"], errors="coerce")
    df = df.dropna(subset=["pose_rank", "docking_score"])
    df["pose_rank"] = df["pose_rank"].astype(int)
    df = df[df["pose_rank"] > 0]

    return df, pose_rank_source



def score_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Compute robust score summaries for one or more grouping columns."""
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_poses=("docking_score", "count"),
            n_unique_zinc=("zinc_id", "nunique"),
            best_score=("docking_score", "min"),
            mean_score=("docking_score", "mean"),
            median_score=("docking_score", "median"),
            std_score=("docking_score", "std"),
            q10_score=("docking_score", lambda x: x.quantile(0.10)),
            q25_score=("docking_score", lambda x: x.quantile(0.25)),
            q75_score=("docking_score", lambda x: x.quantile(0.75)),
            q90_score=("docking_score", lambda x: x.quantile(0.90)),
        )
        .reset_index()
    )
    summary["std_score"] = summary["std_score"].fillna(0.0)
    summary["sem_score"] = summary["std_score"] / np.sqrt(summary["n_poses"].clip(lower=1))
    summary["ci95_score"] = 1.96 * summary["sem_score"]
    return summary



def build_derived_tables(pose_scores: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], str]:
    """Create all new analysis tables from pose-level data."""
    poses, pose_rank_source = prepare_pose_scores(pose_scores)

    target_average_scores = score_summary(poses, ["target"]).sort_values("mean_score")
    pose_rank_average_scores = score_summary(poses, ["pose_rank"]).sort_values("pose_rank")
    pose_rank_by_target_scores = score_summary(poses, ["target", "pose_rank"]).sort_values(
        ["target", "pose_rank"]
    )

    compound_target_pose_scores = (
        poses.groupby(["target", "zinc_id"], dropna=False)
        .agg(
            n_poses=("docking_score", "count"),
            best_score=("docking_score", "min"),
            mean_score=("docking_score", "mean"),
            median_score=("docking_score", "median"),
        )
        .reset_index()
    )

    pose_count_vs_score = (
        compound_target_pose_scores.groupby("n_poses", dropna=False)
        .agg(
            n_compound_target_pairs=("zinc_id", "count"),
            mean_best_score=("best_score", "mean"),
            median_best_score=("best_score", "median"),
            mean_pose_average_score=("mean_score", "mean"),
            median_pose_average_score=("mean_score", "median"),
        )
        .reset_index()
        .sort_values("n_poses")
    )

    first_pose = poses[poses["pose_rank"] == poses.groupby(["target", "zinc_id"])["pose_rank"].transform("min")]
    first_pose_by_target = score_summary(first_pose, ["target"])[
        ["target", "mean_score", "median_score", "best_score", "n_poses"]
    ].rename(
        columns={
            "mean_score": "first_pose_mean_score",
            "median_score": "first_pose_median_score",
            "best_score": "first_pose_best_score",
            "n_poses": "n_first_poses",
        }
    )

    best_pose_by_target = target_average_scores[["target", "best_score", "mean_score", "n_poses"]].rename(
        columns={
            "best_score": "target_best_score",
            "mean_score": "target_mean_score_all_poses",
            "n_poses": "n_all_poses",
        }
    )
    first_vs_best_by_target = first_pose_by_target.merge(best_pose_by_target, on="target", how="left")
    # Positive values mean that considering all poses improves the score relative to the first pose.
    first_vs_best_by_target["score_gain_from_pose_sampling"] = (
        first_vs_best_by_target["first_pose_mean_score"]
        - first_vs_best_by_target["target_best_score"]
    )
    first_vs_best_by_target = first_vs_best_by_target.sort_values(
        "score_gain_from_pose_sampling", ascending=False
    )

    tables = {
        "normalized_pose_scores": poses,
        "target_average_scores": target_average_scores,
        "pose_rank_average_scores": pose_rank_average_scores,
        "pose_rank_by_target_scores": pose_rank_by_target_scores,
        "compound_target_pose_scores": compound_target_pose_scores,
        "pose_count_vs_score": pose_count_vs_score,
        "first_vs_best_by_target": first_vs_best_by_target,
    }
    return tables, pose_rank_source


# ---------------------------------------------------------------------------
# Original plots, with slightly cleaner presentation
# ---------------------------------------------------------------------------


def plot_top_compounds_global(by_zinc: pd.DataFrame, out_dir: Path, top_n: int, formats: list[str]) -> None:
    """Horizontal bar chart: top compounds globally by best docking score."""
    ensure_columns(by_zinc, ["zinc_id", "best_score"], "aggregated_by_zinc.csv")
    df = clean_scores(by_zinc, ["best_score"]).sort_values("best_score", ascending=True).head(top_n)

    fig_height = max(4.2, 0.33 * len(df) + 1.4)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.barh(df["zinc_id"].astype(str), df["best_score"])
    ax.invert_yaxis()
    ax.set_xlabel("Best docking score (lower is better)")
    ax.set_ylabel("ZINC ID")
    ax.set_title(f"Top {len(df)} compounds by best docking score")
    save_figure(fig, out_dir, "top_compounds_global_best_score", formats)



def plot_top_compounds_by_target(by_target_zinc: pd.DataFrame, out_dir: Path, top_n: int, formats: list[str]) -> None:
    """One horizontal bar chart per target: top compounds by best score."""
    ensure_columns(
        by_target_zinc,
        ["target", "zinc_id", "best_score"],
        "aggregated_by_target_and_zinc.csv",
    )
    df = clean_scores(by_target_zinc, ["best_score"])

    for target, group in df.groupby("target", dropna=False):
        target_label = str(target)
        top = group.sort_values("best_score", ascending=True).head(top_n)
        fig_height = max(4.2, 0.33 * len(top) + 1.4)

        fig, ax = plt.subplots(figsize=(8.5, fig_height))
        ax.barh(top["zinc_id"].astype(str), top["best_score"])
        ax.invert_yaxis()
        ax.set_xlabel("Best docking score (lower is better)")
        ax.set_ylabel("ZINC ID")
        ax.set_title(f"Top {len(top)} compounds for {target_label}")

        save_figure(fig, out_dir, f"top_compounds_{sanitize_filename(target_label)}_best_score", formats)



def plot_score_distribution_by_target(pose_scores: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    """Boxplot of pose-level docking scores by target."""
    ensure_columns(pose_scores, ["target", "docking_score"], "pose_scores.csv")
    df = clean_scores(pose_scores, ["docking_score"])
    df["target"] = df["target"].astype(str)

    order = (
        df.groupby("target")["docking_score"]
        .median()
        .sort_values(ascending=True)
        .index.tolist()
    )
    data = [df.loc[df["target"] == target, "docking_score"].values for target in order]

    fig_width = max(7.0, 0.7 * len(order))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.boxplot(data, labels=order, showmeans=True, showfliers=False)
    ax.set_xlabel("Target")
    ax.set_ylabel("Docking score")
    ax.set_title("Pose-level docking score distribution by target")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="x", visible=False)
    save_figure(fig, out_dir, "docking_score_distribution_by_target", formats)



def plot_target_summary(target_summary: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    """Target-level comparison of best and mean scores."""
    ensure_columns(target_summary, ["target", "best_score", "mean_score"], "target_summary.csv")
    df = clean_scores(target_summary, ["best_score", "mean_score"]).sort_values("best_score")

    x = np.arange(len(df))
    width = 0.38

    fig_width = max(7.0, 0.72 * len(df))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.bar(x - width / 2, df["best_score"], width=width, label="Best score")
    ax.bar(x + width / 2, df["mean_score"], width=width, label="Mean score")
    ax.set_xticks(x)
    ax.set_xticklabels(df["target"].astype(str), rotation=45, ha="right")
    ax.set_xlabel("Target")
    ax.set_ylabel("Docking score")
    ax.set_title("Target summary: best and mean docking scores")
    ax.legend()
    ax.grid(axis="x", visible=False)
    save_figure(fig, out_dir, "target_summary_best_mean_scores", formats)



def plot_poses_per_target(target_summary: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    """Bar chart of number of poses and unique ZINC compounds per target."""
    ensure_columns(target_summary, ["target", "n_poses", "n_unique_zinc"], "target_summary.csv")
    df = target_summary.copy().sort_values("n_poses", ascending=False)

    x = np.arange(len(df))
    width = 0.38

    fig_width = max(7.0, 0.72 * len(df))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.bar(x - width / 2, df["n_poses"], width=width, label="Scored poses")
    ax.bar(x + width / 2, df["n_unique_zinc"], width=width, label="Unique ZINC IDs")
    ax.set_xticks(x)
    ax.set_xticklabels(df["target"].astype(str), rotation=45, ha="right")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    ax.set_title("Number of scored poses and unique compounds per target")
    ax.legend()
    ax.grid(axis="x", visible=False)
    save_figure(fig, out_dir, "poses_per_target", formats)



def plot_score_vs_pose_count(by_zinc: pd.DataFrame, out_dir: Path, top_n: int, formats: list[str]) -> None:
    """Scatter plot: number of poses vs best score for top compounds."""
    ensure_columns(by_zinc, ["zinc_id", "n_poses", "best_score", "mean_score"], "aggregated_by_zinc.csv")
    df = clean_scores(by_zinc, ["best_score", "mean_score"]).sort_values("best_score").head(top_n)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(df["n_poses"], df["best_score"], alpha=0.8)

    for _, row in df.head(min(10, len(df))).iterrows():
        ax.annotate(str(row["zinc_id"]), (row["n_poses"], row["best_score"]), fontsize=8)

    ax.set_xlabel("Number of scored poses")
    ax.set_ylabel("Best docking score")
    ax.set_title(f"Pose count vs best score for top {len(df)} compounds")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save_figure(fig, out_dir, "score_vs_pose_count_top_compounds", formats)


# ---------------------------------------------------------------------------
# New analytical plots
# ---------------------------------------------------------------------------


def plot_average_score_by_target(target_avg: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    """Mean docking score by target with 95% confidence intervals."""
    required = ["target", "mean_score", "ci95_score", "n_poses"]
    ensure_columns(target_avg, required, "target_average_scores")
    df = target_avg.sort_values("mean_score", ascending=True)

    fig_height = max(4.2, 0.42 * len(df) + 1.4)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.barh(df["target"].astype(str), df["mean_score"], xerr=df["ci95_score"], capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel("Mean docking score (95% CI; lower is better)")
    ax.set_ylabel("Target")
    ax.set_title("Average docking score by target")
    ax.grid(axis="y", visible=False)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["mean_score"],
            i,
            f"  n={int(row['n_poses'])}",
            va="center",
            fontsize=8,
        )

    save_figure(fig, out_dir, "average_docking_score_by_target", formats)



def plot_target_score_profile(target_avg: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    """Compare best, median, and mean target scores."""
    required = ["target", "best_score", "median_score", "mean_score"]
    ensure_columns(target_avg, required, "target_average_scores")
    df = target_avg.sort_values("best_score")

    x = np.arange(len(df))
    fig_width = max(7.5, 0.75 * len(df))
    fig, ax = plt.subplots(figsize=(fig_width, 5.3))
    ax.plot(x, df["best_score"], marker="o", label="Best score")
    ax.plot(x, df["median_score"], marker="o", label="Median score")
    ax.plot(x, df["mean_score"], marker="o", label="Mean score")
    ax.set_xticks(x)
    ax.set_xticklabels(df["target"].astype(str), rotation=45, ha="right")
    ax.set_xlabel("Target")
    ax.set_ylabel("Docking score")
    ax.set_title("Target docking-score profile")
    ax.legend()
    ax.grid(axis="x", visible=False)
    save_figure(fig, out_dir, "target_docking_score_profile", formats)



def plot_pose_rank_global_trend(
    pose_rank_avg: pd.DataFrame,
    out_dir: Path,
    max_pose_rank: int | None,
    formats: list[str],
) -> None:
    """Show how average docking score changes with pose rank globally."""
    required = ["pose_rank", "mean_score", "ci95_score", "n_poses", "best_score"]
    ensure_columns(pose_rank_avg, required, "pose_rank_average_scores")
    df = pose_rank_avg.copy().sort_values("pose_rank")
    if max_pose_rank is not None:
        df = df[df["pose_rank"] <= max_pose_rank]
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.errorbar(
        df["pose_rank"],
        df["mean_score"],
        yerr=df["ci95_score"],
        marker="o",
        capsize=3,
        label="Mean score",
    )
    ax.plot(df["pose_rank"], df["best_score"], marker="s", linestyle="--", label="Best score")
    ax.set_xlabel("Pose rank")
    ax.set_ylabel("Docking score")
    ax.set_title("Docking score by pose rank")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    save_figure(fig, out_dir, "docking_score_by_pose_rank_global", formats)



def plot_pose_rank_by_target_trend(
    pose_rank_by_target: pd.DataFrame,
    out_dir: Path,
    max_pose_rank: int | None,
    max_targets: int,
    formats: list[str],
) -> None:
    """Show mean docking-score trend across pose ranks for the most represented targets."""
    required = ["target", "pose_rank", "mean_score", "n_poses"]
    ensure_columns(pose_rank_by_target, required, "pose_rank_by_target_scores")
    df = pose_rank_by_target.copy().sort_values(["target", "pose_rank"])
    if max_pose_rank is not None:
        df = df[df["pose_rank"] <= max_pose_rank]
    if df.empty:
        return

    target_order = (
        df.groupby("target")["n_poses"]
        .sum()
        .sort_values(ascending=False)
        .head(max_targets)
        .index.tolist()
    )
    df = df[df["target"].isin(target_order)]

    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    for target in target_order:
        group = df[df["target"] == target].sort_values("pose_rank")
        ax.plot(group["pose_rank"], group["mean_score"], marker="o", label=str(target))

    ax.set_xlabel("Pose rank")
    ax.set_ylabel("Mean docking score")
    ax.set_title("Mean docking score by pose rank and target")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="best", ncol=1 if len(target_order) <= 8 else 2)
    save_figure(fig, out_dir, "docking_score_by_pose_rank_and_target", formats)



def plot_pose_rank_heatmap(
    pose_rank_by_target: pd.DataFrame,
    out_dir: Path,
    max_pose_rank: int | None,
    formats: list[str],
) -> None:
    """Heatmap of target-by-pose-rank mean docking scores."""
    required = ["target", "pose_rank", "mean_score"]
    ensure_columns(pose_rank_by_target, required, "pose_rank_by_target_scores")
    df = pose_rank_by_target.copy()
    if max_pose_rank is not None:
        df = df[df["pose_rank"] <= max_pose_rank]
    if df.empty:
        return

    pivot = df.pivot_table(index="target", columns="pose_rank", values="mean_score", aggfunc="mean")
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    fig_width = max(7.0, 0.45 * len(pivot.columns) + 3.0)
    fig_height = max(4.5, 0.35 * len(pivot.index) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(pivot.values, aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Mean docking score")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns.astype(str))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_xlabel("Pose rank")
    ax.set_ylabel("Target")
    ax.set_title("Mean docking score heatmap by target and pose rank")
    ax.grid(False)
    save_figure(fig, out_dir, "docking_score_pose_rank_target_heatmap", formats)



def plot_pose_count_vs_score_summary(
    pose_count_summary: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
) -> None:
    """Mean score metrics as a function of the number of poses per compound-target pair."""
    required = ["n_poses", "mean_best_score", "mean_pose_average_score", "n_compound_target_pairs"]
    ensure_columns(pose_count_summary, required, "pose_count_vs_score")
    df = pose_count_summary.sort_values("n_poses")
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(df["n_poses"], df["mean_best_score"], marker="o", label="Mean best score")
    ax.plot(
        df["n_poses"],
        df["mean_pose_average_score"],
        marker="o",
        label="Mean pose-average score",
    )
    ax.set_xlabel("Number of poses per compound-target pair")
    ax.set_ylabel("Docking score")
    ax.set_title("Docking score vs number of poses")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    save_figure(fig, out_dir, "docking_score_vs_number_of_poses", formats)



def plot_compound_target_pose_count_scatter(
    compound_target_pose_scores: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
) -> None:
    """Scatter plot: pose count vs best score for each target-compound pair."""
    required = ["target", "zinc_id", "n_poses", "best_score", "mean_score"]
    ensure_columns(compound_target_pose_scores, required, "compound_target_pose_scores")
    df = compound_target_pose_scores.copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    ax.scatter(df["n_poses"], df["best_score"], alpha=0.45)

    # Add a simple least-squares line if there is enough variation in pose counts.
    if df["n_poses"].nunique() > 1 and len(df) >= 3:
        x = df["n_poses"].astype(float).to_numpy()
        y = df["best_score"].astype(float).to_numpy()
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, linestyle="--", label="Linear trend")
        ax.legend()

    ax.set_xlabel("Number of poses per compound-target pair")
    ax.set_ylabel("Best docking score")
    ax.set_title("Best docking score vs pose count per compound-target pair")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save_figure(fig, out_dir, "compound_target_best_score_vs_pose_count", formats)



def plot_first_vs_best_by_target(
    first_vs_best: pd.DataFrame,
    out_dir: Path,
    formats: list[str],
) -> None:
    """How much target-level score improves when all poses are considered."""
    required = ["target", "score_gain_from_pose_sampling"]
    ensure_columns(first_vs_best, required, "first_vs_best_by_target")
    df = first_vs_best.sort_values("score_gain_from_pose_sampling", ascending=False)
    if df.empty:
        return

    fig_height = max(4.2, 0.42 * len(df) + 1.4)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.barh(df["target"].astype(str), df["score_gain_from_pose_sampling"])
    ax.invert_yaxis()
    ax.set_xlabel("Score gain from sampling multiple poses")
    ax.set_ylabel("Target")
    ax.set_title("Improvement from first pose to best observed pose")
    ax.grid(axis="y", visible=False)
    save_figure(fig, out_dir, "score_gain_from_pose_sampling_by_target", formats)


# ---------------------------------------------------------------------------
# Analysis report and index
# ---------------------------------------------------------------------------


def format_float(value: float | int | np.floating | None, digits: int = 3) -> str:
    """Format numbers safely for the Markdown report."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.{digits}f}"



def write_analysis_report(
    out_dir: Path,
    tables: dict[str, pd.DataFrame],
    pose_rank_source: str,
    formats: list[str],
) -> None:
    """Write a concise Markdown report with data-driven highlights."""
    target_avg = tables["target_average_scores"]
    pose_rank_avg = tables["pose_rank_average_scores"]
    pose_count_vs_score = tables["pose_count_vs_score"]
    first_vs_best = tables["first_vs_best_by_target"]

    best_mean_target = target_avg.sort_values("mean_score").iloc[0]
    best_single_target = target_avg.sort_values("best_score").iloc[0]
    best_pose_rank = pose_rank_avg.sort_values("mean_score").iloc[0]

    report = []
    report.append("# Extended docking-score analysis\n")
    report.append("## What was done before\n")
    report.append(
        "The previous script loaded the four docking-result CSV files, generated global and "
        "per-target rankings of top compounds, plotted score distributions by target, compared "
        "best and mean scores per target, counted poses/unique compounds per target, and showed "
        "a basic pose-count versus best-score scatter for top compounds.\n"
    )

    report.append("## What was added\n")
    report.append(
        "This version adds derived analysis tables computed directly from `pose_scores.csv`: "
        "target-level average score summaries, global pose-rank summaries, target-by-pose-rank "
        "summaries, compound-target pose-count summaries, and first-pose-versus-best-pose summaries. "
        f"Pose rank source: `{pose_rank_source}`.\n"
    )
    report.append(
        "New figures focus on three questions: whether targets differ in average docking quality, "
        "whether later/alternative poses systematically change docking scores, and whether compounds "
        "with more scored poses tend to produce better best scores.\n"
    )

    report.append("## Why this is needed\n")
    report.append(
        "A best-score ranking alone can be misleading because it ignores the number of poses, "
        "target-specific score distributions, and uncertainty in the pose-level scores. Averaging "
        "over targets and pose ranks helps separate genuinely strong docking behavior from effects "
        "caused by uneven pose counts or isolated extreme poses. The added confidence intervals, "
        "median/quantile summaries, and score-versus-pose plots make the interpretation more robust.\n"
    )

    report.append("## Data-driven highlights\n")
    report.append(
        f"- Best target by mean docking score: `{best_mean_target['target']}` "
        f"with mean score {format_float(best_mean_target['mean_score'])} "
        f"from {int(best_mean_target['n_poses'])} poses.\n"
    )
    report.append(
        f"- Best single observed target score: `{best_single_target['target']}` "
        f"with best score {format_float(best_single_target['best_score'])}.\n"
    )
    report.append(
        f"- Best pose rank by global mean score: pose rank {int(best_pose_rank['pose_rank'])} "
        f"with mean score {format_float(best_pose_rank['mean_score'])}.\n"
    )

    if not pose_count_vs_score.empty:
        best_pose_count = pose_count_vs_score.sort_values("mean_best_score").iloc[0]
        report.append(
            f"- Pose-count group with the best mean best-score: {int(best_pose_count['n_poses'])} "
            f"poses per compound-target pair, mean best score "
            f"{format_float(best_pose_count['mean_best_score'])}.\n"
        )

    if not first_vs_best.empty:
        biggest_gain = first_vs_best.sort_values("score_gain_from_pose_sampling", ascending=False).iloc[0]
        report.append(
            f"- Largest gain from sampling multiple poses: `{biggest_gain['target']}` "
            f"with gain {format_float(biggest_gain['score_gain_from_pose_sampling'])}.\n"
        )

    report.append("\n## Generated analysis tables\n")
    for filename in [
        "target_average_scores.csv",
        "pose_rank_average_scores.csv",
        "pose_rank_by_target_scores.csv",
        "compound_target_pose_scores.csv",
        "pose_count_vs_score.csv",
        "first_vs_best_by_target.csv",
    ]:
        report.append(f"- `{filename}`\n")

    report.append("\n## Generated plot formats\n")
    report.append(f"{', '.join(formats)}\n")

    (out_dir / "analysis_summary.md").write_text("".join(report), encoding="utf-8")



def write_plot_index(out_dir: Path, formats: list[str]) -> None:
    """Write a text index of generated plot and analysis files."""
    plot_files = sorted(
        path.name
        for path in out_dir.iterdir()
        if path.suffix.lower().lstrip(".") in set(formats)
    )
    table_files = sorted(path.name for path in out_dir.iterdir() if path.suffix.lower() == ".csv")
    report_files = sorted(path.name for path in out_dir.iterdir() if path.suffix.lower() == ".md")

    index_path = out_dir / "plot_index.txt"
    lines = ["Generated docking-result outputs:\n", "\nPlots:\n"]
    lines.extend(f"- {file}\n" for file in plot_files)
    lines.append("\nAnalysis tables:\n")
    lines.extend(f"- {file}\n" for file in table_files)
    lines.append("\nReports:\n")
    lines.extend(f"- {file}\n" for file in report_files)
    index_path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate extended plots from SDF docking result CSV files.")
    parser.add_argument(
        "--results-dir",
        default="sdf_docking_results",
        help="Directory containing pose_scores.csv and aggregation CSVs. Default: sdf_docking_results",
    )
    parser.add_argument(
        "--out",
        default="sdf_docking_results/plots",
        help="Output directory for plots, tables and the analysis report. Default: sdf_docking_results/plots",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top compounds to show in ranking plots. Default: 20",
    )
    parser.add_argument(
        "--max-pose-rank",
        type=int,
        default=10,
        help="Maximum pose rank shown in pose-rank plots. Use 0 to show all. Default: 10",
    )
    parser.add_argument(
        "--max-targets-in-pose-trend",
        type=int,
        default=12,
        help="Maximum number of targets shown in the multi-target pose-rank trend. Default: 12",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "svg"],
        help="Plot file formats to save. Default: png",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out)
    max_pose_rank = None if args.max_pose_rank == 0 else args.max_pose_rank

    try:
        data = load_results(results_dir)
        derived_tables, pose_rank_source = build_derived_tables(data["pose_scores"])

        # Save new analysis tables first, so the script provides inspectable numbers
        # even if a later plotting step fails.
        write_csv(derived_tables["target_average_scores"], out_dir, "target_average_scores.csv")
        write_csv(derived_tables["pose_rank_average_scores"], out_dir, "pose_rank_average_scores.csv")
        write_csv(derived_tables["pose_rank_by_target_scores"], out_dir, "pose_rank_by_target_scores.csv")
        write_csv(derived_tables["compound_target_pose_scores"], out_dir, "compound_target_pose_scores.csv")
        write_csv(derived_tables["pose_count_vs_score"], out_dir, "pose_count_vs_score.csv")
        write_csv(derived_tables["first_vs_best_by_target"], out_dir, "first_vs_best_by_target.csv")

        # Original plots.
        plot_top_compounds_global(data["aggregated_by_zinc"], out_dir, args.top_n, args.formats)
        plot_top_compounds_by_target(
            data["aggregated_by_target_and_zinc"], out_dir, args.top_n, args.formats
        )
        plot_score_distribution_by_target(data["pose_scores"], out_dir, args.formats)
        plot_target_summary(data["target_summary"], out_dir, args.formats)
        plot_poses_per_target(data["target_summary"], out_dir, args.formats)
        plot_score_vs_pose_count(data["aggregated_by_zinc"], out_dir, args.top_n, args.formats)

        # Added analytical plots.
        plot_average_score_by_target(derived_tables["target_average_scores"], out_dir, args.formats)
        plot_target_score_profile(derived_tables["target_average_scores"], out_dir, args.formats)
        plot_pose_rank_global_trend(
            derived_tables["pose_rank_average_scores"], out_dir, max_pose_rank, args.formats
        )
        plot_pose_rank_by_target_trend(
            derived_tables["pose_rank_by_target_scores"],
            out_dir,
            max_pose_rank,
            args.max_targets_in_pose_trend,
            args.formats,
        )
        plot_pose_rank_heatmap(
            derived_tables["pose_rank_by_target_scores"], out_dir, max_pose_rank, args.formats
        )
        plot_pose_count_vs_score_summary(
            derived_tables["pose_count_vs_score"], out_dir, args.formats
        )
        plot_compound_target_pose_count_scatter(
            derived_tables["compound_target_pose_scores"], out_dir, args.formats
        )
        plot_first_vs_best_by_target(
            derived_tables["first_vs_best_by_target"], out_dir, args.formats
        )

        write_analysis_report(out_dir, derived_tables, pose_rank_source, args.formats)
        write_plot_index(out_dir, args.formats)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Plots, tables and report saved to: {out_dir}")
    print(f"Formats: {', '.join(args.formats)}")
    print(f"Pose rank source: {pose_rank_source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
