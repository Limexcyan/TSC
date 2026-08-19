#!/usr/bin/env python3
"""
Create docking-vs-alignment plots and result tables for TaSC.

Inputs
------
1) alignment CSV:
   Required target-pair columns:
     Sample_1/Sample_2 or target_1/target_2 or Target_1/Target_2
   Required protein similarity column:
     Alignment_Score or alignment_score
   Optional ligand-space column:
     JS_Divergence or js_divergence

2) pose score CSV:
   Required columns:
     target, docking_score
   Ligand identifier:
     zinc_id preferred; otherwise molecule_name; otherwise ligand_id
   Optional pose column:
     pose_id or pose_index_in_file

Main output
-----------
- target_pair_alignment_docking_comparison.csv
- correlation_summary.csv
- docking_vs_alignment_distance scatter plot
- docking profile heatmaps
- docking score distribution plots

Interpretation
--------------
Each point in docking-vs-alignment plots is one pair of targets.
Alignment distance measures protein/pocket dissimilarity.
Docking distance measures how differently two targets rank the same ligands.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


# -------------------------
# Plot style
# -------------------------
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
    }
)


# -------------------------
# Helpers
# -------------------------
def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_target_name(value: object) -> str:
    """Normalize target labels across alignment and docking tables.

    Examples:
      5H2A          -> 5HT2A
      5ht1a_8PJK    -> 5HT1A
      5HT4R         -> 5HT4
    """
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    s = s.replace("-", "").replace(" ", "")
    # common typo/format: 5H2A -> 5HT2A
    s = re.sub(r"^5H(?=\d)", "5HT", s)
    # remove PDB/grid suffix, e.g. 5HT1A_8PJK -> 5HT1A
    if "_" in s:
        s = s.split("_")[0]
    # normalize receptor suffix in labels such as 5HT4R or 5HT6R
    s = re.sub(r"^(5HT\d+[A-Z]?)R$", r"\1", s)
    return s


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    lower_to_original = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    raise ValueError(
        f"Cannot find {label}. Tried columns: {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> list[Path]:
    saved = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, bbox_inches="tight")
        saved.append(path)
    plt.close(fig)
    return saved


def pair_key(a: str, b: str) -> str:
    x, y = sorted([clean_target_name(a), clean_target_name(b)])
    return f"{x}__{y}"


def compute_basic_stats(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """Return Pearson, Spearman, R2 and p-values."""
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(valid)
    out = {
        "n": n,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "r2": np.nan,
    }
    if n < 3:
        return out

    if valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return out

    if stats is not None:
        pr = stats.pearsonr(valid["x"], valid["y"])
        sr = stats.spearmanr(valid["x"], valid["y"])
        out["pearson_r"] = float(pr.statistic)
        out["pearson_p"] = float(pr.pvalue)
        out["spearman_rho"] = float(sr.statistic)
        out["spearman_p"] = float(sr.pvalue)
    else:
        out["pearson_r"] = float(valid["x"].corr(valid["y"], method="pearson"))
        out["spearman_rho"] = float(valid["x"].corr(valid["y"], method="spearman"))
    out["r2"] = float(out["pearson_r"] ** 2) if not pd.isna(out["pearson_r"]) else np.nan
    return out


def add_regression_line(ax: plt.Axes, x: pd.Series, y: pd.Series) -> None:
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 2 or valid["x"].nunique() < 2:
        return
    coeff = np.polyfit(valid["x"], valid["y"], deg=1)
    xx = np.linspace(valid["x"].min(), valid["x"].max(), 100)
    yy = coeff[0] * xx + coeff[1]
    ax.plot(xx, yy, linestyle="--", linewidth=1.5, label="Linear fit")


def annotate_points(ax: plt.Axes, df: pd.DataFrame, x_col: str, y_col: str, max_labels: int = 40) -> None:
    if len(df) > max_labels:
        return
    for _, row in df.iterrows():
        if pd.isna(row[x_col]) or pd.isna(row[y_col]):
            continue
        label = f"{row['target_1']}-{row['target_2']}"
        ax.annotate(label, (row[x_col], row[y_col]), fontsize=7, xytext=(3, 3), textcoords="offset points")


# -------------------------
# Loading and preparation
# -------------------------
def load_alignment_table(path: Path, distance_method: str = "minmax") -> pd.DataFrame:
    df = pd.read_csv(path)

    t1_col = first_existing_column(
        df,
        ["target_1", "target1", "Target_1", "Target1", "Sample_1", "sample_1"],
        "first target column",
    )
    t2_col = first_existing_column(
        df,
        ["target_2", "target2", "Target_2", "Target2", "Sample_2", "sample_2"],
        "second target column",
    )

    score_col = None
    dist_col = None
    js_col = None

    for candidates, varname in [
        (["alignment_distance", "Alignment_Distance", "protein_distance"], "dist"),
        (["alignment_score", "Alignment_Score", "score", "AlignmentScore"], "score"),
        (["JS_Divergence", "js_divergence", "js", "ligand_js"], "js"),
    ]:
        try:
            found = first_existing_column(df, candidates, varname)
            if varname == "dist":
                dist_col = found
            elif varname == "score":
                score_col = found
            elif varname == "js":
                js_col = found
        except ValueError:
            pass

    if score_col is None and dist_col is None:
        raise ValueError("Alignment CSV must contain either Alignment_Score or alignment_distance.")

    out = pd.DataFrame()
    out["target_1"] = df[t1_col].map(clean_target_name)
    out["target_2"] = df[t2_col].map(clean_target_name)
    out = out[(out["target_1"] != "") & (out["target_2"] != "")]
    out = out[out["target_1"] != out["target_2"]]
    out["pair_key"] = [pair_key(a, b) for a, b in zip(out["target_1"], out["target_2"])]

    if score_col is not None:
        out["alignment_score"] = pd.to_numeric(df.loc[out.index, score_col], errors="coerce")
    if dist_col is not None:
        out["alignment_distance"] = pd.to_numeric(df.loc[out.index, dist_col], errors="coerce")

    if js_col is not None:
        out["js_divergence"] = pd.to_numeric(df.loc[out.index, js_col], errors="coerce")

    if "alignment_distance" not in out.columns or out["alignment_distance"].isna().all():
        scores = out["alignment_score"]
        if distance_method == "score_over_max":
            max_score = scores.max()
            out["alignment_distance"] = 1.0 - scores / max_score if max_score != 0 else np.nan
        elif distance_method == "minmax":
            min_score = scores.min()
            max_score = scores.max()
            denom = max_score - min_score
            out["alignment_distance"] = (max_score - scores) / denom if denom != 0 else 0.0
        else:
            raise ValueError(f"Unknown distance method: {distance_method}")

    # Aggregate duplicate target pairs, if any
    numeric_cols = [c for c in ["alignment_score", "alignment_distance", "js_divergence"] if c in out.columns]
    out = (
        out.groupby("pair_key", as_index=False)
        .agg(
            target_1=("target_1", "first"),
            target_2=("target_2", "first"),
            **{col: (col, "mean") for col in numeric_cols},
        )
    )
    return out


def load_pose_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    target_col = first_existing_column(df, ["target", "Target", "receptor", "protein"], "target")
    score_col = first_existing_column(
        df,
        ["docking_score", "Docking_Score", "score", "glide_score", "smina_score", "r_i_docking_score"],
        "docking score",
    )

    try:
        ligand_col = first_existing_column(
            df,
            ["zinc_id", "ZINC_ID", "ligand_id", "Ligand_ID", "molecule_name", "compound_id", "name"],
            "ligand identifier",
        )
    except ValueError:
        raise ValueError(
            "Pose score CSV needs a ligand identifier column. "
            "Use one of: zinc_id, ligand_id, molecule_name, compound_id, name."
        )

    pose_col = None
    for candidates in [
        ["pose_id", "Pose_ID"],
        ["pose_index_in_file", "pose_index", "pose_rank", "i_i_glide_posenum"],
    ]:
        try:
            pose_col = first_existing_column(df, candidates, "pose column")
            break
        except ValueError:
            pass

    out = pd.DataFrame()
    out["target"] = df[target_col].map(clean_target_name)
    out["ligand_id"] = df[ligand_col].astype(str).str.strip()
    out["docking_score"] = pd.to_numeric(df[score_col], errors="coerce")

    if pose_col is not None:
        out["pose_id"] = df[pose_col]
    else:
        # fallback: order within ligand-target group
        out["pose_id"] = (
            out.groupby(["target", "ligand_id"], dropna=False).cumcount() + 1
        )

    for col in ["source_file", "score_field", "molecule_name", "zinc_id"]:
        if col in df.columns:
            out[col] = df[col]

    out = out[(out["target"] != "") & (out["ligand_id"] != "")]
    out = out.dropna(subset=["docking_score"])
    if out.empty:
        raise ValueError("No valid docking scores after cleaning.")
    return out


def aggregate_poses(pose_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pose-level scores to ligand-target level.

    Lower docking score is assumed to be better.
    """
    records = []
    group_cols = ["target", "ligand_id"]

    for (target, ligand), group in pose_df.groupby(group_cols, sort=False):
        scores = group["docking_score"].dropna().sort_values()
        if scores.empty:
            continue

        best_idx = group["docking_score"].idxmin()
        best_row = group.loc[best_idx]
        top3 = scores.head(min(3, len(scores)))

        records.append(
            {
                "target": target,
                "ligand_id": ligand,
                "best_score": float(scores.iloc[0]),
                "mean_score": float(scores.mean()),
                "median_score": float(scores.median()),
                "top3_mean_score": float(top3.mean()),
                "n_poses": int(len(scores)),
                "best_pose_id": best_row.get("pose_id", np.nan),
            }
        )

    agg = pd.DataFrame.from_records(records)
    if agg.empty:
        raise ValueError("Pose aggregation produced no rows.")

    # Within-target normalization: lower scores are better.
    agg["score_z_within_target"] = agg.groupby("target")["best_score"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) != 0 else 0.0
    )
    agg["score_percentile_within_target"] = agg.groupby("target")["best_score"].rank(
        method="average", ascending=True, pct=True
    )
    return agg


def compute_target_summary(agg: pd.DataFrame) -> pd.DataFrame:
    return (
        agg.groupby("target")
        .agg(
            n_ligands=("ligand_id", "nunique"),
            n_ligand_target_rows=("ligand_id", "size"),
            mean_best_score=("best_score", "mean"),
            median_best_score=("best_score", "median"),
            sd_best_score=("best_score", "std"),
            min_best_score=("best_score", "min"),
            q05_best_score=("best_score", lambda s: s.quantile(0.05)),
            q25_best_score=("best_score", lambda s: s.quantile(0.25)),
            q75_best_score=("best_score", lambda s: s.quantile(0.75)),
            mean_n_poses=("n_poses", "mean"),
            total_poses=("n_poses", "sum"),
        )
        .reset_index()
        .sort_values("target")
    )


def compute_pairwise_docking(
    agg: pd.DataFrame,
    min_common_ligands: int = 20,
    top_k: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_matrix = agg.pivot_table(index="ligand_id", columns="target", values="best_score", aggfunc="min")
    z_matrix = agg.pivot_table(index="ligand_id", columns="target", values="score_z_within_target", aggfunc="min")

    targets = list(score_matrix.columns)
    records = []

    for a, b in itertools.combinations(targets, 2):
        sub = score_matrix[[a, b]].dropna()
        n_common = len(sub)

        pearson_r = pearson_p = spearman_rho = spearman_p = np.nan
        docking_distance = np.nan

        if n_common >= min_common_ligands and sub[a].nunique() > 1 and sub[b].nunique() > 1:
            if stats is not None:
                pr = stats.pearsonr(sub[a], sub[b])
                sr = stats.spearmanr(sub[a], sub[b])
                pearson_r, pearson_p = float(pr.statistic), float(pr.pvalue)
                spearman_rho, spearman_p = float(sr.statistic), float(sr.pvalue)
            else:
                pearson_r = float(sub[a].corr(sub[b], method="pearson"))
                spearman_rho = float(sub[a].corr(sub[b], method="spearman"))
            docking_distance = 1.0 - spearman_rho

        # Top-k overlap, using available ligands per target
        top_a = set(score_matrix[a].dropna().nsmallest(top_k).index)
        top_b = set(score_matrix[b].dropna().nsmallest(top_k).index)
        union = top_a | top_b
        intersection = top_a & top_b
        topk_jaccard = len(intersection) / len(union) if union else np.nan

        records.append(
            {
                "target_1": a,
                "target_2": b,
                "pair_key": pair_key(a, b),
                "n_common_ligands": n_common,
                "pearson_score_correlation": pearson_r,
                "pearson_score_p": pearson_p,
                "spearman_score_correlation": spearman_rho,
                "spearman_score_p": spearman_p,
                "docking_distance": docking_distance,
                "topk": top_k,
                "topk_jaccard_overlap": topk_jaccard,
                "n_topk_intersection": len(intersection),
                "n_topk_union": len(union),
            }
        )

    pair_df = pd.DataFrame.from_records(records)
    return pair_df, score_matrix, z_matrix


# -------------------------
# Plotting
# -------------------------
def plot_alignment_vs_js(alignment: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if "js_divergence" not in alignment.columns:
        return
    df = alignment.dropna(subset=["alignment_distance", "js_divergence"]).copy()
    if df.empty:
        return
    stats_dict = compute_basic_stats(df["alignment_distance"], df["js_divergence"])

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(df["alignment_distance"], df["js_divergence"], alpha=0.8)
    add_regression_line(ax, df["alignment_distance"], df["js_divergence"])
    annotate_points(ax, df, "alignment_distance", "js_divergence")
    ax.set_xlabel("Alignment distance")
    ax.set_ylabel("Ligand-distribution JS divergence")
    title = "Ligand distribution divergence vs alignment distance"
    if not pd.isna(stats_dict["pearson_r"]):
        title += f"\nPearson r={stats_dict['pearson_r']:.3f}, Spearman ρ={stats_dict['spearman_rho']:.3f}, n={stats_dict['n']}"
    ax.set_title(title)
    ax.legend(loc="best")
    save_figure(fig, out_dir, "fig_e1_alignment_distance_vs_js_divergence", formats)


def plot_score_distribution(agg: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    targets = sorted(agg["target"].unique())
    data = [agg.loc[agg["target"] == t, "best_score"].dropna().values for t in targets]
    fig_width = max(6.0, 0.55 * len(targets))
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    ax.boxplot(data, labels=targets, showmeans=True)
    ax.set_xlabel("Target")
    ax.set_ylabel("Best docking score")
    ax.set_title("Best docking score distribution by target")
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, out_dir, "fig_e2a_best_docking_score_distribution_by_target", formats)


def plot_target_summary(target_summary: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    df = target_summary.sort_values("mean_best_score")
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * len(df)), 4.8))
    x = np.arange(len(df))
    ax.bar(x, df["mean_best_score"], yerr=df["sd_best_score"], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(df["target"], rotation=45, ha="right")
    ax.set_xlabel("Target")
    ax.set_ylabel("Mean best docking score")
    ax.set_title("Target-level mean best docking score")
    save_figure(fig, out_dir, "fig_e2a_target_mean_best_docking_score", formats)


def plot_ligand_counts(target_summary: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    df = target_summary.sort_values("n_ligands", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.4 * len(df))))
    ax.barh(df["target"], df["n_ligands"])
    ax.set_xlabel("Number of ligands")
    ax.set_ylabel("Target")
    ax.set_title("Ligands with docking scores per target")
    save_figure(fig, out_dir, "fig_e2_data_ligand_counts_by_target", formats)


def plot_heatmap_matrix(
    matrix: pd.DataFrame,
    out_dir: Path,
    stem: str,
    title: str,
    colorbar_label: str,
    formats: list[str],
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    labels = list(matrix.index)
    arr = matrix.values.astype(float)

    fig_size = max(4.5, 0.45 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(arr, aspect="auto", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)

    # Write values in cells for small matrices
    if len(labels) <= 15:
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = arr[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    save_figure(fig, out_dir, stem, formats)


def plot_docking_heatmaps(pair_df: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    targets = sorted(set(pair_df["target_1"]) | set(pair_df["target_2"]))
    sim = pd.DataFrame(np.nan, index=targets, columns=targets, dtype=float)
    dist = pd.DataFrame(np.nan, index=targets, columns=targets, dtype=float)
    jacc = pd.DataFrame(np.nan, index=targets, columns=targets, dtype=float)

    for t in targets:
        sim.loc[t, t] = 1.0
        dist.loc[t, t] = 0.0
        jacc.loc[t, t] = 1.0

    for _, row in pair_df.iterrows():
        a, b = row["target_1"], row["target_2"]
        sim.loc[a, b] = sim.loc[b, a] = row["spearman_score_correlation"]
        dist.loc[a, b] = dist.loc[b, a] = row["docking_distance"]
        jacc.loc[a, b] = jacc.loc[b, a] = row["topk_jaccard_overlap"]

    plot_heatmap_matrix(
        sim,
        out_dir,
        "fig_e2b_docking_profile_spearman_similarity_matrix",
        "Docking profile similarity between targets",
        "Spearman correlation",
        formats,
        vmin=-1,
        vmax=1,
    )
    plot_heatmap_matrix(
        dist,
        out_dir,
        "fig_e2b_docking_profile_distance_matrix",
        "Docking profile distance between targets",
        "1 - Spearman correlation",
        formats,
        vmin=0,
        vmax=2,
    )
    plot_heatmap_matrix(
        jacc,
        out_dir,
        "fig_e2b_topk_overlap_matrix",
        "Top-k docked ligand overlap between targets",
        "Jaccard overlap",
        formats,
        vmin=0,
        vmax=1,
    )


def plot_docking_vs_alignment(comparison: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    df = comparison.dropna(subset=["alignment_distance", "docking_distance"]).copy()
    if df.empty:
        return
    stats_dict = compute_basic_stats(df["alignment_distance"], df["docking_distance"])

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(df["alignment_distance"], df["docking_distance"], alpha=0.85)
    add_regression_line(ax, df["alignment_distance"], df["docking_distance"])
    annotate_points(ax, df, "alignment_distance", "docking_distance")
    ax.set_xlabel("Alignment distance")
    ax.set_ylabel("Docking profile distance")
    title = "Docking distance vs alignment distance"
    if not pd.isna(stats_dict["pearson_r"]):
        title += f"\nPearson r={stats_dict['pearson_r']:.3f}, Spearman ρ={stats_dict['spearman_rho']:.3f}, n={stats_dict['n']}"
    elif stats_dict["n"] < 3:
        title += f"\nOnly n={stats_dict['n']} target pairs; correlation is not stable"
    ax.set_title(title)
    ax.legend(loc="best")
    save_figure(fig, out_dir, "fig_e2c_docking_distance_vs_alignment_distance", formats)


def plot_docking_vs_js(comparison: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if "js_divergence" not in comparison.columns:
        return
    df = comparison.dropna(subset=["js_divergence", "docking_distance"]).copy()
    if df.empty:
        return
    stats_dict = compute_basic_stats(df["js_divergence"], df["docking_distance"])

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.scatter(df["js_divergence"], df["docking_distance"], alpha=0.85)
    add_regression_line(ax, df["js_divergence"], df["docking_distance"])
    annotate_points(ax, df, "js_divergence", "docking_distance")
    ax.set_xlabel("Ligand-distribution JS divergence")
    ax.set_ylabel("Docking profile distance")
    title = "Docking distance vs ligand-distribution divergence"
    if not pd.isna(stats_dict["pearson_r"]):
        title += f"\nPearson r={stats_dict['pearson_r']:.3f}, Spearman ρ={stats_dict['spearman_rho']:.3f}, n={stats_dict['n']}"
    elif stats_dict["n"] < 3:
        title += f"\nOnly n={stats_dict['n']} target pairs; correlation is not stable"
    ax.set_title(title)
    ax.legend(loc="best")
    save_figure(fig, out_dir, "fig_e2d_docking_distance_vs_js_divergence", formats)


def plot_top_ligands_heatmap(agg: pd.DataFrame, out_dir: Path, formats: list[str], top_n: int = 30) -> None:
    # Pick globally best ligands by minimum score, then show target scores.
    global_top = (
        agg.groupby("ligand_id")["best_score"]
        .min()
        .sort_values()
        .head(top_n)
        .index
    )
    matrix = (
        agg[agg["ligand_id"].isin(global_top)]
        .pivot_table(index="ligand_id", columns="target", values="best_score", aggfunc="min")
        .loc[global_top]
    )
    if matrix.empty:
        return

    fig_w = max(5.5, 0.5 * len(matrix.columns) + 2)
    fig_h = max(5.5, 0.25 * len(matrix.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix.values.astype(float), aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Best docking score")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=7)
    ax.set_xlabel("Target")
    ax.set_ylabel("Ligand")
    ax.set_title(f"Best docking scores for top {len(matrix)} ligands")
    save_figure(fig, out_dir, "fig_e2_top_global_ligands_score_heatmap", formats)


# -------------------------
# Report
# -------------------------
def write_report(
    out_dir: Path,
    alignment: pd.DataFrame,
    pose_df: pd.DataFrame,
    agg: pd.DataFrame,
    pair_df: pd.DataFrame,
    comparison: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = []
    lines.append("# Docking vs alignment analysis report")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- Alignment file: `{args.alignment_csv}`")
    lines.append(f"- Pose score file: `{args.pose_scores_csv}`")
    lines.append("")
    lines.append("## Data summary")
    lines.append(f"- Alignment target pairs: {len(alignment)}")
    lines.append(f"- Pose-level docking rows: {len(pose_df)}")
    lines.append(f"- Ligand-target rows after aggregation: {len(agg)}")
    lines.append(f"- Unique ligands: {agg['ligand_id'].nunique()}")
    lines.append(f"- Docked targets: {', '.join(sorted(agg['target'].unique()))}")
    lines.append(f"- Target pairs with docking profiles: {len(pair_df)}")
    lines.append(f"- Target pairs with both alignment and docking: {len(comparison)}")
    lines.append("")
    lines.append("## Method")
    lines.append(
        "Pose-level docking scores were aggregated to one best score per ligand-target pair "
        "by taking the minimum docking score, assuming lower scores indicate better predicted binding."
    )
    lines.append(
        "For each target pair, docking profile similarity was calculated as the Spearman correlation "
        "between their best-score vectors across ligands available for both targets."
    )
    lines.append(
        "Docking profile distance was defined as `1 - Spearman correlation`. "
        "Alignment scores were converted into alignment distances using the selected normalization method."
    )
    lines.append("")
    lines.append("## Correlation summary")
    if not correlation_summary.empty:
        lines.append(correlation_summary.to_markdown(index=False))
    else:
        lines.append("No correlation summary could be computed.")
    lines.append("")
    lines.append("## Important interpretation note")
    lines.append(
        "A reliable docking-vs-alignment correlation requires several target pairs. "
        "If only 3 docked targets are available, there are only 3 target pairs, so the scatter plot is useful "
        "as a diagnostic but not yet strong statistical evidence."
    )
    lines.append("")
    lines.append("## Main generated files")
    for p in sorted(out_dir.iterdir()):
        if p.suffix.lower() in [".csv", ".png", ".pdf"]:
            lines.append(f"- `{p.name}`")
    (out_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def create_correlation_summary(alignment: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    records = []

    if "js_divergence" in alignment.columns:
        df = alignment.dropna(subset=["alignment_distance", "js_divergence"])
        st = compute_basic_stats(df["alignment_distance"], df["js_divergence"])
        records.append(
            {
                "analysis": "alignment_distance_vs_js_divergence",
                "x": "alignment_distance",
                "y": "js_divergence",
                **st,
            }
        )

    df = comparison.dropna(subset=["alignment_distance", "docking_distance"])
    st = compute_basic_stats(df["alignment_distance"], df["docking_distance"])
    records.append(
        {
            "analysis": "alignment_distance_vs_docking_distance",
            "x": "alignment_distance",
            "y": "docking_distance",
            **st,
        }
    )

    if "js_divergence" in comparison.columns:
        df = comparison.dropna(subset=["js_divergence", "docking_distance"])
        st = compute_basic_stats(df["js_divergence"], df["docking_distance"])
        records.append(
            {
                "analysis": "js_divergence_vs_docking_distance",
                "x": "js_divergence",
                "y": "docking_distance",
                **st,
            }
        )

    return pd.DataFrame(records)


def create_latex_snippet(out_dir: Path, formats: list[str]) -> None:
    ext = "pdf" if "pdf" in formats else formats[0]
    text = rf"""
% Add these figures to the E2 docking section.

\begin{{figure}}[ht]
    \centering
    \begin{{subfigure}}[t]{{0.48\linewidth}}
        \centering
        \includegraphics[width=\linewidth]{{figures/fig_e2a_best_docking_score_distribution_by_target.{ext}}}
        \caption{{Best docking score distributions by target.}}
    \end{{subfigure}}
    \hfill
    \begin{{subfigure}}[t]{{0.48\linewidth}}
        \centering
        \includegraphics[width=\linewidth]{{figures/fig_e2b_docking_profile_spearman_similarity_matrix.{ext}}}
        \caption{{Docking profile similarity matrix.}}
    \end{{subfigure}}

    \begin{{subfigure}}[t]{{0.48\linewidth}}
        \centering
        \includegraphics[width=\linewidth]{{figures/fig_e2c_docking_distance_vs_alignment_distance.{ext}}}
        \caption{{Docking profile distance versus alignment distance.}}
    \end{{subfigure}}
    \hfill
    \begin{{subfigure}}[t]{{0.48\linewidth}}
        \centering
        \includegraphics[width=\linewidth]{{figures/fig_e2d_docking_distance_vs_js_divergence.{ext}}}
        \caption{{Docking profile distance versus ligand-distribution divergence.}}
    \end{{subfigure}}
    \caption{{Docking-based validation of the ligand-shape/pocket correspondence. Each point in panels C and D is a pair of targets. Docking distance is defined as one minus the Spearman correlation between target docking-score profiles over common ligands.}}
    \label{{fig:e2-docking-validation}}
\end{{figure}}
""".strip()
    (out_dir / "latex_e2_figure_snippet.tex").write_text(text + "\n", encoding="utf-8")


def create_review_pdf(out_dir: Path, stems: list[str], formats: list[str]) -> None:
    # Create a small PDF with available PNG figures for quick review.
    review_path = out_dir / "selected_figures_review.pdf"
    pngs = []
    for stem in stems:
        path = out_dir / f"{stem}.png"
        if path.exists():
            pngs.append(path)

    if not pngs:
        return

    with PdfPages(review_path) as pdf:
        for png in pngs:
            img = plt.imread(png)
            fig, ax = plt.subplots(figsize=(8.5, 6.5))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(png.stem, fontsize=11)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


# -------------------------
# Main
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create TaSC docking-vs-alignment plots and result tables."
    )
    parser.add_argument("--alignment-csv", required=True, help="CSV with target pairs and alignment scores/distances.")
    parser.add_argument("--pose-scores-csv", required=True, help="CSV with pose-level docking scores.")
    parser.add_argument("--out", default="docking_alignment_results", help="Output directory.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=["png", "pdf", "svg"],
        help="Figure formats to save.",
    )
    parser.add_argument(
        "--alignment-distance-method",
        choices=["minmax", "score_over_max"],
        default="minmax",
        help="How to convert alignment score to distance. Default: minmax = (max-score)/(max-min).",
    )
    parser.add_argument(
        "--min-common-ligands",
        type=int,
        default=20,
        help="Minimum ligands shared by two targets to compute docking profile correlation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of best ligands per target for top-k overlap.",
    )
    parser.add_argument(
        "--top-n-ligands-heatmap",
        type=int,
        default=30,
        help="Number of globally best ligands to show in score heatmap.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out)
    safe_mkdir(out_dir)

    alignment = load_alignment_table(Path(args.alignment_csv), distance_method=args.alignment_distance_method)
    pose_df = load_pose_scores(Path(args.pose_scores_csv))
    agg = aggregate_poses(pose_df)
    target_summary = compute_target_summary(agg)
    pair_df, score_matrix, z_matrix = compute_pairwise_docking(
        agg,
        min_common_ligands=args.min_common_ligands,
        top_k=args.top_k,
    )

    comparison = alignment.merge(
        pair_df,
        on="pair_key",
        how="inner",
        suffixes=("_alignment", "_docking"),
    )

    # Resolve target names after merge
    if "target_1_alignment" in comparison.columns:
        comparison["target_1"] = comparison["target_1_alignment"]
        comparison["target_2"] = comparison["target_2_alignment"]

    keep_first = [
        "pair_key", "target_1", "target_2",
        "alignment_score", "alignment_distance", "js_divergence",
        "n_common_ligands", "spearman_score_correlation", "docking_distance",
        "pearson_score_correlation", "topk_jaccard_overlap",
        "n_topk_intersection", "n_topk_union",
    ]
    existing_keep = [c for c in keep_first if c in comparison.columns]
    comparison = comparison[existing_keep].sort_values(["target_1", "target_2"])

    correlation_summary = create_correlation_summary(alignment, comparison)

    # Save tables
    alignment.to_csv(out_dir / "alignment_pair_table.csv", index=False)
    pose_df.to_csv(out_dir / "pose_scores_cleaned.csv", index=False)
    agg.to_csv(out_dir / "ligand_target_best_scores.csv", index=False)
    target_summary.to_csv(out_dir / "target_docking_summary.csv", index=False)
    pair_df.to_csv(out_dir / "docking_pairwise_distances.csv", index=False)
    comparison.to_csv(out_dir / "target_pair_alignment_docking_comparison.csv", index=False)
    correlation_summary.to_csv(out_dir / "correlation_summary.csv", index=False)
    score_matrix.to_csv(out_dir / "docking_score_matrix_best_scores.csv")
    z_matrix.to_csv(out_dir / "docking_score_matrix_z_normalized.csv")

    # Plots
    plot_alignment_vs_js(alignment, out_dir, args.formats)
    plot_score_distribution(agg, out_dir, args.formats)
    plot_target_summary(target_summary, out_dir, args.formats)
    plot_ligand_counts(target_summary, out_dir, args.formats)
    plot_docking_heatmaps(pair_df, out_dir, args.formats)
    plot_docking_vs_alignment(comparison, out_dir, args.formats)
    plot_docking_vs_js(comparison, out_dir, args.formats)
    plot_top_ligands_heatmap(agg, out_dir, args.formats, top_n=args.top_n_ligands_heatmap)

    create_latex_snippet(out_dir, args.formats)
    create_review_pdf(
        out_dir,
        [
            "fig_e2a_best_docking_score_distribution_by_target",
            "fig_e2b_docking_profile_spearman_similarity_matrix",
            "fig_e2c_docking_distance_vs_alignment_distance",
            "fig_e2d_docking_distance_vs_js_divergence",
            "fig_e1_alignment_distance_vs_js_divergence",
        ],
        args.formats,
    )

    write_report(out_dir, alignment, pose_df, agg, pair_df, comparison, correlation_summary, args)

    print(f"Done. Results written to: {out_dir}")
    print(f"Docked targets: {', '.join(sorted(agg['target'].unique()))}")
    print(f"Target pairs with both alignment and docking: {len(comparison)}")
    print("Main table:", out_dir / "target_pair_alignment_docking_comparison.csv")
    print("Main plot:", out_dir / "fig_e2c_docking_distance_vs_alignment_distance.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
