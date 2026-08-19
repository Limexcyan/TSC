#!/usr/bin/env python3
"""
Create smina-vs-Glide concordance plots and tables.

Inputs:
    1) Glide ligand-level aggregation CSV, e.g. aggregated_by_zinc.csv
    2) Glide pose-level CSV, e.g. pose_scores.csv
    3) smina ligand-level aggregation CSV, e.g. aggregated_by_zinc_smina.csv
    4) smina pose-level CSV, e.g. pose_scores_smina.csv

The most interpretable analysis is the structure-level concordance:
for each shared ligand and shared receptor structure, the script compares
the best Glide score against the best smina score.

Outputs:
    - matched_structure_scores.csv
    - concordance_summary_by_structure.csv
    - global_zinc_concordance.csv
    - global_zinc_concordance_summary.csv
    - topk_overlap_by_structure.csv
    - multiple PNG/PDF figures
    - selected_figures_review.pdf
    - analysis_report.md
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress


SCRIPT_VERSION = "2026-06-24-smina-glide-concordance-v4"


plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def ensure_columns(df: pd.DataFrame, required: Iterable[str], file_label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{file_label} is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def normalize_target(target: object) -> str:
    target = str(target).strip().upper()
    target = target.replace("-", "").replace(" ", "")
    return target


def extract_structure_key(row: pd.Series) -> str:
    """
    Extract a target-structure key from source_file if possible.

    Examples:
        zinc_cmds_docked_to_5ht1a_8PJK (1).sdf -> 5HT1A_8PJK
        out_5HT1A_8PJK_combined.sdf           -> 5HT1A_8PJK
    """
    source_file = str(row.get("source_file", ""))
    target = normalize_target(row.get("target", ""))

    patterns = [
        r"to_([A-Za-z0-9]+)_([A-Za-z0-9]{4})",
        r"out_([A-Za-z0-9]+)_([A-Za-z0-9]{4})",
        r"([A-Za-z0-9]+)_([A-Za-z0-9]{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, source_file)
        if match:
            receptor = normalize_target(match.group(1))
            pdb = match.group(2).upper()
            return f"{receptor}_{pdb}"

    return target if target else "UNKNOWN"


def read_pose_scores(path: Path, method: str) -> pd.DataFrame:
    # Read only columns needed for this analysis. This keeps large SDF-derived CSV files manageable.
    header = pd.read_csv(path, nrows=0)
    required = ["zinc_id", "target", "docking_score"]
    missing = [c for c in required if c not in header.columns]
    if missing:
        raise ValueError(
            f"{method} pose CSV is missing required columns {missing}. "
            f"Available columns: {list(header.columns)}"
        )

    usecols = [c for c in ["zinc_id", "target", "docking_score", "source_file", "pose_index_in_file"] if c in header.columns]
    df = pd.read_csv(path, usecols=usecols)

    if "source_file" not in df.columns:
        df["source_file"] = ""
    if "pose_index_in_file" not in df.columns:
        df["pose_index_in_file"] = np.nan

    df["target"] = df["target"].map(normalize_target)
    src = df["source_file"].astype(str)

    # Vectorized structure parsing. Avoid row-wise apply because smina pose files can be >1M rows.
    m_to = src.str.extract(r"to_([A-Za-z0-9]+)_([A-Za-z0-9]{4})", expand=True)
    m_out = src.str.extract(r"out_([A-Za-z0-9]+)_([A-Za-z0-9]{4})", expand=True)
    m_any = src.str.extract(r"([A-Za-z0-9]+)_([A-Za-z0-9]{4})", expand=True)

    receptor = m_to[0].fillna(m_out[0]).fillna(m_any[0]).fillna(df["target"])
    pdb = m_to[1].fillna(m_out[1]).fillna(m_any[1])

    receptor = receptor.map(normalize_target)
    df["structure_key"] = np.where(pdb.notna(), receptor + "_" + pdb.str.upper(), receptor)

    df["docking_score"] = pd.to_numeric(df["docking_score"], errors="coerce")
    df = df.dropna(subset=["zinc_id", "structure_key", "docking_score"])
    df["zinc_id"] = df["zinc_id"].astype(str)

    return df


def aggregate_pose_scores(df: pd.DataFrame, method: str) -> pd.DataFrame:
    # Lower docking scores are assumed to be better.
    best_rows = (
        df.sort_values("docking_score", ascending=True)
        .groupby(["zinc_id", "structure_key"], as_index=False)
        .first()
    )
    counts = (
        df.groupby(["zinc_id", "structure_key"], as_index=False)
        .agg(
            n_poses=("docking_score", "count"),
            mean_score=("docking_score", "mean"),
            median_score=("docking_score", "median"),
            std_score=("docking_score", "std"),
        )
    )
    out = best_rows[["zinc_id", "structure_key", "target", "source_file", "pose_index_in_file", "docking_score"]].merge(
        counts, on=["zinc_id", "structure_key"], how="left"
    )
    out = out.rename(
        columns={
            "docking_score": f"{method}_best_score",
            "target": f"{method}_target",
            "source_file": f"{method}_source_file",
            "pose_index_in_file": f"{method}_best_pose_index",
            "n_poses": f"{method}_n_poses",
            "mean_score": f"{method}_mean_score",
            "median_score": f"{method}_median_score",
            "std_score": f"{method}_std_score",
        }
    )
    return out


def read_global_aggregation(path: Path, method: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    ensure_columns(df, ["zinc_id", "best_score"], f"{method} aggregated-by-zinc CSV")
    df = df.copy()
    df["zinc_id"] = df["zinc_id"].astype(str)
    rename = {
        "best_score": f"{method}_global_best_score",
        "mean_score": f"{method}_global_mean_score",
        "median_score": f"{method}_global_median_score",
        "std_score": f"{method}_global_std_score",
        "n_poses": f"{method}_global_n_poses",
        "n_targets": f"{method}_global_n_targets",
        "targets": f"{method}_targets",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def corr_stats(x: pd.Series, y: pd.Series) -> dict[str, float]:
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(valid)
    out = {
        "n": n,
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_rho": np.nan,
        "spearman_p": np.nan,
        "slope": np.nan,
        "intercept": np.nan,
        "r_squared": np.nan,
    }
    if n < 3 or valid["x"].nunique() < 2 or valid["y"].nunique() < 2:
        return out

    pr = pearsonr(valid["x"], valid["y"])
    sr = spearmanr(valid["x"], valid["y"])
    lr = linregress(valid["x"], valid["y"])
    out.update(
        {
            "pearson_r": float(pr.statistic),
            "pearson_p": float(pr.pvalue),
            "spearman_rho": float(sr.statistic),
            "spearman_p": float(sr.pvalue),
            "slope": float(lr.slope),
            "intercept": float(lr.intercept),
            "r_squared": float(lr.rvalue**2),
        }
    )
    return out


def format_p(p: float) -> str:
    if pd.isna(p):
        return "NA"
    if p < 1e-300:
        return "<1e-300"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.3f}"


def add_fit_line(ax, x, y):
    valid = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid["x"].nunique() < 2:
        return
    lr = linregress(valid["x"], valid["y"])
    xs = np.linspace(valid["x"].min(), valid["x"].max(), 100)
    ax.plot(xs, lr.intercept + lr.slope * xs, linewidth=1.6)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_dir: Path,
    stem: str,
    title: str,
    x_label: str,
    y_label: str,
    formats: list[str],
    color_col: str | None = None,
    annotate_stats: bool = True,
):
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if color_col is None:
        ax.scatter(df[x_col], df[y_col], s=12, alpha=0.32, linewidths=0, rasterized=True)
    else:
        labels = sorted(df[color_col].dropna().astype(str).unique())
        for label in labels:
            sub = df[df[color_col].astype(str) == label]
            ax.scatter(sub[x_col], sub[y_col], s=12, alpha=0.28, linewidths=0, label=label, rasterized=True)
        ax.legend(frameon=False, markerscale=1.6)
    add_fit_line(ax, df[x_col], df[y_col])
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if annotate_stats:
        st = corr_stats(df[x_col], df[y_col])
        txt = (
            f"n = {st['n']}\n"
            f"Pearson r = {st['pearson_r']:.3f}, p = {format_p(st['pearson_p'])}\n"
            f"Spearman ρ = {st['spearman_rho']:.3f}, p = {format_p(st['spearman_p'])}\n"
            f"$R^2$ = {st['r_squared']:.3f}"
        )
        ax.text(
            0.03,
            0.97,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )

    save_figure(fig, out_dir, stem, formats)


def plot_structure_facets(matched: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    structures = sorted(matched["structure_key"].dropna().unique())
    if not structures:
        return

    n = len(structures)
    ncols = 2 if n > 1 else 1
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False)

    for ax, structure in zip(axes.ravel(), structures):
        sub = matched[matched["structure_key"] == structure]
        ax.scatter(sub["glide_best_score"], sub["smina_best_score"], s=10, alpha=0.25, linewidths=0, rasterized=True)
        add_fit_line(ax, sub["glide_best_score"], sub["smina_best_score"])
        st = corr_stats(sub["glide_best_score"], sub["smina_best_score"])
        ax.set_title(f"{structure}")
        ax.set_xlabel("Glide best docking score")
        ax.set_ylabel("smina best docking score")
        ax.text(
            0.03,
            0.97,
            f"n={st['n']}\nr={st['pearson_r']:.3f}\nρ={st['spearman_rho']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )

    for ax in axes.ravel()[len(structures):]:
        ax.axis("off")

    fig.suptitle("smina vs Glide concordance by receptor structure", y=0.995)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_smina_vs_glide_concordance_by_structure", formats)


def plot_correlation_bar(summary: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if summary.empty:
        return
    df = summary.sort_values("spearman_rho", ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.42 * len(df) + 1.5)))
    ax.barh(df["structure_key"], df["spearman_rho"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Spearman rank correlation between best scores")
    ax.set_ylabel("Receptor structure")
    ax.set_title("Rank concordance between smina and Glide")
    for i, row in enumerate(df.itertuples()):
        ax.text(row.spearman_rho, i, f"  n={row.n}", va="center", fontsize=8)
    save_figure(fig, out_dir, "fig_smina_vs_glide_spearman_by_structure", formats)


def topk_overlap(matched: pd.DataFrame, fractions: list[float]) -> pd.DataFrame:
    rows = []
    for structure, sub in matched.groupby("structure_key"):
        sub = sub.dropna(subset=["glide_best_score", "smina_best_score"]).copy()
        n = len(sub)
        if n == 0:
            continue
        for fraction in fractions:
            k = max(1, int(round(fraction * n)))
            glide_top = set(sub.nsmallest(k, "glide_best_score")["zinc_id"])
            smina_top = set(sub.nsmallest(k, "smina_best_score")["zinc_id"])
            inter = len(glide_top & smina_top)
            union = len(glide_top | smina_top)
            rows.append(
                {
                    "structure_key": structure,
                    "fraction": fraction,
                    "top_k": k,
                    "n_ligands": n,
                    "intersection": inter,
                    "jaccard_overlap": inter / union if union else np.nan,
                    "recall_of_glide_topk_in_smina_topk": inter / len(glide_top) if glide_top else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_topk_overlap(overlap: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if overlap.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for structure, sub in overlap.groupby("structure_key"):
        sub = sub.sort_values("fraction")
        ax.plot(sub["fraction"] * 100, sub["jaccard_overlap"], marker="o", label=structure)
    ax.set_xlabel("Top-ranked ligand fraction (%)")
    ax.set_ylabel("Jaccard overlap between smina and Glide top lists")
    ax.set_title("Agreement between top-ranked ligands")
    ax.legend(frameon=False)
    save_figure(fig, out_dir, "fig_smina_vs_glide_topk_overlap", formats)


def plot_score_distributions(matched: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    structures = sorted(matched["structure_key"].dropna().unique())
    if not structures:
        return
    labels = []
    data = []
    for structure in structures:
        sub = matched[matched["structure_key"] == structure]
        labels.extend([f"{structure}\nGlide", f"{structure}\nsmina"])
        data.extend([sub["glide_best_score"].dropna().values, sub["smina_best_score"].dropna().values])
    fig_width = max(8.0, 0.8 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    ax.boxplot(data, labels=labels, showmeans=True, showfliers=False)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Best docking score")
    ax.set_title("Best-score distributions for matched ligand–structure pairs")
    save_figure(fig, out_dir, "fig_smina_vs_glide_score_distributions_common_pairs", formats)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def write_report(
    out_dir: Path,
    matched: pd.DataFrame,
    structure_summary: pd.DataFrame,
    global_summary: pd.DataFrame,
    overlap: pd.DataFrame,
    glide_pose: pd.DataFrame,
    smina_pose: pd.DataFrame,
) -> None:
    common_structures = sorted(set(glide_pose["structure_key"]) & set(smina_pose["structure_key"]))
    glide_only = sorted(set(glide_pose["structure_key"]) - set(smina_pose["structure_key"]))
    smina_only = sorted(set(smina_pose["structure_key"]) - set(glide_pose["structure_key"]))

    lines = []
    lines.append("# smina vs Glide concordance analysis")
    lines.append("")
    lines.append(f"Script version: `{SCRIPT_VERSION}`")
    lines.append("")
    lines.append("## Input coverage")
    lines.append("")
    lines.append(f"- Glide pose rows: {len(glide_pose):,}")
    lines.append(f"- smina pose rows: {len(smina_pose):,}")
    lines.append(f"- Glide structures: {glide_pose['structure_key'].nunique()}")
    lines.append(f"- smina structures: {smina_pose['structure_key'].nunique()}")
    lines.append(f"- Shared structures used for structure-level concordance: {len(common_structures)}")
    lines.append("")
    lines.append("Shared structures:")
    for s in common_structures:
        lines.append(f"- {s}")
    if glide_only:
        lines.append("")
        lines.append("Glide-only structures:")
        for s in glide_only:
            lines.append(f"- {s}")
    if smina_only:
        lines.append("")
        lines.append("smina-only structures:")
        for s in smina_only:
            lines.append(f"- {s}")

    lines.append("")
    lines.append("## Main structure-level result")
    lines.append("")
    lines.append(
        "For every shared ligand–structure pair, poses were aggregated by taking the best "
        "score, i.e. the minimum score because lower docking scores are assumed to be better. "
        "Concordance was then measured between best Glide scores and best smina scores."
    )
    lines.append("")
    lines.append(f"- Matched ligand–structure pairs: {len(matched):,}")

    if not structure_summary.empty:
        lines.append("")
        lines.append("| Structure | n | Pearson r | Pearson p | Spearman rho | Spearman p | R2 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in structure_summary.sort_values("structure_key").itertuples():
            lines.append(
                f"| {row.structure_key} | {row.n:,} | {row.pearson_r:.3f} | {format_p(row.pearson_p)} | "
                f"{row.spearman_rho:.3f} | {format_p(row.spearman_p)} | {row.r_squared:.3f} |"
            )

    lines.append("")
    lines.append("## Global ligand-level result")
    lines.append("")
    lines.append(
        "The global ligand-level comparison uses the uploaded aggregated_by_zinc files. "
        "It compares each method's global best score per ligand. This is useful as a screen-level "
        "diagnostic, but it is less specific than the structure-level comparison because it is not conditioned "
        "on a single receptor structure."
    )
    if not global_summary.empty:
        row = global_summary.iloc[0]
        lines.append("")
        lines.append(f"- Matched ZINC IDs: {int(row['n']):,}")
        lines.append(f"- Pearson r: {row['pearson_r']:.3f}, p = {format_p(row['pearson_p'])}")
        lines.append(f"- Spearman rho: {row['spearman_rho']:.3f}, p = {format_p(row['spearman_p'])}")
        lines.append(f"- R2: {row['r_squared']:.3f}")

    lines.append("")
    lines.append("## Top-ranked ligand agreement")
    lines.append("")
    if not overlap.empty:
        top5 = overlap[np.isclose(overlap["fraction"], 0.05)]
        if not top5.empty:
            lines.append("Top-5% Jaccard overlap by structure:")
            for row in top5.sort_values("structure_key").itertuples():
                lines.append(f"- {row.structure_key}: {row.jaccard_overlap:.3f} ({row.intersection}/{int(row.top_k)} shared among each top-k list)")
        else:
            lines.append("Top-k overlap was computed and saved to `topk_overlap_by_structure.csv`.")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Positive Pearson/Spearman correlations indicate that ligands scoring well in Glide tend to score well in smina for the same receptor structure. "
        "This supports the E2 claim only to the extent that both methods show concordant target-specific ligand preference. "
        "The result should be reported together with coverage: how many structures and shared ligands were compared."
    )
    lines.append("")
    lines.append("## Generated figures")
    lines.append("")
    for p in sorted(out_dir.glob("fig_*.png")):
        lines.append(f"- `{p.name}`")

    (out_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def create_review_pdf(out_dir: Path) -> None:
    figures = [
        "fig_smina_vs_glide_global_best_score_concordance.png",
        "fig_smina_vs_glide_concordance_all_common_structures.png",
        "fig_smina_vs_glide_concordance_by_structure.png",
        "fig_smina_vs_glide_spearman_by_structure.png",
        "fig_smina_vs_glide_topk_overlap.png",
        "fig_smina_vs_glide_score_distributions_common_pairs.png",
    ]
    pdf_path = out_dir / "selected_figures_review.pdf"
    with PdfPages(pdf_path) as pdf:
        for fig_name in figures:
            img_path = out_dir / fig_name
            if not img_path.exists():
                continue
            img = plt.imread(img_path)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(fig_name, fontsize=11)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def zip_outputs(out_dir: Path) -> None:
    zip_path = out_dir.with_name(out_dir.name + "_package.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                z.write(path, arcname=str(path.relative_to(out_dir.parent)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create smina-vs-Glide concordance plots and tables.")
    parser.add_argument("--glide-agg", default="aggregated_by_zinc.csv", help="Glide aggregated_by_zinc CSV.")
    parser.add_argument("--glide-poses", default="pose_scores.csv", help="Glide pose_scores CSV.")
    parser.add_argument("--smina-agg", default="aggregated_by_zinc_smina.csv", help="smina aggregated_by_zinc CSV.")
    parser.add_argument("--smina-poses", default="pose_scores_smina.csv", help="smina pose_scores CSV.")
    parser.add_argument("--out", default="smina_glide_concordance_results", help="Output directory.")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"], help="Plot formats.")
    parser.add_argument("--topk-fractions", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.20], help="Top-k fractions for overlap.")
    parser.add_argument("--make-review-pdf", action="store_true", help="Also create selected_figures_review.pdf from generated PNGs.")
    parser.add_argument("--zip-results", action="store_true", help="Also create a ZIP package next to the output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        glide_pose = read_pose_scores(Path(args.glide_poses), "glide")
        smina_pose = read_pose_scores(Path(args.smina_poses), "smina")

        glide_struct = aggregate_pose_scores(glide_pose, "glide")
        smina_struct = aggregate_pose_scores(smina_pose, "smina")

        matched = glide_struct.merge(smina_struct, on=["zinc_id", "structure_key"], how="inner")
        if matched.empty:
            raise ValueError(
                "No matched ligand-structure pairs between Glide and smina. "
                "Check zinc_id and target/source_file naming."
            )
        matched.to_csv(out_dir / "matched_structure_scores.csv", index=False)

        # Structure-level summaries.
        rows = []
        for structure, sub in matched.groupby("structure_key"):
            st = corr_stats(sub["glide_best_score"], sub["smina_best_score"])
            st["structure_key"] = structure
            st["n_unique_zinc"] = sub["zinc_id"].nunique()
            st["glide_score_mean"] = sub["glide_best_score"].mean()
            st["smina_score_mean"] = sub["smina_best_score"].mean()
            rows.append(st)
        structure_summary = pd.DataFrame(rows)
        if not structure_summary.empty:
            first_cols = ["structure_key", "n", "n_unique_zinc"]
            structure_summary = structure_summary[first_cols + [c for c in structure_summary.columns if c not in first_cols]]
        structure_summary.to_csv(out_dir / "concordance_summary_by_structure.csv", index=False)

        # Global ligand-level summaries from aggregated_by_zinc.
        global_summary = pd.DataFrame()
        try:
            glide_global = read_global_aggregation(Path(args.glide_agg), "glide")
            smina_global = read_global_aggregation(Path(args.smina_agg), "smina")
            keep_g = [c for c in glide_global.columns if c.startswith("glide_") or c == "zinc_id"]
            keep_s = [c for c in smina_global.columns if c.startswith("smina_") or c == "zinc_id"]
            global_match = glide_global[keep_g].merge(smina_global[keep_s], on="zinc_id", how="inner")
            global_match.to_csv(out_dir / "global_zinc_concordance.csv", index=False)
            st = corr_stats(global_match["glide_global_best_score"], global_match["smina_global_best_score"])
            global_summary = pd.DataFrame([{**st, "comparison": "global_best_score_by_zinc"}])
            global_summary.to_csv(out_dir / "global_zinc_concordance_summary.csv", index=False)
        except Exception as exc:
            (out_dir / "global_zinc_concordance_error.txt").write_text(str(exc), encoding="utf-8")
            global_match = pd.DataFrame()

        # Top-k overlap.
        overlap = topk_overlap(matched, args.topk_fractions)
        overlap.to_csv(out_dir / "topk_overlap_by_structure.csv", index=False)

        # Coverage tables.
        coverage = pd.DataFrame(
            {
                "method": ["Glide", "smina"],
                "pose_rows": [len(glide_pose), len(smina_pose)],
                "unique_zinc": [glide_pose["zinc_id"].nunique(), smina_pose["zinc_id"].nunique()],
                "unique_structures": [glide_pose["structure_key"].nunique(), smina_pose["structure_key"].nunique()],
            }
        )
        coverage.to_csv(out_dir / "input_coverage_summary.csv", index=False)

        structure_coverage = pd.concat(
            [
                glide_pose.groupby("structure_key").agg(glide_pose_rows=("docking_score", "size"), glide_unique_zinc=("zinc_id", "nunique")),
                smina_pose.groupby("structure_key").agg(smina_pose_rows=("docking_score", "size"), smina_unique_zinc=("zinc_id", "nunique")),
            ],
            axis=1,
        ).fillna(0).reset_index()
        structure_coverage.to_csv(out_dir / "structure_coverage_summary.csv", index=False)

        # Figures.
        if not global_match.empty:
            plot_scatter(
                global_match,
                "glide_global_best_score",
                "smina_global_best_score",
                out_dir,
                "fig_smina_vs_glide_global_best_score_concordance",
                "Global ligand-level smina vs Glide concordance",
                "Glide global best docking score",
                "smina global best docking score",
                args.formats,
            )

        plot_scatter(
            matched,
            "glide_best_score",
            "smina_best_score",
            out_dir,
            "fig_smina_vs_glide_concordance_all_common_structures",
            "smina vs Glide concordance on shared receptor structures",
            "Glide best docking score",
            "smina best docking score",
            args.formats,
            color_col="structure_key",
        )
        plot_structure_facets(matched, out_dir, args.formats)
        plot_correlation_bar(structure_summary, out_dir, args.formats)
        plot_topk_overlap(overlap, out_dir, args.formats)
        plot_score_distributions(matched, out_dir, args.formats)

        write_report(out_dir, matched, structure_summary, global_summary, overlap, glide_pose, smina_pose)
        if args.make_review_pdf:
            create_review_pdf(out_dir)
        if args.zip_results:
            zip_outputs(out_dir)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Done. Results saved to: {out_dir}")
    print(f"Package: {out_dir.with_name(out_dir.name + '_package.zip')}")
    print(f"Matched ligand-structure pairs: {len(matched):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
