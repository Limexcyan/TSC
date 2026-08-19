#!/usr/bin/env python3
"""
E4 boundary/selectivity benchmark for TaSC.

Goal
----
Operationalise whether ligands near boundaries/overlaps between target ligand
shape-distributions are experimentally less selective, and whether ligands with
large predicted target-distribution margins have larger cross-subtype Ki ratios.

Typical use
-----------
1) Run E3 first and produce e3_test_predictions.csv. That file must contain
   per-ligand target probabilities in columns named prob__TARGET.
2) Provide a Ki activity table with ligand, target and Ki standard values.
3) This script joins the two and validates boundary/selectivity scores against
   cross-subtype Ki ratios.

Required activity columns, flexible names accepted
--------------------------------------------------
- ligand_id / chembl_id / molecule_chembl_id / ID
- target / target_id / target_name / pref_name
- standard_type / activity_type  (should contain Ki; optional if you pass --activity-type-col none)
- standard_value / Ki / value
- standard_units / units         (nM preferred; uM/M/mM/pM also converted)
- standard_relation / relation   (optional; exact '=' retained by default)

Outputs
-------
- e4_ligand_selectivity_table.csv
- e4_pairwise_selectivity_table.csv
- e4_boundary_selectivity_correlations.csv
- e4_threshold_classification.csv
- fig_e4_boundary_vs_ki_selectivity.*
- fig_e4_entropy_vs_ki_selectivity.*
- fig_e4_pair_boundary_vs_pair_delta_pki.*
- fig_e4_selective_vs_promiscuous_boundary.*
- latex_e4_snippet.tex
- analysis_report.md
"""
from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    for cand in candidates:
        pat = re.compile(re.escape(cand), re.I)
        for c in cols:
            if pat.search(c):
                return c
    return None


def normalize_target(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.replace(" ", "")
    repl = {
        "5H2A": "5HT2A",
        "5-HT1A": "5HT1A", "5-HT1B": "5HT1B", "5-HT1D": "5HT1D",
        "5-HT1E": "5HT1E", "5-HT1F": "5HT1F",
        "5-HT2A": "5HT2A", "5-HT2B": "5HT2B", "5-HT2C": "5HT2C",
        "5-HT4R": "5HT4", "5-HT4": "5HT4",
        "5-HT5A": "5HT5A", "5-HT6R": "5HT6", "5-HT6": "5HT6",
        "5-HT7R": "5HT7", "5-HT7": "5HT7",
    }
    return repl.get(s, s)


def load_predictions(path: Path, ligand_col: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    ligand_col = ligand_col or first_existing(df.columns, [
        "ligand_id", "chembl_id", "molecule_chembl_id", "compound_id", "ID", "id", "zinc_id"
    ])
    if ligand_col is None:
        raise ValueError(f"Cannot detect ligand column in predictions. Columns: {list(df.columns)}")
    prob_cols = [c for c in df.columns if c.startswith("prob__")]
    if not prob_cols:
        raise ValueError("Predictions file must contain probability columns named prob__TARGET, e.g. prob__5HT1A.")
    out = df[[ligand_col] + prob_cols].copy()
    out = out.rename(columns={ligand_col: "ligand_id"})
    out["ligand_id"] = out["ligand_id"].astype(str)
    # If multiple methods/splits are in one file, user should filter before this script.
    # But duplicate ligand IDs can happen; average probabilities to keep deterministic behavior.
    out[prob_cols] = out[prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    out = out.groupby("ligand_id", as_index=False)[prob_cols].mean()
    # Normalize defensively.
    mat = out[prob_cols].to_numpy(dtype=float)
    mat = np.clip(mat, 0.0, np.inf)
    rs = mat.sum(axis=1, keepdims=True)
    mat = np.divide(mat, rs, out=np.full_like(mat, 1.0 / len(prob_cols)), where=rs > 0)
    out[prob_cols] = mat
    return out


def unit_to_nm(value: float, unit: object) -> float:
    if pd.isna(value):
        return np.nan
    u = str(unit).strip().lower().replace("µ", "u").replace("μ", "u")
    if u in {"", "nan", "none"}:
        return float(value)  # assume nM if missing
    if u in {"nm", "nanomolar", "nanomole", "nanomoles", "nmol/l", "nm/l"}:
        return float(value)
    if u in {"um", "µm", "μm", "micromolar", "micromole", "umol/l"}:
        return float(value) * 1e3
    if u in {"mm", "millimolar", "mmol/l"}:
        return float(value) * 1e6
    if u in {"m", "molar", "mol/l"}:
        return float(value) * 1e9
    if u in {"pm", "picomolar", "pmol/l"}:
        return float(value) * 1e-3
    # Last-resort heuristic: many ChEMBL exports use 'nM'.
    return float(value)


def detect_activity_columns(df: pd.DataFrame, args: argparse.Namespace) -> dict[str, Optional[str]]:
    return {
        "ligand": args.activity_ligand_col or first_existing(df.columns, [
            "ligand_id", "chembl_id", "molecule_chembl_id", "compound_id", "ID", "id", "zinc_id"
        ]),
        "target": args.activity_target_col or first_existing(df.columns, [
            "target", "target_id", "target_name", "pref_name", "receptor", "protein", "assay_target"
        ]),
        "type": args.activity_type_col or first_existing(df.columns, [
            "standard_type", "activity_type", "type", "measurement_type"
        ]),
        "value": args.activity_value_col or first_existing(df.columns, [
            "standard_value", "activity_value", "Ki", "ki", "value"
        ]),
        "unit": args.activity_unit_col or first_existing(df.columns, [
            "standard_units", "activity_unit", "units", "unit", "standard_unit"
        ]),
        "relation": args.activity_relation_col or first_existing(df.columns, [
            "standard_relation", "relation", "activity_relation", "standard_flag"
        ]),
        "smiles": args.activity_smiles_col or first_existing(df.columns, [
            "canonical_smiles", "smiles", "SMILES", "isomeric_smiles"
        ]),
    }


def load_ki_activities(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    raw = pd.read_csv(path)
    cols = detect_activity_columns(raw, args)
    missing = [k for k in ["ligand", "target", "value"] if cols[k] is None]
    if missing:
        raise ValueError(f"Missing required activity columns {missing}. Available columns: {list(raw.columns)}")

    df = pd.DataFrame({
        "ligand_id": raw[cols["ligand"]].astype(str),
        "target": raw[cols["target"]].map(normalize_target),
        "standard_value": pd.to_numeric(raw[cols["value"]], errors="coerce"),
    })
    df["standard_units"] = raw[cols["unit"]].astype(str) if cols["unit"] else "nM"
    df["standard_type"] = raw[cols["type"]].astype(str) if cols["type"] else "Ki"
    df["standard_relation"] = raw[cols["relation"]].astype(str) if cols["relation"] else "="
    if cols["smiles"]:
        df["canonical_smiles"] = raw[cols["smiles"]].astype(str)

    if args.ki_only and cols["type"]:
        df = df[df["standard_type"].str.upper().str.strip().eq("KI")].copy()
    if not args.allow_inequalities and cols["relation"]:
        df = df[df["standard_relation"].astype(str).str.strip().isin(["=", "'='", "None", "nan", ""])].copy()

    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_value"] > 0].copy()
    df["ki_nM"] = [unit_to_nm(v, u) for v, u in zip(df["standard_value"], df["standard_units"])]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ki_nM"])
    df = df[df["ki_nM"] > 0].copy()
    df["pKi"] = 9.0 - np.log10(df["ki_nM"])
    # Collapse duplicate measurements per ligand-target using median pKi.
    grouped = df.groupby(["ligand_id", "target"], as_index=False).agg(
        ki_nM=("ki_nM", "median"),
        pKi=("pKi", "median"),
        n_measurements=("pKi", "size"),
    )
    return grouped


def ligand_level_selectivity(ki: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ligand_id, g in ki.groupby("ligand_id"):
        gg = g.sort_values("pKi", ascending=False).reset_index(drop=True)
        if len(gg) < 2:
            continue
        best = gg.iloc[0]
        second = gg.iloc[1]
        delta_pki = float(best["pKi"] - second["pKi"])
        ratio = float(second["ki_nM"] / best["ki_nM"]) if best["ki_nM"] > 0 else np.nan
        rows.append({
            "ligand_id": ligand_id,
            "best_ki_target": best["target"],
            "second_ki_target": second["target"],
            "best_ki_nM": best["ki_nM"],
            "second_ki_nM": second["ki_nM"],
            "best_pKi": best["pKi"],
            "second_pKi": second["pKi"],
            "delta_pKi_best_minus_second": delta_pki,
            "ki_selectivity_ratio_second_over_best": ratio,
            "n_ki_targets": int(len(gg)),
            "n_ki_measurements": int(gg["n_measurements"].sum()),
        })
    return pd.DataFrame(rows)


def add_distribution_boundary_scores(pred: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [c for c in pred.columns if c.startswith("prob__")]
    targets = [c.replace("prob__", "", 1) for c in prob_cols]
    P = pred[prob_cols].to_numpy(dtype=float)
    eps = 1e-12
    order = np.argsort(-P, axis=1)
    p1 = P[np.arange(len(P)), order[:, 0]]
    p2 = P[np.arange(len(P)), order[:, 1]] if P.shape[1] > 1 else np.zeros(len(P))
    entropy = -np.sum(np.clip(P, eps, 1.0) * np.log(np.clip(P, eps, 1.0)), axis=1) / math.log(max(P.shape[1], 2))
    out = pred.copy()
    out["top1_distribution_target"] = [targets[i] for i in order[:, 0]]
    out["top2_distribution_target"] = [targets[i] for i in order[:, 1]] if P.shape[1] > 1 else ""
    out["p_top1"] = p1
    out["p_top2"] = p2
    out["distribution_probability_margin"] = p1 - p2
    out["distribution_log_margin"] = np.log((p1 + eps) / (p2 + eps))
    out["boundary_score_top2_ratio"] = p2 / np.maximum(p1, eps)
    out["boundary_score_margin_inverse"] = 1.0 - (p1 - p2)
    out["distribution_entropy"] = entropy
    out["effective_number_of_targets"] = np.exp(entropy * math.log(max(P.shape[1], 2)))
    return out


def pairwise_selectivity(ki: pd.DataFrame, pred_scores: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [c for c in pred_scores.columns if c.startswith("prob__")]
    prob_targets = {c.replace("prob__", "", 1): c for c in prob_cols}
    pred_idx = pred_scores.set_index("ligand_id")
    rows = []
    eps = 1e-12
    for ligand_id, g in ki.groupby("ligand_id"):
        if ligand_id not in pred_idx.index or len(g) < 2:
            continue
        row = pred_idx.loc[ligand_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        gg = g.reset_index(drop=True)
        for i in range(len(gg)):
            for j in range(i + 1, len(gg)):
                a = gg.iloc[i]
                b = gg.iloc[j]
                ta, tb = str(a["target"]), str(b["target"])
                if ta not in prob_targets or tb not in prob_targets:
                    continue
                pa = float(row[prob_targets[ta]])
                pb = float(row[prob_targets[tb]])
                pki_a = float(a["pKi"])
                pki_b = float(b["pKi"])
                # Experimental pairwise selectivity magnitude.
                abs_delta = abs(pki_a - pki_b)
                # Distribution overlap between the two target probabilities: 1 at equal, 0 at one-sided.
                pair_boundary = 1.0 - abs(pa - pb) / max(pa + pb, eps)
                rows.append({
                    "ligand_id": ligand_id,
                    "target_a": ta,
                    "target_b": tb,
                    "pKi_a": pki_a,
                    "pKi_b": pki_b,
                    "abs_delta_pKi": abs_delta,
                    "ki_ratio_larger_over_smaller": 10 ** abs_delta,
                    "prob_a": pa,
                    "prob_b": pb,
                    "pair_probability_margin_abs": abs(pa - pb),
                    "pair_distribution_log_margin_abs": abs(math.log((pa + eps) / (pb + eps))),
                    "pair_boundary_score": pair_boundary,
                })
    return pd.DataFrame(rows)


def safe_corr(x, y) -> dict[str, float]:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    m = x.notna() & y.notna()
    if m.sum() < 3 or x[m].nunique() < 2 or y[m].nunique() < 2:
        return {"n": int(m.sum()), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}
    pr = pearsonr(x[m], y[m])
    sr = spearmanr(x[m], y[m])
    return {"n": int(m.sum()), "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue), "spearman_rho": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def add_correlations(lig: pd.DataFrame, pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tests = [
        ("ligand_boundary_top2_ratio_vs_negative_delta_pKi", lig, "boundary_score_top2_ratio", "negative_delta_pKi"),
        ("ligand_entropy_vs_negative_delta_pKi", lig, "distribution_entropy", "negative_delta_pKi"),
        ("ligand_margin_vs_delta_pKi", lig, "distribution_log_margin", "delta_pKi_best_minus_second"),
        ("best_vs_second_pred_margin_vs_delta_pKi", lig, "best_vs_second_pred_log_margin", "delta_pKi_best_minus_second"),
        ("pair_boundary_vs_negative_abs_delta_pKi", pair, "pair_boundary_score", "negative_abs_delta_pKi"),
        ("pair_margin_vs_abs_delta_pKi", pair, "pair_distribution_log_margin_abs", "abs_delta_pKi"),
    ]
    for name, df, xcol, ycol in tests:
        if df is None or df.empty or xcol not in df or ycol not in df:
            continue
        corr = safe_corr(df[xcol], df[ycol])
        corr.update({"test": name, "x": xcol, "y": ycol})
        rows.append(corr)
    return pd.DataFrame(rows)


def classification_table(lig: pd.DataFrame, promiscuous_delta: float, selective_delta: float) -> pd.DataFrame:
    rows = []
    if lig.empty:
        return pd.DataFrame()
    # Binary task: selective if delta >= selective_delta, promiscuous if delta <= promiscuous_delta; ignore gray zone.
    df = lig[(lig["delta_pKi_best_minus_second"] >= selective_delta) | (lig["delta_pKi_best_minus_second"] <= promiscuous_delta)].copy()
    if df.empty or df["delta_pKi_best_minus_second"].nunique() < 2:
        return pd.DataFrame()
    y = (df["delta_pKi_best_minus_second"] >= selective_delta).astype(int).to_numpy()
    predictors = [
        ("distribution_log_margin", True),
        ("best_vs_second_pred_log_margin", True),
        ("boundary_score_top2_ratio", False),
        ("distribution_entropy", False),
    ]
    for col, higher_means_selective in predictors:
        if col not in df:
            continue
        score = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(score)
        if mask.sum() < 3 or len(np.unique(y[mask])) < 2:
            continue
        s = score[mask] if higher_means_selective else -score[mask]
        rows.append({
            "predictor": col,
            "n": int(mask.sum()),
            "positive_class": f"selective_delta_pKi_ge_{selective_delta:g}",
            "negative_class": f"promiscuous_delta_pKi_le_{promiscuous_delta:g}",
            "roc_auc": float(roc_auc_score(y[mask], s)),
            "average_precision": float(average_precision_score(y[mask], s)),
        })
    return pd.DataFrame(rows)


def save_scatter(df: pd.DataFrame, x: str, y: str, path_base: Path, formats: list[str], title: str, xlabel: str, ylabel: str) -> None:
    if df.empty or x not in df or y not in df:
        return
    clean = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(clean) < 3:
        return
    corr = safe_corr(clean[x], clean[y])
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.scatter(clean[x], clean[y], alpha=0.7, s=22)
    if clean[x].nunique() > 1:
        coeff = np.polyfit(clean[x], clean[y], 1)
        xs = np.linspace(clean[x].min(), clean[x].max(), 100)
        ax.plot(xs, coeff[0] * xs + coeff[1], linestyle="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.text(0.02, 0.98, f"n={corr['n']}\nSpearman ρ={corr['spearman_rho']:.3f}\np={corr['spearman_p']:.2g}",
            transform=ax.transAxes, va="top", ha="left",
            bbox={"boxstyle": "round", "alpha": 0.15})
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(path_base.with_suffix(f".{fmt}"), bbox_inches="tight")
    plt.close(fig)


def save_group_box(lig: pd.DataFrame, out_dir: Path, formats: list[str], promiscuous_delta: float, selective_delta: float) -> None:
    if lig.empty:
        return
    df = lig.copy()
    df["class"] = "intermediate"
    df.loc[df["delta_pKi_best_minus_second"] <= promiscuous_delta, "class"] = "promiscuous"
    df.loc[df["delta_pKi_best_minus_second"] >= selective_delta, "class"] = "selective"
    groups = [df.loc[df["class"] == c, "boundary_score_top2_ratio"].dropna().to_numpy() for c in ["promiscuous", "intermediate", "selective"]]
    if sum(len(g) for g in groups) < 3:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.boxplot(groups, labels=[f"promiscuous\n≤{promiscuous_delta:g} ΔpKi", "intermediate", f"selective\n≥{selective_delta:g} ΔpKi"], showfliers=False)
    ax.set_ylabel("Distribution boundary score, p2/p1")
    ax.set_title("Boundary score by experimental Ki selectivity class")
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"fig_e4_selective_vs_promiscuous_boundary.{fmt}", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate E4 boundary/selectivity scores against cross-subtype Ki ratios.")
    ap.add_argument("--predictions-csv", required=True, help="E3 predictions with prob__TARGET columns, e.g. e3_test_predictions.csv")
    ap.add_argument("--ki-csv", required=True, help="Activity table with cross-subtype Ki standard values.")
    ap.add_argument("--out", default="e4_boundary_selectivity_results")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    ap.add_argument("--prediction-ligand-col", default=None)

    ap.add_argument("--activity-ligand-col", default=None)
    ap.add_argument("--activity-target-col", default=None)
    ap.add_argument("--activity-type-col", default=None)
    ap.add_argument("--activity-value-col", default=None)
    ap.add_argument("--activity-unit-col", default=None)
    ap.add_argument("--activity-relation-col", default=None)
    ap.add_argument("--activity-smiles-col", default=None)
    ap.add_argument("--ki-only", action="store_true", default=True, help="Keep only standard_type == Ki when type column exists. Default: true.")
    ap.add_argument("--include-non-ki", action="store_true", help="Disable Ki-only filter.")
    ap.add_argument("--allow-inequalities", action="store_true", help="Keep <, >, <=, >= relations. Default keeps exact values only.")
    ap.add_argument("--promiscuous-delta-pki", type=float, default=0.5, help="Promiscuous threshold: ΔpKi <= this value (~3.16x at 0.5).")
    ap.add_argument("--selective-delta-pki", type=float, default=1.0, help="Selective threshold: ΔpKi >= this value (10x at 1.0).")
    args = ap.parse_args()
    if args.include_non_ki:
        args.ki_only = False

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = load_predictions(Path(args.predictions_csv), args.prediction_ligand_col)
    pred_scores = add_distribution_boundary_scores(pred)
    pred_scores.to_csv(out_dir / "e4_distribution_boundary_scores.csv", index=False)

    ki = load_ki_activities(Path(args.ki_csv), args)
    ki.to_csv(out_dir / "e4_clean_ki_activity_table.csv", index=False)

    lig = ligand_level_selectivity(ki)
    if lig.empty:
        raise ValueError("No ligands have Ki measurements for at least two targets. E4 needs cross-subtype Ki values.")
    lig = lig.merge(pred_scores, on="ligand_id", how="inner")
    if lig.empty:
        raise ValueError("No overlap between prediction ligand IDs and Ki activity ligand IDs.")
    lig["negative_delta_pKi"] = -lig["delta_pKi_best_minus_second"]
    eps = 1e-12
    prob_cols = {c.replace("prob__", "", 1): c for c in lig.columns if c.startswith("prob__")}
    best_probs = []
    second_probs = []
    pred_log_margins = []
    pred_boundary = []
    for _, r in lig.iterrows():
        bt, st = str(r["best_ki_target"]), str(r["second_ki_target"])
        if bt in prob_cols and st in prob_cols:
            pb = float(r[prob_cols[bt]])
            ps = float(r[prob_cols[st]])
            best_probs.append(pb)
            second_probs.append(ps)
            pred_log_margins.append(math.log((pb + eps) / (ps + eps)))
            pred_boundary.append(1.0 - abs(pb - ps) / max(pb + ps, eps))
        else:
            best_probs.append(np.nan)
            second_probs.append(np.nan)
            pred_log_margins.append(np.nan)
            pred_boundary.append(np.nan)
    lig["prob_best_ki_target"] = best_probs
    lig["prob_second_ki_target"] = second_probs
    lig["best_vs_second_pred_log_margin"] = pred_log_margins
    lig["best_vs_second_boundary_score"] = pred_boundary
    lig.to_csv(out_dir / "e4_ligand_selectivity_table.csv", index=False)

    pair = pairwise_selectivity(ki, pred_scores)
    if not pair.empty:
        pair["negative_abs_delta_pKi"] = -pair["abs_delta_pKi"]
    pair.to_csv(out_dir / "e4_pairwise_selectivity_table.csv", index=False)

    corr = add_correlations(lig, pair)
    corr.to_csv(out_dir / "e4_boundary_selectivity_correlations.csv", index=False)

    clf = classification_table(lig, args.promiscuous_delta_pki, args.selective_delta_pki)
    clf.to_csv(out_dir / "e4_threshold_classification.csv", index=False)

    save_scatter(
        lig, "boundary_score_top2_ratio", "delta_pKi_best_minus_second",
        out_dir / "fig_e4_boundary_vs_ki_selectivity", args.formats,
        "Target-distribution boundary score vs experimental Ki selectivity",
        "Boundary score p2/p1; high = overlap/shared region",
        "Experimental selectivity ΔpKi = pKi(best) - pKi(second)",
    )
    save_scatter(
        lig, "distribution_entropy", "delta_pKi_best_minus_second",
        out_dir / "fig_e4_entropy_vs_ki_selectivity", args.formats,
        "Distribution entropy vs experimental Ki selectivity",
        "Distribution entropy; high = promiscuous/shared region",
        "Experimental selectivity ΔpKi",
    )
    save_scatter(
        lig, "best_vs_second_pred_log_margin", "delta_pKi_best_minus_second",
        out_dir / "fig_e4_pred_margin_vs_ki_selectivity", args.formats,
        "Predicted best-vs-second target margin vs Ki selectivity",
        "Predicted log probability margin for Ki-best vs Ki-second target",
        "Experimental selectivity ΔpKi",
    )
    if not pair.empty:
        save_scatter(
            pair, "pair_boundary_score", "abs_delta_pKi",
            out_dir / "fig_e4_pair_boundary_vs_pair_delta_pki", args.formats,
            "Pairwise target-boundary score vs pairwise Ki selectivity",
            "Pair boundary score between measured target pair; high = overlap",
            "|ΔpKi| between the two targets",
        )
    save_group_box(lig, out_dir, args.formats, args.promiscuous_delta_pki, args.selective_delta_pki)

    # Report.
    lines = ["# E4 boundary/selectivity benchmark", ""]
    lines.append("## Operational definitions")
    lines.append("- Boundary score: `p_top2 / p_top1`, where `p_top1` and `p_top2` are the two largest target-distribution probabilities for a ligand. Higher values indicate a ligand lies in an overlap/shared region.")
    lines.append("- Entropy: normalized entropy of the target probability vector. Higher values indicate less target-specific localisation.")
    lines.append("- Distributional selectivity margin: `log(p_top1 / p_top2)` or, for measured Ki targets, `log(p_best_Ki_target / p_second_Ki_target)`.")
    lines.append("- Experimental selectivity: `ΔpKi = pKi_best - pKi_second = log10(Ki_second / Ki_best)`.")
    lines.append("")
    lines.append("## Input coverage")
    lines.append(f"- Prediction ligands: {len(pred_scores):,}")
    lines.append(f"- Clean ligand-target Ki rows: {len(ki):,}")
    lines.append(f"- Ligands with Ki for at least two targets and prediction overlap: {len(lig):,}")
    lines.append(f"- Pairwise ligand-target comparisons: {len(pair):,}")
    lines.append("")
    if not corr.empty:
        lines.append("## Main correlations")
        for _, r in corr.iterrows():
            lines.append(f"- {r['test']}: n={int(r['n'])}, Pearson r={r['pearson_r']:.3f}, Spearman rho={r['spearman_rho']:.3f}, p={r['spearman_p']:.3g}")
    if not clf.empty:
        lines.append("")
        lines.append("## Selective-vs-promiscuous classification")
        for _, r in clf.iterrows():
            lines.append(f"- {r['predictor']}: n={int(r['n'])}, ROC-AUC={r['roc_auc']:.3f}, AP={r['average_precision']:.3f}")
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("E4 is supported when boundary/entropy scores correlate negatively with ΔpKi, and predicted distributional margins correlate positively with ΔpKi. In words: ligands in shared regions should have small Ki selectivity ratios, while ligands with one dominant target-distribution probability should have larger Ki ratios.")
    (out_dir / "analysis_report.md").write_text("\n".join(lines))

    latex = r"""
% E4 operationalisation and validation.
egin{table}[ht]
\\centering
\\caption{Boundary/selectivity validation against cross-subtype Ki ratios. Experimental selectivity is measured as $\\Delta pK_i=pK_{i,\\mathrm{best}}-pK_{i,\\mathrm{second}}=\\log_{10}(K_{i,\\mathrm{second}}/K_{i,\\mathrm{best}})$. Boundary scores are expected to correlate negatively with $\\Delta pK_i$, while distributional margins are expected to correlate positively.}
\\label{tab:e4-boundary-selectivity}
\\begin{tabular}{lrrrr}
\\hline
Score & n & Pearson $r$ & Spearman $\\rho$ & $p$ \\\\
\\hline
% Fill from e4_boundary_selectivity_correlations.csv
\\hline
\\end{tabular}
\\end{table}

\\begin{figure}[ht]
    \\centering
    \\includegraphics[width=0.72\\linewidth]{figures/fig_e4_boundary_vs_ki_selectivity.pdf}
    \\caption{Distribution-boundary score versus measured cross-subtype selectivity. High boundary scores indicate ligands lying in shared target-distribution regions; these are expected to have smaller $\\Delta pK_i$ and lower Ki selectivity ratios.}
    \\label{fig:e4-boundary-selectivity}
\\end{figure}
"""
    (out_dir / "latex_e4_snippet.tex").write_text(latex)

    print(f"Done. E4 results saved to: {out_dir}")
    print(f"Ligand table: {out_dir / 'e4_ligand_selectivity_table.csv'}")
    print(f"Correlations:  {out_dir / 'e4_boundary_selectivity_correlations.csv'}")
    print(f"Report:        {out_dir / 'analysis_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
