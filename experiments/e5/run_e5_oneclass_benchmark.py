#!/usr/bin/env python3
"""
E5 one-class binder-vs-outlier benchmark for TaSC.

Goal
----
Test the claim: non-binders are distributional shape outliers.
For each target, train one-class models on known binders only and rank held-out
binders above decoys/non-binders.

Inputs
------
- e5_binder_decoy_dataset.csv with columns:
  ligand_id, canonical_smiles, target, label, source
- one or more descriptor CSVs with ligand_id + numeric descriptor columns.
- optional e5_train_val_test_split.csv with ligand_id,target,partition.

Examples
--------
python run_e5_oneclass_benchmark.py \
  --dataset-csv e5_data_from_scratch/e5_binder_decoy_dataset.csv \
  --descriptor morgan:e5_data_from_scratch/e5_morgan_descriptors.csv \
  --split-csv e5_data_from_scratch/e5_train_val_test_split.csv \
  --out e5_oneclass_results \
  --formats png pdf

Outputs
-------
- e5_oneclass_metrics.csv
- e5_oneclass_scores.csv
- e5_summary_by_descriptor_model.csv
- figures and LaTeX snippet
"""
from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.spatial.distance import cdist
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ----------------------------- IO helpers -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run E5 one-class binder/outlier benchmark.")
    p.add_argument("--dataset-csv", required=True, help="E5 binder/decoy CSV.")
    p.add_argument("--descriptor", action="append", required=True,
                   help="Descriptor spec name:path.csv. Repeat for Morgan/USR/BDC/etc.")
    p.add_argument("--split-csv", default=None, help="Optional split CSV with ligand_id,target,partition.")
    p.add_argument("--out", default="e5_oneclass_results", help="Output directory.")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    p.add_argument("--models", nargs="+", default=["centroid", "iforest"],
                   choices=["centroid", "iforest", "ocsvm"],
                   help="One-class models to run. ocsvm can be slow for large targets.")
    p.add_argument("--targets", nargs="*", default=None, help="Optional target subset.")
    p.add_argument("--max-train-positives", type=int, default=5000,
                   help="Subsample positive train ligands per target/model for speed/reproducibility.")
    p.add_argument("--n-estimators", type=int, default=150, help="IsolationForest trees.")
    p.add_argument("--ocsvm-nu", type=float, default=0.05)
    p.add_argument("--ocsvm-gamma", default="scale")
    p.add_argument("--pca-components", type=int, default=128,
                   help="Reduce very high-dimensional descriptors before iforest/ocsvm. 0 disables PCA.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-train-positives", type=int, default=30)
    p.add_argument("--min-test-positives", type=int, default=10)
    p.add_argument("--min-test-negatives", type=int, default=50)
    return p.parse_args()


def read_dataset(path: Path, split_csv: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"ligand_id", "target", "label", "source"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["ligand_id"] = df["ligand_id"].astype(str)
    df["target"] = df["target"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    if split_csv:
        sp = pd.read_csv(split_csv)
        sp["ligand_id"] = sp["ligand_id"].astype(str)
        sp["target"] = sp["target"].astype(str)
        if "partition" not in sp.columns:
            raise ValueError("split CSV must contain a partition column")
        df = df.merge(sp[["ligand_id", "target", "partition"]], on=["ligand_id", "target"], how="left")
    if "partition" not in df.columns or df["partition"].isna().all():
        # Fallback: train on 70% positives and test on held-out positives + all negatives.
        rng = np.random.default_rng(42)
        df["partition"] = "test"
        for target, g in df[df["label"] == 1].groupby("target"):
            idx = g.index.to_numpy()
            rng.shuffle(idx)
            n_train = max(1, int(round(0.7 * len(idx))))
            n_val = max(0, int(round(0.15 * len(idx))))
            df.loc[idx[:n_train], "partition"] = "train"
            df.loc[idx[n_train:n_train + n_val], "partition"] = "val"
            df.loc[idx[n_train + n_val:], "partition"] = "test"
    df["partition"] = df["partition"].fillna("test").astype(str)
    return df


def read_descriptor(spec: str, ligand_ids: Iterable[str]) -> tuple[str, pd.DataFrame]:
    if ":" not in spec:
        raise ValueError("--descriptor must be name:path.csv")
    name, path = spec.split(":", 1)
    desc = pd.read_csv(path)
    if "ligand_id" not in desc.columns:
        raise ValueError(f"Descriptor {path} must contain ligand_id column")
    desc["ligand_id"] = desc["ligand_id"].astype(str)
    numeric_cols = [c for c in desc.columns if c != "ligand_id" and pd.api.types.is_numeric_dtype(desc[c])]
    if not numeric_cols:
        # try coercing all non-id columns
        for c in desc.columns:
            if c != "ligand_id":
                desc[c] = pd.to_numeric(desc[c], errors="coerce")
        numeric_cols = [c for c in desc.columns if c != "ligand_id" and pd.api.types.is_numeric_dtype(desc[c])]
    if not numeric_cols:
        raise ValueError(f"Descriptor {path} has no numeric columns")
    desc = desc[["ligand_id"] + numeric_cols].drop_duplicates("ligand_id")
    needed = pd.DataFrame({"ligand_id": pd.Series(list(ligand_ids), dtype=str).drop_duplicates()})
    aligned = needed.merge(desc, on="ligand_id", how="left")
    missing_frac = aligned[numeric_cols].isna().all(axis=1).mean()
    if missing_frac > 0:
        warnings.warn(f"Descriptor {name}: {missing_frac:.1%} ligands missing descriptor rows; filled with zeros.")
    aligned[numeric_cols] = aligned[numeric_cols].fillna(0.0)
    return name, aligned


# ----------------------------- metrics -----------------------------


def enrichment_factor(y_true: np.ndarray, score: np.ndarray, fraction: float) -> float:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    n = len(y)
    n_act = int(y.sum())
    if n == 0 or n_act == 0:
        return np.nan
    k = max(1, int(math.ceil(fraction * n)))
    top = np.argsort(-s)[:k]
    hit_rate_top = y[top].sum() / k
    hit_rate_all = n_act / n
    return float(hit_rate_top / hit_rate_all) if hit_rate_all > 0 else np.nan


def bedroc_score(y_true: np.ndarray, score: np.ndarray, alpha: float = 20.0) -> float:
    """BEDROC if RDKit scoring is available; otherwise normalized exponential early enrichment.

    Both variants range roughly 0..1 and reward actives ranked early. The fallback is
    BEDROC-style and is clearly documented in the report.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score).astype(float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return np.nan
    try:
        from rdkit.ML.Scoring import Scoring
        data = [(float(sc), int(yy)) for sc, yy in zip(s, y)]
        data.sort(key=lambda t: t[0], reverse=True)
        return float(Scoring.CalcBEDROC(data, 1, alpha))
    except Exception:
        # Fallback: exact min-max normalized exponential sum over active ranks.
        order = np.argsort(-s)
        y_sorted = y[order]
        N = len(y_sorted)
        n_act = int(y_sorted.sum())
        ranks0 = np.flatnonzero(y_sorted == 1)  # 0-based, lower is better
        if N <= 1 or n_act <= 0 or n_act >= N:
            return np.nan
        weights = np.exp(-alpha * np.arange(N) / max(N - 1, 1))
        raw = float(weights[ranks0].sum())
        ideal = float(weights[:n_act].sum())
        worst = float(weights[N - n_act:].sum())
        denom = ideal - worst
        if denom <= 0:
            return np.nan
        return float((raw - worst) / denom)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


def safe_auprc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if np.asarray(y_true).sum() == 0:
            return np.nan
        return float(average_precision_score(y_true, score))
    except Exception:
        return np.nan


def metric_dict(y_true: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "AUROC": safe_auc(y_true, score),
        "AUPRC": safe_auprc(y_true, score),
        "EF1%": enrichment_factor(y_true, score, 0.01),
        "EF5%": enrichment_factor(y_true, score, 0.05),
        "BEDROC20": bedroc_score(y_true, score, 20.0),
        "n_test": int(len(y_true)),
        "n_test_actives": int(np.asarray(y_true).sum()),
        "n_test_decoys": int((np.asarray(y_true) == 0).sum()),
    }


# ----------------------------- models -----------------------------


def prepare_features(X_train: np.ndarray, X_test: np.ndarray, descriptor_name: str, pca_components: int, seed: int):
    # Morgan is sparse/binary. Scaling can make OCSVM/IForest better; centroid distance also benefits.
    scaler = StandardScaler(with_mean=False if descriptor_name.lower() == "morgan" else True)
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    pca = None
    if pca_components and pca_components > 0 and Xtr.shape[1] > pca_components and len(Xtr) > pca_components + 1:
        pca = PCA(n_components=min(pca_components, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=seed)
        Xtr = pca.fit_transform(Xtr)
        Xte = pca.transform(Xte)
    return np.asarray(Xtr, dtype=np.float32), np.asarray(Xte, dtype=np.float32)


def fit_score_model(model_name: str, X_train_pos: np.ndarray, X_test: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if model_name == "centroid":
        center = np.mean(X_train_pos, axis=0, keepdims=True)
        # Higher score = closer to positive cloud.
        dist = cdist(X_test, center, metric="euclidean").ravel()
        return -dist
    if model_name == "iforest":
        model = IsolationForest(
            n_estimators=args.n_estimators,
            contamination="auto",
            random_state=args.seed,
            n_jobs=-1,
        )
        model.fit(X_train_pos)
        return model.decision_function(X_test).astype(float)
    if model_name == "ocsvm":
        model = OneClassSVM(kernel="rbf", nu=args.ocsvm_nu, gamma=args.ocsvm_gamma)
        model.fit(X_train_pos)
        return model.decision_function(X_test).ravel().astype(float)
    raise ValueError(model_name)


# ----------------------------- evaluation -----------------------------


def evaluate_descriptor(df: pd.DataFrame, desc: pd.DataFrame, descriptor_name: str, args: argparse.Namespace):
    feature_cols = [c for c in desc.columns if c != "ligand_id"]
    data = df.merge(desc, on="ligand_id", how="left")
    data[feature_cols] = data[feature_cols].fillna(0.0)
    if args.targets:
        data = data[data["target"].isin(args.targets)].copy()

    metrics_rows = []
    score_rows = []
    rng = np.random.default_rng(args.seed)

    for target, g in data.groupby("target", sort=True):
        train_pos = g[(g["partition"] == "train") & (g["label"] == 1)]
        test = g[g["partition"] == "test"].copy()
        n_train = len(train_pos)
        n_test_pos = int((test["label"] == 1).sum())
        n_test_neg = int((test["label"] == 0).sum())
        if n_train < args.min_train_positives or n_test_pos < args.min_test_positives or n_test_neg < args.min_test_negatives:
            warnings.warn(
                f"Skipping {target}/{descriptor_name}: insufficient data "
                f"train_pos={n_train}, test_pos={n_test_pos}, test_neg={n_test_neg}"
            )
            continue
        if n_train > args.max_train_positives:
            train_pos = train_pos.sample(args.max_train_positives, random_state=args.seed)

        X_train_raw = train_pos[feature_cols].to_numpy(dtype=np.float32)
        X_test_raw = test[feature_cols].to_numpy(dtype=np.float32)
        X_train, X_test = prepare_features(X_train_raw, X_test_raw, descriptor_name, args.pca_components, args.seed)
        y_test = test["label"].to_numpy(dtype=int)

        for model_name in args.models:
            try:
                score = fit_score_model(model_name, X_train, X_test, args)
            except Exception as exc:
                warnings.warn(f"Model failed for {target}/{descriptor_name}/{model_name}: {type(exc).__name__}: {exc}")
                continue

            base = {
                "target": target,
                "descriptor": descriptor_name,
                "model": model_name,
                "negative_source": "all_decoys",
                "n_train_positives": int(len(train_pos)),
                "n_features_raw": int(X_train_raw.shape[1]),
                "n_features_model": int(X_train.shape[1]),
            }
            metrics_rows.append({**base, **metric_dict(y_test, score)})

            # Per negative source while keeping the same positive test rows.
            pos_mask = test["label"].to_numpy(dtype=int) == 1
            for source in sorted(test.loc[test["label"] == 0, "source"].astype(str).unique()):
                m = pos_mask | ((test["label"].to_numpy(dtype=int) == 0) & (test["source"].astype(str).to_numpy() == source))
                if m.sum() and len(np.unique(y_test[m])) == 2:
                    metrics_rows.append({**base, "negative_source": source, **metric_dict(y_test[m], score[m])})

            sr = test[["ligand_id", "canonical_smiles", "target", "label", "source", "partition"]].copy()
            sr["descriptor"] = descriptor_name
            sr["model"] = model_name
            sr["binder_likeness_score"] = score
            score_rows.append(sr)

    return metrics_rows, score_rows


# ----------------------------- plots/reports -----------------------------


def plot_metric_bars(summary: pd.DataFrame, metric: str, out_dir: Path, formats: list[str]) -> None:
    clean = summary.dropna(subset=[metric]).copy()
    if clean.empty:
        return
    clean["label"] = clean["descriptor"].astype(str) + " / " + clean["model"].astype(str)
    clean = clean.sort_values(metric)
    fig, ax = plt.subplots(figsize=(7.6, max(3.2, 0.45 * len(clean) + 1.3)))
    ax.barh(clean["label"], clean[metric])
    ax.set_xlabel(f"Mean {metric} across targets")
    ax.set_ylabel("Descriptor / model")
    ax.set_title(f"E5 one-class outlier benchmark: {metric}")
    for i, v in enumerate(clean[metric].values):
        if np.isfinite(v):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    safe = metric.replace("%", "pct").replace("/", "_").replace(" ", "_")
    for fmt in formats:
        fig.savefig(out_dir / f"fig_e5_{safe.lower()}_by_descriptor_model.{fmt}", bbox_inches="tight")
    plt.close(fig)


def plot_score_distributions(scores: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if scores.empty or metrics.empty:
        return
    good = metrics[(metrics["negative_source"] == "all_decoys")].dropna(subset=["BEDROC20"])
    if good.empty:
        return
    best = good.sort_values("BEDROC20", ascending=False).iloc[0]
    m = (
        (scores["target"] == best["target"]) &
        (scores["descriptor"] == best["descriptor"]) &
        (scores["model"] == best["model"])
    )
    df = scores.loc[m].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    act = df.loc[df["label"] == 1, "binder_likeness_score"].dropna()
    dec = df.loc[df["label"] == 0, "binder_likeness_score"].dropna()
    ax.hist(dec, bins=50, alpha=0.55, density=True, label="Decoys / non-binders")
    ax.hist(act, bins=50, alpha=0.55, density=True, label="Held-out binders")
    ax.set_xlabel("Binder-likeness score")
    ax.set_ylabel("Density")
    ax.set_title(f"Score distributions: {best['target']} / {best['descriptor']} / {best['model']}")
    ax.legend(frameon=False)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"fig_e5_score_distributions_best_model.{fmt}", bbox_inches="tight")
    plt.close(fig)


def write_review_pdf(out_dir: Path) -> None:
    pngs = [
        "fig_e5_auroc_by_descriptor_model.png",
        "fig_e5_ef1pct_by_descriptor_model.png",
        "fig_e5_bedroc20_by_descriptor_model.png",
        "fig_e5_score_distributions_best_model.png",
    ]
    try:
        from PIL import Image
    except Exception:
        return
    with PdfPages(out_dir / "selected_figures_review.pdf") as pdf:
        for fn in pngs:
            path = out_dir / fn
            if not path.exists():
                continue
            img = Image.open(path)
            w, h = img.size
            fig = plt.figure(figsize=(min(11, w / 120), min(8.5, h / 120)))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def write_reports(df: pd.DataFrame, metrics: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    lines = [
        "# E5 one-class non-binder/outlier benchmark",
        "",
        "## What was evaluated",
        "For each target, a one-class model was trained only on known binders from the train partition. Held-out binders and decoys/non-binders in the test partition were ranked by binder-likeness score.",
        "",
        "## Input coverage",
        f"- Rows: {len(df):,}",
        f"- Targets: {df['target'].nunique():,}",
        f"- Unique ligands: {df['ligand_id'].nunique():,}",
        f"- Positives: {int((df['label'] == 1).sum()):,}",
        f"- Negatives: {int((df['label'] == 0).sum()):,}",
        "",
        "## Metrics",
        "AUROC and AUPRC measure global ranking. EF1%, EF5%, and BEDROC20 measure early enrichment, which is more relevant for virtual screening. When RDKit scoring is unavailable, BEDROC20 is computed with a normalized exponential early-recognition fallback.",
        "",
        "## Best descriptor/model combinations",
    ]
    if not summary.empty:
        for _, r in summary.sort_values("BEDROC20", ascending=False).head(10).iterrows():
            lines.append(
                f"- {r['descriptor']} / {r['model']}: mean BEDROC20={r['BEDROC20']:.3f}, "
                f"AUROC={r['AUROC']:.3f}, EF1%={r['EF1%']:.3f}, targets={int(r['n_targets'])}."
            )
    else:
        lines.append("No successful metrics were produced.")
    lines += [
        "",
        "## Important caveat",
        "The current negative labels are cross-target property-matched decoys, meaning 'not known active for this target'. They are useful for a first pipeline run, but the paper-grade E5 experiment should add DUD-E/DEKOIS decoys and docking-confirmed non-binders.",
        "",
        "## Main files",
        "- e5_oneclass_metrics.csv",
        "- e5_oneclass_scores.csv",
        "- e5_summary_by_descriptor_model.csv",
        "- selected_figures_review.pdf",
    ]
    (out_dir / "analysis_report.md").write_text("\n".join(lines))

    latex = r"""
% E5 table generated from e5_summary_by_descriptor_model.csv.
\begin{table}[ht]
\centering
\caption{One-class non-binder/outlier detection. Models are trained only on known binders for each target and evaluated on held-out binders versus decoys/non-binders.}
\label{tab:e5-oneclass-outliers}
\begin{tabular}{llrrrrr}
\hline
Descriptor & Model & AUROC & AUPRC & EF1\% & EF5\% & BEDROC20 \\
\hline
% Fill from e5_summary_by_descriptor_model.csv
\hline
\end{tabular}
\end{table}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.85\linewidth]{figures/fig_e5_bedroc20_by_descriptor_model.pdf}
    \caption{Non-binder outlier detection by one-class models trained on target-specific active ligand distributions. Bars report mean BEDROC20 across targets.}
    \label{fig:e5-bedroc}
\end{figure}
"""
    (out_dir / "latex_e5_snippet.tex").write_text(latex)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset(Path(args.dataset_csv), args.split_csv)
    if args.targets:
        df = df[df["target"].isin(args.targets)].copy()
    df.to_csv(out_dir / "e5_input_with_partitions.csv", index=False)

    descriptor_tables = []
    for spec in args.descriptor:
        name, desc = read_descriptor(spec, df["ligand_id"].unique())
        descriptor_tables.append((name, desc))

    all_metrics = []
    all_scores = []
    for name, desc in descriptor_tables:
        rows, scores = evaluate_descriptor(df, desc, name, args)
        all_metrics.extend(rows)
        all_scores.extend(scores)

    metrics = pd.DataFrame(all_metrics)
    metrics.to_csv(out_dir / "e5_oneclass_metrics.csv", index=False)
    scores = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    if not scores.empty:
        scores.to_csv(out_dir / "e5_oneclass_scores.csv", index=False)

    if not metrics.empty:
        summary = (
            metrics[metrics["negative_source"] == "all_decoys"]
            .groupby(["descriptor", "model"], as_index=False)
            .agg(
                AUROC=("AUROC", "mean"),
                AUPRC=("AUPRC", "mean"),
                **{"EF1%": ("EF1%", "mean"), "EF5%": ("EF5%", "mean"), "BEDROC20": ("BEDROC20", "mean")},
                n_targets=("target", "nunique"),
                total_test_actives=("n_test_actives", "sum"),
                total_test_decoys=("n_test_decoys", "sum"),
            )
        )
    else:
        summary = pd.DataFrame()
    summary.to_csv(out_dir / "e5_summary_by_descriptor_model.csv", index=False)

    for metric in ["AUROC", "EF1%", "EF5%", "BEDROC20"]:
        if metric in summary.columns:
            plot_metric_bars(summary, metric, out_dir, args.formats)
    plot_score_distributions(scores, metrics, out_dir, args.formats)
    write_review_pdf(out_dir)
    write_reports(df, metrics, summary, out_dir)

    print(f"Done. E5 one-class results saved to: {out_dir}")
    print(f"Metrics: {out_dir / 'e5_oneclass_metrics.csv'}")
    print(f"Summary: {out_dir / 'e5_summary_by_descriptor_model.csv'}")
    print(f"Report:  {out_dir / 'analysis_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
