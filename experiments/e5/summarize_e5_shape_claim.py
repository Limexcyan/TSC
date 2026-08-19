#!/usr/bin/env python3
"""Summarize E5 shape-vs-Morgan claim from e5_oneclass_metrics.csv."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Create manuscript-ready E5 descriptor/model summary.")
    p.add_argument("--metrics-csv", required=True, help="e5_oneclass_metrics.csv from run_e5_oneclass_benchmark.py")
    p.add_argument("--out", default="e5_shape_claim_summary")
    p.add_argument("--rank-by", default="BEDROC20", choices=["BEDROC20", "EF1%", "EF5%", "AUROC", "AUPRC"])
    p.add_argument("--baseline-descriptor", default="morgan")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.metrics_csv)
    # Normalize column names defensively.
    if "descriptor" not in df.columns or "model" not in df.columns:
        raise ValueError(f"Expected columns descriptor and model. Found: {list(df.columns)}")
    metrics = [c for c in ["AUROC", "AUPRC", "EF1%", "EF5%", "BEDROC20"] if c in df.columns]
    rank_col = args.rank_by
    if rank_col not in df.columns:
        raise ValueError(f"Missing rank column {rank_col}. Found: {list(df.columns)}")

    # Average across targets for each descriptor/model, preserving the benchmark script's naming.
    grouped = (df.groupby(["descriptor", "model"], as_index=False)[metrics]
                 .mean(numeric_only=True))
    grouped.to_csv(out / "e5_mean_metrics_by_descriptor_model.csv", index=False)

    # Best model per descriptor by selected early-recognition metric.
    best = (grouped.sort_values(["descriptor", rank_col], ascending=[True, False])
                   .groupby("descriptor", as_index=False)
                   .head(1)
                   .reset_index(drop=True))
    base_mask = best["descriptor"].str.lower() == args.baseline_descriptor.lower()
    if base_mask.any():
        base = best.loc[base_mask].iloc[0]
    else:
        # fallback: best Morgan row in grouped if descriptor was capitalised weirdly
        m = grouped[grouped["descriptor"].str.lower() == args.baseline_descriptor.lower()]
        if len(m) == 0:
            raise ValueError(f"Cannot find baseline descriptor {args.baseline_descriptor}")
        base = m.sort_values(rank_col, ascending=False).iloc[0]
    for m in metrics:
        best[f"delta_{m}_vs_{args.baseline_descriptor}"] = best[m] - float(base[m])

    def verdict(row):
        if row["descriptor"].lower() == args.baseline_descriptor.lower():
            return "baseline"
        d_bed = row.get(f"delta_BEDROC20_vs_{args.baseline_descriptor}", np.nan)
        d_ef1 = row.get(f"delta_EF1%_vs_{args.baseline_descriptor}", np.nan)
        if pd.notna(d_bed) and pd.notna(d_ef1) and d_bed > 0 and d_ef1 > 0:
            return "supports shape advantage"
        if pd.notna(d_bed) and d_bed > 0:
            return "mixed: BEDROC improves"
        if pd.notna(d_ef1) and d_ef1 > 0:
            return "mixed: EF1 improves"
        return "does not support advantage"
    best["verdict_vs_morgan"] = best.apply(verdict, axis=1)

    best.to_csv(out / "e5_best_model_per_descriptor_vs_morgan.csv", index=False)

    # Manuscript table: compact, rounded.
    cols = ["descriptor", "model"] + metrics + [f"delta_BEDROC20_vs_{args.baseline_descriptor}", f"delta_EF1%_vs_{args.baseline_descriptor}", "verdict_vs_morgan"]
    cols = [c for c in cols if c in best.columns]
    tab = best[cols].copy()
    for c in tab.columns:
        if pd.api.types.is_numeric_dtype(tab[c]):
            tab[c] = tab[c].map(lambda x: "--" if pd.isna(x) else f"{x:.3f}")
    tab.to_csv(out / "e5_manuscript_shape_claim_table.csv", index=False)
    try:
        tex = tab.to_latex(index=False, escape=False)
    except Exception:
        # Avoid requiring jinja2 by writing a basic tabular manually.
        header = " & ".join(tab.columns) + r" \\" + "\n"
        rows = "".join(" & ".join(map(str, row)) + r" \\" + "\n" for row in tab.to_numpy())
        tex = "\\begin{tabular}{" + "l" * len(tab.columns) + "}\n\\hline\n" + header + "\\hline\n" + rows + "\\hline\n\\end{tabular}\n"
    (out / "e5_manuscript_shape_claim_table.tex").write_text(tex)

    # Short report.
    shape_rows = best[best["descriptor"].str.lower() != args.baseline_descriptor.lower()]
    n_support = int((shape_rows["verdict_vs_morgan"] == "supports shape advantage").sum()) if len(shape_rows) else 0
    lines = [
        "# E5 shape-claim summary",
        "",
        f"Ranking metric for best model per descriptor: `{rank_col}`.",
        f"Baseline descriptor: `{args.baseline_descriptor}`.",
        "",
        f"Shape descriptors supporting advantage over Morgan on both BEDROC20 and EF1%: {n_support}/{len(shape_rows)}.",
        "",
        "Use `e5_manuscript_shape_claim_table.csv` or `.tex` in the paper.",
        "Interpret cautiously if negatives are cross-target property-matched decoys rather than experimentally confirmed non-binders.",
    ]
    (out / "analysis_report.md").write_text("\n".join(lines) + "\n")
    print("saved:", out)


if __name__ == "__main__":
    main()
