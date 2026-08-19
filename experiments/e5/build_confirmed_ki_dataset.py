#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def first_existing(cols, names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n in cols:
            return n
        if n.lower() in low:
            return low[n.lower()]
    return None

def normalize_target(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    repl = {
        "5-HT1A": "5HT1A", "5-HT1B": "5HT1B", "5-HT1D": "5HT1D",
        "5-HT2A": "5HT2A", "5-HT2B": "5HT2B", "5-HT2C": "5HT2C",
        "5-HT4R": "5HT4", "5-HT5A": "5HT5A", "5-HT6R": "5HT6", "5-HT7R": "5HT7",
    }
    return repl.get(s, s)

def to_nm(value, unit):
    if pd.isna(value):
        return np.nan
    try:
        v = float(value)
    except Exception:
        return np.nan
    u = str(unit).strip().lower() if not pd.isna(unit) else "nm"
    if u in {"nm", "nanomolar", "nanomole", "nanomoles"}:
        return v
    if u in {"um", "µm", "μm", "micromolar"}:
        return v * 1000.0
    if u in {"pm", "picomolar"}:
        return v * 0.001
    if u in {"m", "molar"}:
        return v * 1e9
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ki-csv", required=True)
    ap.add_argument("--out", default="e5_confirmed_ki")
    ap.add_argument("--positive-ki-nm", type=float, default=1000.0)
    ap.add_argument("--negative-ki-nm", type=float, default=10000.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.ki_csv)

    if "standard_type" in df.columns:
        df = df[df["standard_type"].astype(str).str.lower().eq("ki")].copy()

    ligand_col = first_existing(df.columns, ["ligand_id", "molecule_chembl_id", "chembl_id", "ID", "compound_id"])
    smiles_col = first_existing(df.columns, ["canonical_smiles", "smiles", "SMILES", "isomeric_smiles"])
    target_col = first_existing(df.columns, ["target", "target_id", "target_name", "receptor"])
    value_col = first_existing(df.columns, ["ki_nM", "Ki_nM", "standard_value_nM", "standard_value", "value"])
    unit_col = first_existing(df.columns, ["standard_units", "standard_unit", "units", "unit"])

    if ligand_col is None or smiles_col is None or target_col is None or value_col is None:
        raise SystemExit(f"Missing required columns. Found: {list(df.columns)}")

    if unit_col is None:
        df["_unit"] = "nM"
        unit_col = "_unit"

    rows = pd.DataFrame({
        "ligand_id": df[ligand_col].astype(str),
        "canonical_smiles": df[smiles_col].astype(str),
        "target": df[target_col].map(normalize_target),
        "ki_nM_raw": [to_nm(v, u) for v, u in zip(df[value_col], df[unit_col])],
    })

    rows = rows.replace([np.inf, -np.inf], np.nan)
    rows = rows.dropna(subset=["ligand_id", "canonical_smiles", "target", "ki_nM_raw"])
    rows = rows[(rows["ligand_id"].str.len() > 0) & (rows["canonical_smiles"].str.len() > 0) & (rows["target"].str.len() > 0)]

    grouped = (
        rows.groupby(["ligand_id", "target"], as_index=False)
        .agg(
            canonical_smiles=("canonical_smiles", "first"),
            min_ki_nM=("ki_nM_raw", "min"),
            median_ki_nM=("ki_nM_raw", "median"),
            n_measurements=("ki_nM_raw", "size"),
        )
    )

    pos = grouped[grouped["min_ki_nM"] <= args.positive_ki_nm].copy()
    pos["label"] = 1
    pos["source"] = "experimentally_measured_binder"

    neg = grouped[grouped["min_ki_nM"] >= args.negative_ki_nm].copy()
    neg["label"] = 0
    neg["source"] = "experimentally_measured_weak_or_nonbinder"

    dataset = pd.concat([pos, neg], ignore_index=True)
    dataset = dataset[[
        "ligand_id", "canonical_smiles", "target", "label", "source",
        "min_ki_nM", "median_ki_nM", "n_measurements",
    ]]
    dataset = dataset.drop_duplicates(["ligand_id", "target", "label"])

    summary = (
        dataset.groupby(["target", "label"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "n_negatives", 1: "n_positives"})
        .reset_index()
    )

    dataset.to_csv(out / "e5_confirmed_ki_dataset.csv", index=False)
    summary.to_csv(out / "e5_confirmed_ki_summary_by_target.csv", index=False)

    print("saved:", out / "e5_confirmed_ki_dataset.csv")
    print("saved:", out / "e5_confirmed_ki_summary_by_target.csv")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
