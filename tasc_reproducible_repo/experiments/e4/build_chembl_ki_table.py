#!/usr/bin/env python3
"""
Build a cross-subtype Ki table for E4 boundary/selectivity validation.

Input:
  - full_sdfs.csv or another ligand metadata file containing molecule ChEMBL IDs
  - target mapping CSV mapping your local target labels to ChEMBL target IDs

Output:
  - raw_chembl_ki_activities.csv: all retrieved Ki measurements
  - chembl_ki_activities.csv: one aggregated row per ligand-target, suitable for E4
  - ki_pairwise_selectivity_preview.csv: quick check of ligands with >=2 target Ki values

Install:
  pip install chembl_webresource_client pandas numpy

Example target_mapping_opioids.csv:
  target,target_chembl_id,target_name
  muopioid,CHEMBL233,Mu opioid receptor
  deltaopioid,CHEMBL236,Delta opioid receptor
  kappaopioid,CHEMBL237,Kappa opioid receptor
  sigmaopioid,CHEMBL287,Sigma opioid receptor

Run:
  python build_chembl_ki_table.py \
    --metadata-csv full_sdfs.csv \
    --target-map target_mapping_opioids.csv \
    --out chembl_5ht_or_opioid_ki
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


def first_existing(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def chunks(xs, n):
    xs = list(xs)
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


def norm_units_to_nm(value, units):
    """Return Ki in nM from ChEMBL standard_value/standard_units."""
    try:
        v = float(value)
    except Exception:
        return np.nan
    if not np.isfinite(v) or v <= 0:
        return np.nan
    u = str(units).strip().lower().replace(" ", "")
    if u in {"nm", "nanomolar", "nanomole", "nanomoles"}:
        return v
    if u in {"um", "µm", "μm", "micromolar", "micromole", "micromoles"}:
        return v * 1000.0
    if u in {"mm", "millimolar", "millimole", "millimoles"}:
        return v * 1_000_000.0
    if u in {"m", "molar", "mole", "moles"}:
        return v * 1_000_000_000.0
    if u in {"pm", "picomolar", "picomole", "picomoles"}:
        return v / 1000.0
    return np.nan


def pki_from_nm(ki_nm):
    ki_nm = float(ki_nm)
    if not np.isfinite(ki_nm) or ki_nm <= 0:
        return np.nan
    return 9.0 - math.log10(ki_nm)


def fetch_activities(molecule_ids, target_ids, batch_size=80, sleep_s=0.05):
    from chembl_webresource_client.new_client import new_client

    activity = new_client.activity
    fields = [
        "activity_id",
        "molecule_chembl_id",
        "target_chembl_id",
        "target_pref_name",
        "assay_chembl_id",
        "document_chembl_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "pchembl_value",
        "assay_type",
        "confidence_score",
        "data_validity_comment",
    ]

    rows = []
    target_string = ",".join(target_ids)
    for batch_no, mol_batch in enumerate(chunks(molecule_ids, batch_size), start=1):
        mol_string = ",".join(mol_batch)
        print(f"Fetching batch {batch_no}: {len(mol_batch)} molecules")
        q = (
            activity
            .filter(molecule_chembl_id__in=mol_string)
            .filter(target_chembl_id__in=target_string)
            .filter(standard_type="Ki")
            .only(fields)
        )
        try:
            rows.extend(list(q))
        except Exception as exc:
            print(f"WARNING: batch {batch_no} failed: {type(exc).__name__}: {exc}")
        time.sleep(sleep_s)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", required=True, help="CSV with ligand ChEMBL IDs, e.g. full_sdfs.csv")
    ap.add_argument("--target-map", required=True, help="CSV with columns target,target_chembl_id[,target_name]")
    ap.add_argument("--out", default="chembl_ki_table")
    ap.add_argument("--ligand-col", default=None, help="Ligand ChEMBL ID column. Default autodetects chembl_id/molecule_chembl_id/ligand_id")
    ap.add_argument("--metadata-target-col", default=None, help="Optional local target column in metadata, used only to restrict ligands to mapped target labels")
    ap.add_argument("--keep-relations", default="=", help="Comma-separated standard_relation values to keep for final aggregate; default '='")
    ap.add_argument("--min-confidence", type=float, default=None, help="Optional ChEMBL confidence_score cutoff, e.g. 7")
    ap.add_argument("--batch-size", type=int, default=80)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata_csv)
    ligand_col = args.ligand_col or first_existing(meta.columns, [
        "molecule_chembl_id", "chembl_id", "ligand_id", "compound_id", "chembl", "ID"
    ])
    if ligand_col is None:
        raise ValueError(f"Could not detect ligand ChEMBL ID column. Columns: {list(meta.columns)}")

    tmap = pd.read_csv(args.target_map)
    if not {"target", "target_chembl_id"}.issubset(tmap.columns):
        raise ValueError("--target-map must contain columns: target,target_chembl_id")

    # Optional restriction: only use ligands that appeared in the local metadata for mapped targets.
    target_col = args.metadata_target_col or first_existing(meta.columns, ["target", "target_name", "receptor"])
    if target_col and set(meta[target_col].astype(str)).intersection(set(tmap["target"].astype(str))):
        meta = meta[meta[target_col].astype(str).isin(set(tmap["target"].astype(str)))].copy()

    molecule_ids = (
        meta[ligand_col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s.str.match(r"^CHEMBL\d+$", na=False)]
        .drop_duplicates()
        .tolist()
    )
    target_ids = tmap["target_chembl_id"].dropna().astype(str).str.strip().drop_duplicates().tolist()

    print(f"Molecules to query: {len(molecule_ids):,}")
    print(f"Targets to query:   {len(target_ids):,} ({', '.join(target_ids)})")

    raw = fetch_activities(molecule_ids, target_ids, batch_size=args.batch_size)
    if raw.empty:
        print("No ChEMBL Ki activities found for this molecule/target set.")
        return 2

    raw.to_csv(out / "raw_chembl_ki_activities.csv", index=False)

    # Map ChEMBL target IDs back to local labels.
    id_to_local = dict(zip(tmap["target_chembl_id"].astype(str), tmap["target"].astype(str)))
    raw["target"] = raw["target_chembl_id"].astype(str).map(id_to_local)
    raw["ligand_id"] = raw["molecule_chembl_id"].astype(str)
    raw["ki_nM"] = [norm_units_to_nm(v, u) for v, u in zip(raw.get("standard_value", []), raw.get("standard_units", []))]
    raw["pKi"] = raw["ki_nM"].map(lambda x: pki_from_nm(x) if pd.notna(x) else np.nan)

    keep = raw.copy()
    keep = keep[keep["target"].notna()]
    keep = keep[keep["standard_type"].astype(str).str.upper() == "KI"]
    keep = keep[keep["ki_nM"].notna() & (keep["ki_nM"] > 0)]

    relations = {x.strip() for x in args.keep_relations.split(",") if x.strip()}
    if relations:
        keep = keep[keep["standard_relation"].astype(str).isin(relations)]
    if args.min_confidence is not None and "confidence_score" in keep.columns:
        keep["confidence_score_num"] = pd.to_numeric(keep["confidence_score"], errors="coerce")
        keep = keep[keep["confidence_score_num"] >= args.min_confidence]

    # Aggregate replicate measurements in log space: median pKi, then back-convert to Ki nM.
    group_cols = ["ligand_id", "target"]
    agg = (
        keep.groupby(group_cols, as_index=False)
        .agg(
            pKi=("pKi", "median"),
            n_measurements=("activity_id", "count"),
            target_chembl_id=("target_chembl_id", "first"),
            target_pref_name=("target_pref_name", "first"),
        )
    )
    agg["standard_value"] = np.power(10.0, 9.0 - agg["pKi"])
    agg["standard_type"] = "Ki"
    agg["standard_relation"] = "="
    agg["standard_units"] = "nM"

    # Attach SMILES if available from metadata.
    smiles_col = first_existing(meta.columns, ["canonical_smiles", "smiles", "SMILES", "isomeric_smiles"])
    if smiles_col:
        smi = meta[[ligand_col, smiles_col]].dropna().drop_duplicates(subset=[ligand_col])
        smi.columns = ["ligand_id", "canonical_smiles"]
        smi["ligand_id"] = smi["ligand_id"].astype(str)
        agg = agg.merge(smi, on="ligand_id", how="left")

    # E4 expects these columns; keep extras too.
    first_cols = ["ligand_id", "target", "standard_type", "standard_relation", "standard_value", "standard_units"]
    extra_cols = [c for c in agg.columns if c not in first_cols]
    agg = agg[first_cols + extra_cols].sort_values(["ligand_id", "target"])
    agg.to_csv(out / "chembl_ki_activities.csv", index=False)

    # Quick selectivity preview: ligands with >=2 measured subtype Ki values.
    rows = []
    for lig, g in agg.groupby("ligand_id"):
        if g["target"].nunique() < 2:
            continue
        gg = g.sort_values("pKi", ascending=False).reset_index(drop=True)
        best = gg.iloc[0]
        second = gg.iloc[1]
        delta_pki = float(best["pKi"] - second["pKi"])
        ratio = float(second["standard_value"] / best["standard_value"])
        rows.append({
            "ligand_id": lig,
            "best_target": best["target"],
            "second_target": second["target"],
            "best_Ki_nM": best["standard_value"],
            "second_Ki_nM": second["standard_value"],
            "delta_pKi": delta_pki,
            "Ki_ratio_second_over_best": ratio,
            "n_measured_targets": int(g["target"].nunique()),
        })
    pd.DataFrame(rows).sort_values("delta_pKi", ascending=False).to_csv(out / "ki_pairwise_selectivity_preview.csv", index=False)

    print("Wrote:")
    print(f"  {out / 'raw_chembl_ki_activities.csv'}")
    print(f"  {out / 'chembl_ki_activities.csv'}")
    print(f"  {out / 'ki_pairwise_selectivity_preview.csv'}")
    print(f"Rows in final Ki table: {len(agg):,}")
    print(f"Ligands with >=2 Ki targets: {sum(agg.groupby('ligand_id')['target'].nunique() >= 2):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
