#!/usr/bin/env python3
"""
Build an E5 binder/decoy dataset from the existing ligand-target table.

This creates a first-pass, offline E5 benchmark without external DUD-E/DEKOIS downloads:
- positives: known active ligands from metadata
- negatives: cross-target property-matched decoys selected from ligands active for other targets,
  not known active for the query target

Caveat: cross-target decoys are "not known active for this target", not experimentally confirmed
non-binders. Use DUD-E/DEKOIS or docking-confirmed non-binders later for paper-grade negatives.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
except Exception as exc:  # pragma: no cover
    Chem = None
    DataStructs = None
    Descriptors = None
    rdMolDescriptors = None
    Crippen = None
    Lipinski = None


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def normalize_target(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    repl = {
        "5H2A": "5HT2A", "5-HT1A": "5HT1A", "5-HT1B": "5HT1B", "5-HT1D": "5HT1D",
        "5-HT2A": "5HT2A", "5-HT2B": "5HT2B", "5-HT2C": "5HT2C",
        "5-HT4R": "5HT4", "5-HT5A": "5HT5A", "5-HT6R": "5HT6", "5-HT7R": "5HT7",
    }
    return repl.get(s, s)


def tanimoto_max_to_set(X_query: np.ndarray, X_ref: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Max Tanimoto of each query binary fingerprint to any reference binary fingerprint."""
    X_query = (X_query > 0).astype(np.float32)
    X_ref = (X_ref > 0).astype(np.float32)
    if len(X_ref) == 0:
        return np.zeros(len(X_query), dtype=np.float32)
    ref_sum = X_ref.sum(axis=1)[None, :]
    out = np.zeros(len(X_query), dtype=np.float32)
    for start in range(0, len(X_query), batch_size):
        q = X_query[start:start+batch_size]
        dot = q @ X_ref.T
        denom = q.sum(axis=1, keepdims=True) + ref_sum - dot
        sims = dot / np.maximum(denom, 1e-12)
        out[start:start+batch_size] = sims.max(axis=1)
    return out


def rdkit_props(smiles: str) -> dict:
    if Chem is None:
        raise ImportError("RDKit is required for property-matched decoy generation.")
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {
            "valid_smiles": False, "mol_wt": np.nan, "logp": np.nan, "hbd": np.nan,
            "hba": np.nan, "tpsa": np.nan, "rot_bonds": np.nan, "heavy_atoms": np.nan,
            "ring_count": np.nan, "formal_charge": np.nan,
        }
    return {
        "valid_smiles": True,
        "mol_wt": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "rot_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        "ring_count": float(rdMolDescriptors.CalcNumRings(mol)),
        "formal_charge": float(sum(a.GetFormalCharge() for a in mol.GetAtoms())),
    }


def load_metadata(path: Path, ligand_col: str|None, smiles_col: str|None, target_col: str|None) -> pd.DataFrame:
    raw = pd.read_csv(path)
    ligand_col = ligand_col or first_existing(raw.columns, ["ligand_id", "chembl_id", "molecule_chembl_id", "zinc_id", "ID", "id"])
    smiles_col = smiles_col or first_existing(raw.columns, ["canonical_smiles", "smiles", "SMILES", "isomeric_smiles"])
    target_col = target_col or first_existing(raw.columns, ["target", "target_id", "target_name", "receptor"])
    if ligand_col is None or smiles_col is None or target_col is None:
        raise ValueError(f"Cannot detect required columns. Found columns: {list(raw.columns)}")
    meta = pd.DataFrame({
        "ligand_id": raw[ligand_col].astype(str),
        "canonical_smiles": raw[smiles_col].astype(str),
        "target": raw[target_col].map(normalize_target),
        "source_row": np.arange(len(raw)),
    })
    meta = meta.replace([np.inf, -np.inf], np.nan)
    meta = meta.dropna(subset=["ligand_id", "canonical_smiles", "target"])
    meta = meta[(meta["ligand_id"].str.len() > 0) & (meta["canonical_smiles"].str.len() > 0) & (meta["target"].str.len() > 0)]
    meta = meta.drop_duplicates(subset=["ligand_id", "target"]).reset_index(drop=True)
    return meta


def aggregate_unique_ligands(meta: pd.DataFrame, fp_csv: Path|None) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    lig_rows = []
    for ligand_id, g in meta.groupby("ligand_id", sort=False):
        smiles = g["canonical_smiles"].dropna().astype(str).iloc[0]
        lig_rows.append({
            "ligand_id": ligand_id,
            "canonical_smiles": smiles,
            "active_targets": ";".join(sorted(g["target"].unique())),
            "n_active_targets": int(g["target"].nunique()),
        })
    ligands = pd.DataFrame(lig_rows)

    fp_unique = None
    if fp_csv:
        print(f"Loading Morgan fingerprints from {fp_csv} ...")
        fp = pd.read_csv(fp_csv)
        if len(fp) != len(pd.read_csv(args.metadata_csv, usecols=[0])):
            # Fallback to length check against original source_row max not available; this branch is just safety.
            warnings.warn("Morgan fingerprint CSV row count may not match metadata row count.")
        # Need original raw-row alignment: use meta source_row and original ligand_id.
        # Since meta dropped duplicate ligand-target, use source_row entries available; repeated ligand rows are averaged.
        fp_meta = pd.read_csv(args.metadata_csv, usecols=[args.ligand_col] if args.ligand_col else None)
        # Simpler and robust for this project: use full metadata file ligand column by detection.
        raw = pd.read_csv(args.metadata_csv, usecols=None)
        ligand_col = args.ligand_col or first_existing(raw.columns, ["ligand_id", "chembl_id", "molecule_chembl_id", "zinc_id", "ID", "id"])
        if len(fp) != len(raw):
            raise ValueError(f"Fingerprint rows ({len(fp)}) do not match metadata rows ({len(raw)}).")
        fp_num = fp.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
        fp_num["ligand_id"] = raw[ligand_col].astype(str).values
        fp_unique = fp_num.groupby("ligand_id", sort=False).mean().reset_index()
        fp_unique = ligands[["ligand_id"]].merge(fp_unique, on="ligand_id", how="left").fillna(0.0)
    return ligands, fp_unique


def build_dataset(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(Path(args.metadata_csv), args.ligand_col, args.smiles_col, args.target_col)
    ligands, fp_unique = aggregate_unique_ligands(meta, Path(args.morgan_fp_csv) if args.morgan_fp_csv else None)

    # Compute RDKit properties.
    print("Computing RDKit physicochemical properties ...")
    props = pd.DataFrame([rdkit_props(s) for s in ligands["canonical_smiles"]])
    ligands_props = pd.concat([ligands.reset_index(drop=True), props], axis=1)
    ligands_props.to_csv(out / "e5_ligand_properties.csv", index=False)

    # Save unique Morgan descriptors for later E5 analysis.
    if fp_unique is not None:
        fp_unique.to_csv(out / "e5_morgan_descriptors.csv", index=False)

    prop_cols = ["mol_wt", "logp", "hbd", "hba", "tpsa", "rot_bonds", "heavy_atoms", "ring_count", "formal_charge"]
    valid = ligands_props["valid_smiles"].fillna(False).to_numpy()
    prop_mat = ligands_props[prop_cols].to_numpy(dtype=float)
    mean = np.nanmean(prop_mat[valid], axis=0)
    std = np.nanstd(prop_mat[valid], axis=0)
    std[std == 0] = 1.0
    zprops = (prop_mat - mean) / std
    zprops[~np.isfinite(zprops)] = 0.0

    lig_index = {x: i for i, x in enumerate(ligands_props["ligand_id"])}
    target_to_ligands = {t: set(g["ligand_id"].tolist()) for t, g in meta.groupby("target")}
    all_ligs = set(ligands_props["ligand_id"])
    target_counts = meta.groupby("target")["ligand_id"].nunique().sort_values(ascending=False)
    usable_targets = [t for t, n in target_counts.items() if n >= args.min_positives]
    if args.targets:
        keep = set(args.targets)
        usable_targets = [t for t in usable_targets if t in keep]
    print("Usable targets:", usable_targets)

    fp_matrix = None
    fp_cols = []
    if fp_unique is not None:
        fp_cols = [c for c in fp_unique.columns if c != "ligand_id"]
        fp_matrix = fp_unique[fp_cols].to_numpy(dtype=np.float32)

    rows = []
    summary_rows = []
    rng = np.random.default_rng(args.seed)

    for target in usable_targets:
        pos_ids = sorted(target_to_ligands[target])
        pos_idx = np.array([lig_index[x] for x in pos_ids if x in lig_index and valid[lig_index[x]]], dtype=int)
        if len(pos_idx) < args.min_positives:
            continue
        neg_ids = sorted(all_ligs - set(pos_ids))
        neg_idx = np.array([lig_index[x] for x in neg_ids if x in lig_index and valid[lig_index[x]]], dtype=int)
        if len(neg_idx) == 0:
            continue
        # Property distance to the target's positive centroid.
        pos_centroid = zprops[pos_idx].mean(axis=0)
        prop_dist = np.linalg.norm(zprops[neg_idx] - pos_centroid[None, :], axis=1)

        # Optional topological dissimilarity filter using Morgan Tanimoto.
        max_tanimoto = np.full(len(neg_idx), np.nan)
        candidate_mask = np.ones(len(neg_idx), dtype=bool)
        tanimoto_threshold_used = np.nan
        if fp_matrix is not None and args.max_tanimoto is not None:
            max_tanimoto = tanimoto_max_to_set(fp_matrix[neg_idx], fp_matrix[pos_idx], batch_size=args.tanimoto_batch_size)
            # Try threshold, then relax if insufficient.
            for thr in [args.max_tanimoto, 0.50, 0.60, 0.70, 0.85, 1.01]:
                cm = max_tanimoto <= thr
                desired = min(args.max_decoys_per_target, args.negatives_per_positive * len(pos_idx), len(neg_idx))
                if cm.sum() >= min(desired, args.min_decoys):
                    candidate_mask = cm
                    tanimoto_threshold_used = thr
                    break

        candidate_idx_local = np.where(candidate_mask)[0]
        if len(candidate_idx_local) == 0:
            candidate_idx_local = np.arange(len(neg_idx))
        order = candidate_idx_local[np.argsort(prop_dist[candidate_idx_local])]
        desired = min(args.max_decoys_per_target, args.negatives_per_positive * len(pos_idx), len(order))
        selected_local = order[:desired]
        selected_neg_idx = neg_idx[selected_local]

        for i in pos_idx:
            lig = ligands_props.iloc[i]
            rows.append({
                "ligand_id": lig["ligand_id"],
                "canonical_smiles": lig["canonical_smiles"],
                "target": target,
                "label": 1,
                "source": "known_binder",
                "source_targets": lig["active_targets"],
                "active_for_other_target": int(lig["n_active_targets"] > 1),
                "property_match_distance": 0.0,
                "max_tanimoto_to_target_actives": 1.0,
                "activity_type": "unknown_active_from_input",
                "activity_value": np.nan,
                "activity_unit": "",
                "docking_score": np.nan,
            })
        for i, local_i in zip(selected_neg_idx, selected_local):
            lig = ligands_props.iloc[i]
            rows.append({
                "ligand_id": lig["ligand_id"],
                "canonical_smiles": lig["canonical_smiles"],
                "target": target,
                "label": 0,
                "source": "cross_target_property_matched_decoy",
                "source_targets": lig["active_targets"],
                "active_for_other_target": 1,
                "property_match_distance": float(prop_dist[local_i]),
                "max_tanimoto_to_target_actives": float(max_tanimoto[local_i]) if np.isfinite(max_tanimoto[local_i]) else np.nan,
                "activity_type": "",
                "activity_value": np.nan,
                "activity_unit": "",
                "docking_score": np.nan,
            })
        summary_rows.append({
            "target": target,
            "n_positives": int(len(pos_idx)),
            "n_negative_pool": int(len(neg_idx)),
            "n_selected_decoys": int(len(selected_neg_idx)),
            "negative_source": "cross_target_property_matched_decoy",
            "tanimoto_threshold_used": tanimoto_threshold_used,
            "median_decoy_property_distance": float(np.median(prop_dist[selected_local])) if len(selected_local) else np.nan,
            "median_decoy_max_tanimoto_to_actives": float(np.nanmedian(max_tanimoto[selected_local])) if len(selected_local) else np.nan,
        })

    dataset = pd.DataFrame(rows)
    # Add physicochemical properties to every row for audit.
    dataset = dataset.merge(ligands_props[["ligand_id"] + prop_cols], on="ligand_id", how="left")
    dataset.to_csv(out / "e5_binder_decoy_dataset.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "e5_dataset_summary.csv", index=False)
    target_counts.rename_axis("target").reset_index(name="n_input_known_binders").to_csv(out / "e5_input_target_counts.csv", index=False)

    # Split helper: train/val/test positives by ligand per target + all negatives for test by default.
    split_rows = []
    for target, g in dataset.groupby("target"):
        pos = g[g["label"] == 1]["ligand_id"].drop_duplicates().to_numpy()
        rng.shuffle(pos)
        n = len(pos)
        n_test = max(1, int(round(args.test_size * n)))
        n_val = max(1, int(round(args.val_size * n)))
        test_pos = set(pos[:n_test])
        val_pos = set(pos[n_test:n_test+n_val])
        train_pos = set(pos[n_test+n_val:])
        for _, row in g.iterrows():
            if row["label"] == 0:
                part = "test"  # decoys evaluate outlier detection; one-class model trains only on positives
            elif row["ligand_id"] in train_pos:
                part = "train"
            elif row["ligand_id"] in val_pos:
                part = "val"
            else:
                part = "test"
            split_rows.append({"ligand_id": row["ligand_id"], "target": target, "partition": part})
    pd.DataFrame(split_rows).to_csv(out / "e5_train_val_test_split.csv", index=False)

    # README/report.
    report = []
    report.append("# E5 binder/decoy dataset generated from scratch")
    report.append("")
    report.append("This is a first-pass E5 dataset constructed without external DUD-E/DEKOIS downloads.")
    report.append("")
    report.append("## Sources")
    report.append(f"- Positives: `{Path(args.metadata_csv).name}` known ligand-target assignments.")
    if args.morgan_fp_csv:
        report.append(f"- Morgan descriptors: `{Path(args.morgan_fp_csv).name}`, row-aligned to metadata and aggregated by ligand.")
    report.append("- Negatives: cross-target property-matched decoys, selected from ligands active against other targets but not known active for the query target.")
    report.append("")
    report.append("## Important caveat")
    report.append("The negative labels here mean *not known active for this target*, not experimentally confirmed non-binders. For paper-grade E5, add DUD-E/DEKOIS decoys and/or docking-confirmed non-binders.")
    report.append("")
    report.append("## Output files")
    for fn in ["e5_binder_decoy_dataset.csv", "e5_morgan_descriptors.csv", "e5_ligand_properties.csv", "e5_train_val_test_split.csv", "e5_dataset_summary.csv", "e5_input_target_counts.csv"]:
        p = out / fn
        if p.exists():
            report.append(f"- `{fn}`")
    report.append("")
    report.append("## Counts")
    report.append(f"- Rows: {len(dataset):,}")
    report.append(f"- Unique ligands: {dataset['ligand_id'].nunique():,}")
    report.append(f"- Targets: {dataset['target'].nunique():,}")
    report.append(f"- Positives: {(dataset['label'] == 1).sum():,}")
    report.append(f"- Negatives: {(dataset['label'] == 0).sum():,}")
    report.append("")
    report.append("## Per-target summary")
    report.append(summary.to_markdown(index=False))
    (out / "analysis_report.md").write_text("\n".join(report))

    print(f"Done. E5 data written to: {out}")
    print(f"Dataset: {out / 'e5_binder_decoy_dataset.csv'}")
    print(f"Summary: {out / 'e5_dataset_summary.csv'}")


def parse_args():
    p = argparse.ArgumentParser(description="Build E5 binder/decoy data from existing ligand-target metadata.")
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--morgan-fp-csv", default=None)
    p.add_argument("--out", default="e5_data_from_scratch")
    p.add_argument("--ligand-col", default=None)
    p.add_argument("--smiles-col", default=None)
    p.add_argument("--target-col", default=None)
    p.add_argument("--targets", nargs="*", default=None)
    p.add_argument("--min-positives", type=int, default=50)
    p.add_argument("--min-decoys", type=int, default=50)
    p.add_argument("--negatives-per-positive", type=int, default=5)
    p.add_argument("--max-decoys-per-target", type=int, default=5000)
    p.add_argument("--max-tanimoto", type=float, default=0.60)
    p.add_argument("--tanimoto-batch-size", type=int, default=384)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args)
