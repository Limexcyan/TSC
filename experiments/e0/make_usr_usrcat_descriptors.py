#!/usr/bin/env python3
"""
Generate RDKit USR and USRCAT descriptor CSVs for the E0 descriptor-recovery table.

Input: metadata CSV such as full_sdfs.csv with ligand and SMILES columns.
Output format: ligand_id,desc_0,desc_1,..., matching run_e0_descriptor_recovery.py --precomputed-descriptor.

Examples
--------
python make_e0_usr_usrcat_descriptors.py \
  --metadata-csv full_sdfs.csv \
  --out e0_shape_descriptors \
  --descriptors usr usrcat \
  --n-jobs 8 \
  --resume

python run_e0_descriptor_recovery.py \
  --metadata-csv full_sdfs.csv \
  --morgan-fp-csv processed_sdfs-2.csv \
  --precomputed-descriptor USR:e0_shape_descriptors/e0_usr_descriptors.csv \
  --precomputed-descriptor USRCAT:e0_shape_descriptors/e0_usrcat_descriptors.csv \
  --descriptors morgan pca_sh \
  --out e0_descriptor_recovery_results_with_precomputed_usr_usrcat \
  --n-jobs 8 \
  --mantel-permutations 999
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols = list(columns)
    lower = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None


def load_unique_ligands(path: Path, ligand_col: Optional[str], smiles_col: Optional[str]) -> pd.DataFrame:
    raw = pd.read_csv(path)
    ligand_col = ligand_col or first_existing(
        raw.columns,
        ["ligand_id", "chembl_id", "molecule_chembl_id", "zinc_id", "compound_id", "ID", "id"],
    )
    smiles_col = smiles_col or first_existing(raw.columns, ["canonical_smiles", "smiles", "SMILES", "canonical_smile"])
    if ligand_col is None or smiles_col is None:
        raise ValueError(
            f"Could not detect ligand/smiles columns. Available columns: {list(raw.columns)}. "
            "Use --ligand-col and --smiles-col."
        )
    df = raw[[ligand_col, smiles_col]].copy()
    df.columns = ["ligand_id", "canonical_smiles"]
    df["ligand_id"] = df["ligand_id"].astype(str)
    df["canonical_smiles"] = df["canonical_smiles"].astype(str)
    df = df[df["ligand_id"].notna()]
    df = df[df["canonical_smiles"].notna()]
    df = df[df["canonical_smiles"].str.lower().ne("nan")]
    df = df[df["canonical_smiles"].str.len().gt(0)]
    # Preserve first SMILES per ligand, matching run_e0_descriptor_recovery.py collapse_ligands().
    return df.drop_duplicates("ligand_id", keep="first").reset_index(drop=True)


def import_rdkit():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    return Chem, AllChem, rdMolDescriptors


def mol_with_conformer(smi: str, seed: int, optimize: bool = True, max_iters: int = 150):
    Chem, AllChem, _ = import_rdkit()
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=int(seed), useRandomCoords=True)
    if status != 0:
        return None
    if optimize:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_iters))
        except Exception:
            # Keep embedded conformer even if UFF typing/optimization fails.
            pass
    return mol


def compute_one(task: Tuple[str, str, int, Tuple[str, ...], bool, int]):
    ligand_id, smi, seed, descriptors, optimize, max_iters = task
    _, _, rdMolDescriptors = import_rdkit()
    result = {"ligand_id": ligand_id, "ok": True, "error": ""}
    mol = mol_with_conformer(smi, seed=seed, optimize=optimize, max_iters=max_iters)
    if mol is None:
        result["ok"] = False
        result["error"] = "conformer_generation_failed"
        for d in descriptors:
            result[d] = None
        return result
    for d in descriptors:
        try:
            if d == "usr":
                result[d] = np.asarray(rdMolDescriptors.GetUSR(mol), dtype=np.float32)
            elif d == "usrcat":
                result[d] = np.asarray(rdMolDescriptors.GetUSRCAT(mol), dtype=np.float32)
            else:
                raise ValueError(f"unsupported descriptor: {d}")
        except Exception as e:
            result[d] = None
            result["ok"] = False
            result["error"] = (result["error"] + ";" if result["error"] else "") + f"{d}:{type(e).__name__}:{e}"
    return result


def save_descriptor_csv(path: Path, ligand_ids: Sequence[str], vectors: Sequence[Optional[np.ndarray]], expected_dim: Optional[int] = None) -> Tuple[int, int]:
    dim = expected_dim or max((len(v) for v in vectors if v is not None), default=0)
    if dim == 0:
        raise RuntimeError(f"No descriptors could be saved to {path}")
    rows = []
    failed = 0
    for ligand_id, vec in zip(ligand_ids, vectors):
        arr = np.zeros(dim, dtype=np.float32)
        if vec is None or len(vec) == 0 or not np.isfinite(vec).all():
            failed += 1
        else:
            flat = np.asarray(vec, dtype=np.float32).ravel()
            arr[: min(dim, len(flat))] = flat[:dim]
        rows.append([ligand_id] + arr.tolist())
    cols = ["ligand_id"] + [f"desc_{i}" for i in range(dim)]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return dim, failed


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate E0 USR/USRCAT descriptor CSVs.")
    ap.add_argument("--metadata-csv", required=True)
    ap.add_argument("--out", default="e0_shape_descriptors")
    ap.add_argument("--descriptors", nargs="+", default=["usr", "usrcat"], choices=["usr", "usrcat"])
    ap.add_argument("--ligand-col", default=None)
    ap.add_argument("--smiles-col", default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-iters", type=int, default=150)
    ap.add_argument("--no-optimize", action="store_true", help="Skip UFF optimization after embedding.")
    ap.add_argument("--max-ligands", type=int, default=None, help="Optional debug limit.")
    ap.add_argument("--resume", action="store_true", help="Skip descriptors whose output CSV already exists.")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    descriptors = tuple(d.lower() for d in args.descriptors)

    ligands = load_unique_ligands(Path(args.metadata_csv), args.ligand_col, args.smiles_col)
    if args.max_ligands:
        ligands = ligands.head(args.max_ligands).copy()

    print(f"Loaded {len(ligands):,} unique ligands from {args.metadata_csv}")
    print(f"Descriptors: {', '.join(descriptors)}")

    # If resuming and all requested outputs exist, just exit.
    output_paths = {d: out / f"e0_{d}_descriptors.csv" for d in descriptors}
    pending = [d for d in descriptors if not (args.resume and output_paths[d].exists())]
    if not pending:
        print("All requested descriptor files already exist; nothing to do.")
        return 0

    tasks = [
        (row.ligand_id, row.canonical_smiles, args.seed, tuple(pending), not args.no_optimize, args.max_iters)
        for row in ligands.itertuples(index=False)
    ]

    start = time.time()
    results = []
    if args.n_jobs and args.n_jobs > 1:
        with Pool(processes=args.n_jobs) as pool:
            for i, res in enumerate(pool.imap(compute_one, tasks, chunksize=64), 1):
                results.append(res)
                if i % 500 == 0 or i == len(tasks):
                    elapsed = time.time() - start
                    print(f"{i:,}/{len(tasks):,} ligands done ({elapsed/60:.1f} min)", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            results.append(compute_one(task))
            if i % 100 == 0 or i == len(tasks):
                elapsed = time.time() - start
                print(f"{i:,}/{len(tasks):,} ligands done ({elapsed/60:.1f} min)", flush=True)

    ligand_ids = [r["ligand_id"] for r in results]
    summary_rows = []
    for d in pending:
        path = output_paths[d]
        vectors = [r.get(d) for r in results]
        expected_dim = 12 if d == "usr" else 60 if d == "usrcat" else None
        dim, failed = save_descriptor_csv(path, ligand_ids, vectors, expected_dim=expected_dim)
        print(f"saved {path} | dim={dim} | failed_zero_vectors={failed}")
        summary_rows.append({"descriptor": d, "path": str(path), "n_ligands": len(ligand_ids), "descriptor_dim": dim, "failed_zero_vectors": failed})

    error_rows = []
    for r in results:
        if not r.get("ok", False):
            error_rows.append({"ligand_id": r["ligand_id"], "error": r.get("error", "")})
    if error_rows:
        pd.DataFrame(error_rows).to_csv(out / "e0_usr_usrcat_descriptor_errors.csv", index=False)
        print(f"saved {out / 'e0_usr_usrcat_descriptor_errors.csv'}")

    summary_path = out / "e0_usr_usrcat_descriptor_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
