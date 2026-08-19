#!/usr/bin/env python3
"""
Generate E5 shape descriptor matrices for the same ligand IDs used by the
E5 binder/decoy dataset.

Outputs one CSV per descriptor:
    e5_usr_descriptors.csv
    e5_usrcat_descriptors.csv
    e5_pca_sh_descriptors.csv
    e5_bdc_descriptors.csv  (optional; requires local TSC wrapper)

Input dataset must contain at least:
    ligand_id, canonical_smiles

Examples
--------
# RDKit descriptors only
python make_e5_shape_descriptors.py \
  --dataset-csv e5_data_from_scratch/e5_binder_decoy_dataset.csv \
  --descriptors usr usrcat pca_sh \
  --out e5_data_from_scratch \
  --n-jobs 8

# Include BDC via local tsc_bdc_wrapper.py
PYTHONPATH="../TSC-restore-3d-desc/core:." python make_e5_shape_descriptors.py \
  --dataset-csv e5_data_from_scratch/e5_binder_decoy_dataset.csv \
  --descriptors usr usrcat pca_sh bdc \
  --tsc-core-path ../TSC-restore-3d-desc/core \
  --bdc-function tsc_bdc_wrapper:compute_bdc_descriptor \
  --out e5_data_from_scratch \
  --n-jobs 8
"""
from __future__ import annotations

import argparse
import importlib
import sys
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd

try:
    from scipy.special import sph_harm_y
except Exception:  # old scipy compatibility
    from scipy.special import sph_harm
    def sph_harm_y(l, m, theta, phi):
        return sph_harm(m, l, phi, theta)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate E5 USR/USRCAT/PCA-SH/BDC descriptors.")
    p.add_argument("--dataset-csv", required=True, help="e5_binder_decoy_dataset.csv")
    p.add_argument("--out", default="e5_data_from_scratch", help="Output directory")
    p.add_argument("--descriptors", nargs="+", default=["usr", "usrcat", "pca_sh"],
                   help="Any of: usr usrcat pca_sh bdc")
    p.add_argument("--ligand-col", default="ligand_id")
    p.add_argument("--smiles-col", default="canonical_smiles")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-iters", type=int, default=150, help="UFF optimization max iterations")
    p.add_argument("--tsc-core-path", default=None, help="Path to TSC/core for BDC")
    p.add_argument("--bdc-function", default="tsc_bdc_wrapper:compute_bdc_descriptor",
                   help="module:function for BDC wrapper")
    p.add_argument("--resume", action="store_true", help="Skip descriptors whose output CSV already exists")
    return p.parse_args()


def load_ligands(path: Path, ligand_col: str, smiles_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in [ligand_col, smiles_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns {missing}. Available: {list(df.columns)}")
    ligs = (df[[ligand_col, smiles_col]]
            .rename(columns={ligand_col: "ligand_id", smiles_col: "canonical_smiles"})
            .dropna(subset=["ligand_id", "canonical_smiles"])
            .astype({"ligand_id": str, "canonical_smiles": str})
            .drop_duplicates("ligand_id")
            .reset_index(drop=True))
    ligs = ligs[ligs["canonical_smiles"].str.len() > 0].reset_index(drop=True)
    return ligs


def import_rdkit():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    return Chem, AllChem, rdMolDescriptors


def mol_with_conformer(smiles: str, seed: int = 1, max_iters: int = 150):
    Chem, AllChem, _ = import_rdkit()
    mol = Chem.MolFromSmiles(str(smiles))
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
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=int(max_iters))
    except Exception:
        # Keep embedded conformer even if UFF parameterisation/optimisation fails.
        pass
    return mol


def pca_sh_descriptor_from_mol(mol, lmax: int = 4) -> np.ndarray:
    conf = mol.GetConformer()
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    dim = ((lmax + 1) ** 2) * 2
    if len(heavy) < 3:
        return np.zeros(dim, dtype=np.float32)
    coords = np.asarray([conf.GetAtomPosition(i) for i in heavy], dtype=float)
    coords -= coords.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(coords, full_matrices=False)
        coords = coords @ vt.T
    except Exception:
        pass
    r = np.linalg.norm(coords, axis=1)
    r_safe = np.maximum(r, 1e-8)
    theta = np.arccos(np.clip(coords[:, 2] / r_safe, -1.0, 1.0))
    phi = np.arctan2(coords[:, 1], coords[:, 0])
    weights = r_safe / max(float(np.mean(r_safe)), 1e-8)
    feats = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            y = sph_harm_y(l, m, theta, phi)
            val = np.mean(weights * y)
            feats.extend([float(np.real(val)), float(np.imag(val))])
    return np.asarray(feats, dtype=np.float32)


def compute_rdkit_one(task):
    ligand_id, smiles, requested, seed, max_iters = task
    _, _, rdMolDescriptors = import_rdkit()
    mol = mol_with_conformer(smiles, seed=seed, max_iters=max_iters)
    out = {"ligand_id": ligand_id}
    if mol is None:
        for name in requested:
            out[name] = None
        return out
    for name in requested:
        try:
            if name == "usr":
                out[name] = np.asarray(rdMolDescriptors.GetUSR(mol), dtype=np.float32).ravel()
            elif name == "usrcat":
                out[name] = np.asarray(rdMolDescriptors.GetUSRCAT(mol), dtype=np.float32).ravel()
            elif name in {"pca_sh", "sh", "pca-sh"}:
                out[name] = pca_sh_descriptor_from_mol(mol).ravel()
        except Exception:
            out[name] = None
    return out


def write_descriptor_csv(rows, name: str, out_dir: Path) -> dict:
    # rows are dicts with ligand_id and descriptor arrays under key name
    values = [(r["ligand_id"], r.get(name)) for r in rows]
    dim = max((len(v) for _, v in values if v is not None), default=0)
    if dim == 0:
        warnings.warn(f"No {name} descriptors were computed; not writing output.")
        return {"descriptor": name, "n_ligands": len(values), "dim": 0, "n_failed": len(values), "path": ""}
    out_rows = []
    failed = 0
    for ligand_id, vec in values:
        arr = np.zeros(dim, dtype=np.float32)
        if vec is None or len(vec) == 0:
            failed += 1
        else:
            v = np.asarray(vec, dtype=np.float32).ravel()
            arr[:min(dim, len(v))] = v[:dim]
        out_rows.append([ligand_id] + arr.tolist())
    cols = ["ligand_id"] + [f"desc_{i}" for i in range(dim)]
    out_path = out_dir / f"e5_{canonical_name(name)}_descriptors.csv"
    pd.DataFrame(out_rows, columns=cols).to_csv(out_path, index=False)
    print(f"saved {out_path}  dim={dim}  failed={failed}/{len(values)}")
    return {"descriptor": canonical_name(name), "n_ligands": len(values), "dim": dim,
            "n_failed": failed, "failure_rate": failed / max(len(values), 1), "path": str(out_path)}


def canonical_name(name: str) -> str:
    n = name.lower().replace("-", "_")
    if n == "sh":
        return "pca_sh"
    return n


def load_bdc_function(tsc_core_path: str | None, spec: str):
    if tsc_core_path:
        sys.path.insert(0, str(Path(tsc_core_path).resolve()))
    sys.path.insert(0, str(Path(".").resolve()))
    if ":" not in spec:
        raise ValueError("--bdc-function must be module:function, e.g. tsc_bdc_wrapper:compute_bdc_descriptor")
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def compute_bdc(ligs: pd.DataFrame, out_dir: Path, args) -> dict:
    fn = load_bdc_function(args.tsc_core_path, args.bdc_function)
    rows = []
    errors = []
    for i, row in ligs.iterrows():
        ligand_id = row["ligand_id"]
        smiles = row["canonical_smiles"]
        try:
            vec = np.asarray(fn(smiles), dtype=np.float32).ravel()
            if vec.size == 0 or not np.isfinite(vec).all():
                raise ValueError("empty or non-finite BDC descriptor")
            rows.append({"ligand_id": ligand_id, "bdc": vec})
        except Exception as e:
            rows.append({"ligand_id": ligand_id, "bdc": None})
            errors.append({"ligand_id": ligand_id, "canonical_smiles": smiles, "error": str(e)})
        if (i + 1) % 100 == 0:
            print(f"BDC {i+1}/{len(ligs)} done; errors={len(errors)}")
    summary = write_descriptor_csv(rows, "bdc", out_dir)
    if errors:
        err_path = out_dir / "e5_bdc_descriptor_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False)
        summary["error_path"] = str(err_path)
    return summary


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    descriptors = [canonical_name(d) for d in args.descriptors]
    ligs = load_ligands(Path(args.dataset_csv), args.ligand_col, args.smiles_col)
    print(f"unique ligands: {len(ligs)}")

    summaries = []
    rdkit_desc = [d for d in descriptors if d in {"usr", "usrcat", "pca_sh"}]
    rdkit_desc = list(dict.fromkeys(rdkit_desc))
    if rdkit_desc:
        skip_all = False
        if args.resume:
            rdkit_desc = [d for d in rdkit_desc if not (out_dir / f"e5_{d}_descriptors.csv").exists()]
            skip_all = len(rdkit_desc) == 0
        if not skip_all:
            print("computing RDKit descriptors:", rdkit_desc)
            tasks = [(r.ligand_id, r.canonical_smiles, rdkit_desc, args.seed, args.max_iters)
                     for r in ligs.itertuples(index=False)]
            if args.n_jobs and args.n_jobs > 1:
                rows = []
                with Pool(processes=args.n_jobs) as pool:
                    for i, res in enumerate(pool.imap(compute_rdkit_one, tasks, chunksize=64), start=1):
                        rows.append(res)
                        if i % 500 == 0:
                            print(f"RDKit {i}/{len(tasks)} done")
            else:
                rows = []
                for i, task in enumerate(tasks, start=1):
                    rows.append(compute_rdkit_one(task))
                    if i % 500 == 0:
                        print(f"RDKit {i}/{len(tasks)} done")
            for d in rdkit_desc:
                summaries.append(write_descriptor_csv(rows, d, out_dir))

    if "bdc" in descriptors:
        bdc_path = out_dir / "e5_bdc_descriptors.csv"
        if args.resume and bdc_path.exists():
            print(f"resume: skipping existing {bdc_path}")
        else:
            print("computing BDC descriptors")
            summaries.append(compute_bdc(ligs, out_dir, args))

    if summaries:
        summary_path = out_dir / "e5_descriptor_generation_summary.csv"
        pd.DataFrame(summaries).to_csv(summary_path, index=False)
        print("summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
