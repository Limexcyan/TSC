#!/usr/bin/env python3
"""
E3 target-assignment benchmark for TaSC.

Goal
----
Evaluate pocket-free, multi-label target assignment from ligand descriptors under:
  1) scaffold split
  2) time split, when a year/date column is available

Implemented methods
-------------------
1. shape_distribution
   Cluster training ligands in descriptor space, estimate target-level cluster
   distributions, and assign unseen ligands to targets by nearest distribution
   using Jensen-Shannon distance. Scores are converted to calibrated multi-label
   probabilities by global Platt calibration on the validation set.

2. knn
   k-nearest-neighbor ligand-label transfer in descriptor space.

3. sea_like
   SEA-inspired ligand-set baseline. For each target, score a query ligand by
   its best similarity to the target's known ligands. This is not a full SEA
   implementation, but it is a useful ligand-set baseline when SwissTargetPrediction
   is not included.

Inputs
------
Preferred metadata CSV columns, with flexible auto-detection:
  ligand_id, canonical_smiles, target_id/target, pchembl_value, document_year

For previous ZINC/prepared-ligand data used in this project, this script supports:
  full_sdfs.csv             metadata, with columns such as ID, smiles, target
  processed_sdfs-2.csv      precomputed Morgan fingerprints, row-aligned to full_sdfs.csv

Optional descriptor sources
---------------------------
- Precomputed descriptor CSV: one row per ligand, either row-aligned to metadata
  or containing a ligand_id column.
- RDKit-generated Morgan fingerprints from SMILES.
- RDKit USR/USRCAT descriptors from generated conformers.
- TSC descriptor hook from a local clone of https://github.com/Limexcyan/TSC,
  using --tsc-core-path and --tsc-function module:function.

Examples
--------
# Previous ZINC/prepared data, Morgan only:
python run_e3_target_assignment.py \
  --metadata-csv full_sdfs.csv \
  --morgan-fp-csv processed_sdfs-2.csv \
  --descriptors morgan \
  --out e3_results

# Compute Morgan from SMILES:
python run_e3_target_assignment.py \
  --metadata-csv activities.csv \
  --descriptors morgan \
  --out e3_results

# Use a local TSC descriptor function:
python run_e3_target_assignment.py \
  --metadata-csv full_sdfs.csv \
  --descriptors tsc_bdc \
  --tsc-core-path /path/to/TSC/core \
  --tsc-function descriptors.bdc:compute_bdc_descriptor \
  --out e3_results

Outputs
-------
CSV tables and manuscript-ready plots are written to --out.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import os
import re
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.special import softmax
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import NearestNeighbors


plt.rcParams.update({
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
})


# ---------------------------------------------------------------------------
# Column detection and data loading
# ---------------------------------------------------------------------------


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    # fuzzy fallback
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
    replacements = {
        "5H2A": "5HT2A",
        "5-HT1A": "5HT1A",
        "5-HT1B": "5HT1B",
        "5-HT1D": "5HT1D",
        "5-HT2A": "5HT2A",
        "5-HT2B": "5HT2B",
        "5-HT2C": "5HT2C",
        "5-HT4R": "5HT4",
        "5-HT5A": "5HT5A",
        "5-HT6R": "5HT6",
        "5-HT7R": "5HT7",
    }
    return replacements.get(s, s)


def detect_metadata_columns(df: pd.DataFrame) -> dict[str, Optional[str]]:
    columns = df.columns
    ligand_col = _first_existing(columns, [
        "ligand_id", "compound_id", "molecule_id", "chembl_id", "molecule_chembl_id",
        "zinc_id", "zinc", "ID", "id", "name", "title", "s_m_entry_name",
    ])
    smiles_col = _first_existing(columns, [
        "canonical_smiles", "smiles", "SMILES", "canonical_smile", "isomeric_smiles"
    ])
    target_col = _first_existing(columns, [
        "target", "target_id", "target_name", "pref_name", "receptor", "protein", "assay_target"
    ])
    year_col = _first_existing(columns, [
        "document_year", "publication_year", "year", "activity_year", "date", "standard_year"
    ])
    value_col = _first_existing(columns, [
        "pchembl_value", "pChEMBL", "activity_value", "standard_value", "value", "Ki", "IC50", "Kd", "EC50"
    ])
    return {
        "ligand": ligand_col,
        "smiles": smiles_col,
        "target": target_col,
        "year": year_col,
        "activity": value_col,
    }


def load_metadata(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(path)
    detected = detect_metadata_columns(df)
    ligand_col = args.ligand_col or detected["ligand"]
    smiles_col = args.smiles_col or detected["smiles"]
    target_col = args.target_col or detected["target"]
    year_col = args.year_col or detected["year"]
    activity_col = args.activity_col or detected["activity"]

    if target_col is None:
        raise ValueError(
            "Could not detect target column. Pass --target-col explicitly. "
            f"Available columns: {list(df.columns)}"
        )
    if ligand_col is None:
        # Stable synthetic ID from SMILES + row if no explicit ID exists.
        ligand_col = "__row_ligand_id"
        df[ligand_col] = [f"ligand_{i}" for i in range(len(df))]
    if smiles_col is None:
        warnings.warn("No SMILES column detected. Scaffold split and RDKit descriptors may be unavailable.")

    out = pd.DataFrame({
        "ligand_id": df[ligand_col].astype(str),
        "target": df[target_col].map(normalize_target),
    })
    out["canonical_smiles"] = df[smiles_col].astype(str) if smiles_col else ""

    if year_col:
        years = pd.to_numeric(df[year_col].astype(str).str.extract(r"(\d{4})", expand=False), errors="coerce")
        out["document_year"] = years
    else:
        out["document_year"] = np.nan

    if activity_col:
        out["activity_value"] = pd.to_numeric(df[activity_col], errors="coerce")
    else:
        out["activity_value"] = np.nan

    # Optional activity filtering for pChEMBL-like values. We do not assume units.
    if args.min_activity_value is not None and activity_col:
        out = out.loc[out["activity_value"] >= args.min_activity_value].copy()

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=["ligand_id", "target"])
    out = out[out["target"].astype(str).str.len() > 0]
    out = out.drop_duplicates(subset=["ligand_id", "target"]).reset_index(drop=True)
    return out


def collapse_to_ligands(meta: pd.DataFrame) -> tuple[pd.DataFrame, list[list[str]]]:
    # One row per ligand. If the same ligand has multiple SMILES, keep the first non-empty.
    rows = []
    label_sets = []
    for ligand_id, g in meta.groupby("ligand_id", sort=False):
        smiles_values = [s for s in g["canonical_smiles"].astype(str).tolist() if s and s.lower() != "nan"]
        year_values = g["document_year"].dropna()
        rows.append({
            "ligand_id": ligand_id,
            "canonical_smiles": smiles_values[0] if smiles_values else "",
            "document_year": float(year_values.min()) if len(year_values) else np.nan,
            "n_targets": int(g["target"].nunique()),
        })
        label_sets.append(sorted(g["target"].unique().tolist()))
    ligand_df = pd.DataFrame(rows)
    return ligand_df, label_sets


# ---------------------------------------------------------------------------
# RDKit helpers
# ---------------------------------------------------------------------------


def import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolDescriptors
        from rdkit.Chem.Scaffolds import MurckoScaffold
        return Chem, AllChem, rdMolDescriptors, MurckoScaffold
    except Exception as exc:
        raise ImportError(
            "RDKit is required for this operation. Install it with conda, e.g. "
            "conda install -c conda-forge rdkit"
        ) from exc


def morgan_from_smiles(smiles: list[str], radius: int = 2, n_bits: int = 1024) -> np.ndarray:
    Chem, AllChem, _, _ = import_rdkit()
    X = np.zeros((len(smiles), n_bits), dtype=np.float32)
    bad = 0
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            bad += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        # RDKit DataStructs is in Chem namespace indirectly but import explicitly.
        from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr.astype(np.float32)
    if bad:
        warnings.warn(f"Morgan fingerprint: {bad} invalid SMILES were encoded as all-zero vectors.")
    return X


def murcko_scaffolds(smiles: list[str]) -> list[str]:
    try:
        Chem, _, _, MurckoScaffold = import_rdkit()
    except ImportError:
        warnings.warn("RDKit unavailable; using canonical_smiles string as scaffold fallback.")
        return [str(s) if str(s) else f"missing_{i}" for i, s in enumerate(smiles)]
    scaffolds = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            scaffolds.append(f"invalid_{i}")
            continue
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        scaffolds.append(Chem.MolToSmiles(scaf, isomericSmiles=False) if scaf is not None else f"empty_{i}")
    return scaffolds


def mol_with_conformer(smi: str, seed: int = 1):
    Chem, AllChem, _, _ = import_rdkit()
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        return None
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=100)
    except Exception:
        pass
    return mol


def rdkit_3d_descriptor(smiles: list[str], descriptor: str, seed: int = 1) -> np.ndarray:
    _, _, rdMolDescriptors, _ = import_rdkit()
    rows = []
    bad = 0
    for smi in smiles:
        mol = mol_with_conformer(str(smi), seed=seed)
        if mol is None:
            rows.append(None)
            bad += 1
            continue
        try:
            if descriptor.lower() == "usr":
                vec = list(rdMolDescriptors.GetUSR(mol))
            elif descriptor.lower() == "usrcat":
                vec = list(rdMolDescriptors.GetUSRCAT(mol))
            else:
                raise ValueError(descriptor)
            rows.append(np.asarray(vec, dtype=np.float32))
        except Exception:
            rows.append(None)
            bad += 1
    dim = max((len(r) for r in rows if r is not None), default=0)
    if dim == 0:
        raise ValueError(f"Could not compute any {descriptor} descriptors.")
    X = np.zeros((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        if r is not None:
            X[i, : len(r)] = r
    if bad:
        warnings.warn(f"{descriptor}: {bad} molecules failed conformer/descriptor generation and were encoded as zero vectors.")
    return X


# ---------------------------------------------------------------------------
# TSC descriptor hook
# ---------------------------------------------------------------------------


def load_tsc_function(core_path: Path, function_spec: Optional[str]) -> Callable:
    """Load a descriptor function from a local TSC core checkout.

    The GitHub URL supplied in the project notes points to a branch/path, but
    local function names may differ. To keep this script usable, the most robust
    path is to pass --tsc-function module:function.
    """
    if not core_path.exists():
        raise FileNotFoundError(f"TSC core path does not exist: {core_path}")

    # Add core and its parent, because local code may use either style.
    sys.path.insert(0, str(core_path))
    sys.path.insert(0, str(core_path.parent))

    candidates = []
    if function_spec:
        if ":" not in function_spec:
            raise ValueError("--tsc-function must have form module:function")
        mod, fn = function_spec.split(":", 1)
        candidates.append((mod, fn))
    else:
        # Best-effort guesses. Users can override with --tsc-function.
        candidates.extend([
            ("descriptors.bdc", "compute_bdc_descriptor"),
            ("descriptors.bdc", "compute_bdc"),
            ("descriptors.bdc", "bdc_descriptor"),
            ("bdc", "compute_bdc_descriptor"),
            ("bdc", "compute_bdc"),
            ("shape_descriptors", "compute_bdc_descriptor"),
            ("shape_descriptors", "compute_descriptor"),
            ("core.descriptors.bdc", "compute_bdc_descriptor"),
        ])

    errors = []
    for mod_name, fn_name in candidates:
        try:
            mod = importlib.import_module(mod_name)
            fn = getattr(mod, fn_name)
            return fn
        except Exception as exc:
            errors.append(f"{mod_name}:{fn_name} -> {type(exc).__name__}: {exc}")
    raise ImportError(
        "Could not load a TSC descriptor function. Pass --tsc-function module:function.\n"
        + "Tried:\n"
        + "\n".join(errors[:20])
    )


def compute_tsc_descriptor(
    smiles: list[str],
    core_path: Path,
    function_spec: Optional[str],
    seed: int = 1,
) -> np.ndarray:
    fn = load_tsc_function(core_path, function_spec)
    Chem = None
    try:
        Chem, _, _, _ = import_rdkit()
    except Exception:
        pass

    rows = []
    errors = 0
    for i, smi in enumerate(smiles):
        value = None
        call_errors = []
        # Try calling with SMILES first.
        try:
            value = fn(str(smi))
        except Exception as exc:
            call_errors.append(exc)
        # Try calling with RDKit mol if possible.
        if value is None and Chem is not None:
            try:
                mol = mol_with_conformer(str(smi), seed=seed)
                if mol is not None:
                    value = fn(mol)
            except Exception as exc:
                call_errors.append(exc)
        # Try keyword style.
        if value is None:
            try:
                value = fn(smiles=str(smi))
            except Exception as exc:
                call_errors.append(exc)
        if value is None:
            errors += 1
            rows.append(None)
            continue
        arr = np.asarray(value, dtype=np.float32).ravel()
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            errors += 1
            rows.append(None)
        else:
            rows.append(arr)
    dim = max((len(r) for r in rows if r is not None), default=0)
    if dim == 0:
        raise ValueError("TSC descriptor hook did not produce any numeric descriptors.")
    X = np.zeros((len(rows), dim), dtype=np.float32)
    for i, r in enumerate(rows):
        if r is not None:
            X[i, : len(r)] = r[:dim]
    if errors:
        warnings.warn(f"TSC descriptor: {errors} molecules failed and were encoded as zero vectors.")
    return X


# ---------------------------------------------------------------------------
# Descriptor loading
# ---------------------------------------------------------------------------


def load_row_aligned_descriptor_csv(path: Path, ligand_df: pd.DataFrame, ligand_col: str = "ligand_id") -> np.ndarray:
    df = pd.read_csv(path)
    # If a ligand_id-like column is present, align by ligand_id.
    id_col = _first_existing(df.columns, [ligand_col, "ligand_id", "ID", "chembl_id", "zinc_id"])
    if id_col:
        desc_cols = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
        if not desc_cols:
            raise ValueError(f"No numeric descriptor columns found in {path}")
        aligned = ligand_df[["ligand_id"]].merge(df[[id_col] + desc_cols], left_on="ligand_id", right_on=id_col, how="left")
        X = aligned[desc_cols].fillna(0.0).to_numpy(dtype=np.float32)
        return X
    # Else assume row alignment. If metadata was collapsed from duplicate ligand-target rows,
    # this may no longer match; in that case, require an ID column or pre-collapse metadata.
    if len(df) != len(ligand_df):
        raise ValueError(
            f"Descriptor CSV {path} has {len(df)} rows but unique ligand table has {len(ligand_df)} rows. "
            "Add a ligand_id column to the descriptor CSV or pass metadata with one row per ligand."
        )
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] == 0:
        # Some processed_sdfs files may contain only 0/1 values loaded as object strings.
        numeric = df.apply(pd.to_numeric, errors="coerce")
    return numeric.fillna(0.0).to_numpy(dtype=np.float32)


def load_descriptors(args: argparse.Namespace, ligand_df: pd.DataFrame) -> dict[str, np.ndarray]:
    descriptors: dict[str, np.ndarray] = {}
    requested = [d.lower() for d in args.descriptors]

    if "morgan" in requested:
        if args.morgan_fp_csv:
            X = pd.read_csv(args.morgan_fp_csv)
            if len(X) == len(ligand_df):
                descriptors["morgan"] = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            else:
                # In previous project data processed_sdfs-2.csv is row-aligned to full_sdfs.csv before collapsing.
                # If there are duplicate ligand-target rows, aggregate descriptor rows by ligand id.
                raw_meta = pd.read_csv(args.metadata_csv)
                detected = detect_metadata_columns(raw_meta)
                ligand_col = args.ligand_col or detected["ligand"]
                if ligand_col is None or len(X) != len(raw_meta):
                    raise ValueError(
                        f"Morgan fingerprint file has {len(X)} rows; metadata has {len(raw_meta)} rows; "
                        f"unique ligand table has {len(ligand_df)} rows. Cannot align."
                    )
                X_num = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
                X_num["ligand_id"] = raw_meta[ligand_col].astype(str).values
                X_agg = X_num.groupby("ligand_id", sort=False).mean()
                descriptors["morgan"] = ligand_df[["ligand_id"]].merge(
                    X_agg, left_on="ligand_id", right_index=True, how="left"
                ).drop(columns=["ligand_id"]).fillna(0.0).to_numpy(dtype=np.float32)
        else:
            descriptors["morgan"] = morgan_from_smiles(
                ligand_df["canonical_smiles"].tolist(), radius=args.morgan_radius, n_bits=args.morgan_bits
            )

    if "usr" in requested:
        descriptors["usr"] = rdkit_3d_descriptor(ligand_df["canonical_smiles"].tolist(), "usr", seed=args.seed)
    if "usrcat" in requested:
        descriptors["usrcat"] = rdkit_3d_descriptor(ligand_df["canonical_smiles"].tolist(), "usrcat", seed=args.seed)
    if "tsc_bdc" in requested or "bdc" in requested:
        if not args.tsc_core_path:
            raise ValueError("Descriptor bdc/tsc_bdc requires --tsc-core-path and usually --tsc-function.")
        descriptors["bdc"] = compute_tsc_descriptor(
            ligand_df["canonical_smiles"].tolist(),
            core_path=Path(args.tsc_core_path),
            function_spec=args.tsc_function,
            seed=args.seed,
        )

    for spec in args.precomputed_descriptor or []:
        # Format name:path.csv
        if ":" not in spec:
            raise ValueError("--precomputed-descriptor entries must have form name:path.csv")
        name, path_str = spec.split(":", 1)
        descriptors[name] = load_row_aligned_descriptor_csv(Path(path_str), ligand_df)

    if not descriptors:
        raise ValueError("No descriptors loaded. Pass --descriptors morgan or --precomputed-descriptor name:path.csv")
    return descriptors


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def split_by_groups(
    ligand_df: pd.DataFrame,
    label_sets: list[list[str]],
    test_size: float,
    val_size: float,
    seed: int,
) -> dict[str, np.ndarray]:
    scaffolds = np.array(murcko_scaffolds(ligand_df["canonical_smiles"].tolist()))
    idx = np.arange(len(ligand_df))

    # Very small or chemically homogeneous toy datasets can have too few unique
    # Murcko scaffolds for GroupShuffleSplit. Real scaffold experiments should
    # use the grouped path; the fallback keeps the script runnable and marks the
    # split as random_like.
    unique_groups = np.unique(scaffolds)
    if len(unique_groups) < 3:
        warnings.warn(
            f"Only {len(unique_groups)} unique scaffolds detected; falling back to a random split. "
            "For the paper, use real SMILES with enough scaffold diversity."
        )
        trainval_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=seed, shuffle=True)
        relative_val = val_size / max(1e-9, 1.0 - test_size)
        train_idx, val_idx = train_test_split(trainval_idx, test_size=relative_val, random_state=seed + 1, shuffle=True)
        return {"train": train_idx, "val": val_idx, "test": test_idx, "scaffold": scaffolds, "split_note": "random_fallback_too_few_scaffolds"}

    # train+val vs test
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(idx, groups=scaffolds))
    # train vs val among trainval. val_size is global fraction; convert to relative.
    relative_val = val_size / max(1e-9, 1.0 - test_size)
    trainval_scaffolds = scaffolds[trainval_idx]
    if len(np.unique(trainval_scaffolds)) < 2:
        warnings.warn("Too few scaffolds remain for validation split; using random validation split within train/val.")
        train_idx, val_idx = train_test_split(trainval_idx, test_size=relative_val, random_state=seed + 1, shuffle=True)
    else:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed + 1)
        train_rel, val_rel = next(gss2.split(trainval_idx, groups=trainval_scaffolds))
        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]
    return {"train": train_idx, "val": val_idx, "test": test_idx, "scaffold": scaffolds}


def split_by_time(
    ligand_df: pd.DataFrame,
    test_size: float,
    val_size: float,
) -> Optional[dict[str, np.ndarray]]:
    years = ligand_df["document_year"]
    valid = years.notna()
    if valid.mean() < 0.80:
        warnings.warn("Skipping time split: fewer than 80% of ligands have document_year.")
        return None
    sorted_idx = ligand_df.sort_values("document_year").index.to_numpy()
    n = len(sorted_idx)
    n_test = max(1, int(round(test_size * n)))
    n_val = max(1, int(round(val_size * n)))
    train_idx = sorted_idx[: n - n_val - n_test]
    val_idx = sorted_idx[n - n_val - n_test : n - n_test]
    test_idx = sorted_idx[n - n_test :]
    if len(train_idx) < 10 or len(val_idx) < 3 or len(test_idx) < 3:
        warnings.warn("Skipping time split: not enough rows after split.")
        return None
    return {"train": train_idx, "val": val_idx, "test": test_idx}


# ---------------------------------------------------------------------------
# Metrics and calibration
# ---------------------------------------------------------------------------


def js_distance_matrix_to_targets(Q: np.ndarray, P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return JS distance between query distributions Q[n,k] and target distributions P[t,k]."""
    Q = np.asarray(Q, dtype=np.float64) + eps
    P = np.asarray(P, dtype=np.float64) + eps
    Q = Q / Q.sum(axis=1, keepdims=True)
    P = P / P.sum(axis=1, keepdims=True)
    # For manageable dimensions, loop targets to avoid huge n x t x k allocation.
    out = np.zeros((Q.shape[0], P.shape[0]), dtype=np.float64)
    logQ = np.log(Q)
    for j in range(P.shape[0]):
        p = P[j]
        M = 0.5 * (Q + p[None, :])
        out[:, j] = 0.5 * np.sum(Q * (logQ - np.log(M)), axis=1) + 0.5 * np.sum(p[None, :] * (np.log(p)[None, :] - np.log(M)), axis=1)
    return out


def fit_global_platt(scores_val: np.ndarray, y_val: np.ndarray) -> Optional[LogisticRegression]:
    x = scores_val.reshape(-1, 1)
    y = y_val.reshape(-1)
    mask = np.isfinite(x[:, 0])
    x = x[mask]
    y = y[mask]
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x, y)
    return clf


def apply_global_platt(model: Optional[LogisticRegression], scores: np.ndarray) -> np.ndarray:
    if model is None:
        finite = np.isfinite(scores)
        s = scores.copy().astype(float)
        if finite.sum() == 0:
            return np.full_like(scores, 0.5, dtype=float)
        lo, hi = np.nanpercentile(s[finite], [5, 95])
        if hi <= lo:
            return np.full_like(scores, 0.5, dtype=float)
        z = (s - lo) / (hi - lo)
        return 1.0 / (1.0 + np.exp(-6.0 * (z - 0.5)))
    return model.predict_proba(scores.reshape(-1, 1))[:, 1].reshape(scores.shape)


def choose_threshold(proba_val: np.ndarray, y_val: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = (proba_val >= t).astype(int)
        try:
            f1 = f1_score(y_val, pred, average="micro", zero_division=0)
        except Exception:
            f1 = 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def topk_hit_rate(y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    if y_true.shape[0] == 0:
        return np.nan
    k = min(k, proba.shape[1])
    top = np.argsort(-proba, axis=1)[:, :k]
    hits = []
    for i in range(y_true.shape[0]):
        positives = np.flatnonzero(y_true[i] > 0)
        if len(positives) == 0:
            continue
        hits.append(bool(set(top[i]).intersection(positives)))
    return float(np.mean(hits)) if hits else np.nan


def mean_reciprocal_rank(y_true: np.ndarray, proba: np.ndarray) -> float:
    vals = []
    order = np.argsort(-proba, axis=1)
    for i in range(y_true.shape[0]):
        positives = set(np.flatnonzero(y_true[i] > 0))
        if not positives:
            continue
        rank = None
        for r, j in enumerate(order[i], start=1):
            if j in positives:
                rank = r
                break
        if rank is not None:
            vals.append(1.0 / rank)
    return float(np.mean(vals)) if vals else np.nan


def expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> tuple[float, pd.DataFrame]:
    p = proba.reshape(-1)
    y = y_true.reshape(-1)
    mask = np.isfinite(p)
    p = p[mask]
    y = y[mask]
    rows = []
    ece = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            rows.append({"bin": b, "bin_start": lo, "bin_end": hi, "n": 0, "mean_probability": np.nan, "empirical_frequency": np.nan})
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        weight = m.mean()
        ece += weight * abs(acc - conf)
        rows.append({"bin": b, "bin_start": lo, "bin_end": hi, "n": int(m.sum()), "mean_probability": conf, "empirical_frequency": acc})
    return float(ece), pd.DataFrame(rows)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    metrics: dict[str, float] = {"threshold": threshold}
    for k in [1, 3, 5]:
        metrics[f"top{k}_accuracy"] = topk_hit_rate(y_true, proba, k)
    metrics["mrr"] = mean_reciprocal_rank(y_true, proba)
    metrics["micro_f1"] = f1_score(y_true, pred, average="micro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, pred, average="macro", zero_division=0)
    metrics["subset_accuracy"] = float(np.mean(np.all(pred == y_true, axis=1))) if len(y_true) else np.nan
    try:
        metrics["micro_average_precision"] = average_precision_score(y_true, proba, average="micro")
    except Exception:
        metrics["micro_average_precision"] = np.nan
    try:
        metrics["macro_average_precision"] = average_precision_score(y_true, proba, average="macro")
    except Exception:
        metrics["macro_average_precision"] = np.nan
    try:
        metrics["micro_auroc"] = roc_auc_score(y_true, proba, average="micro")
    except Exception:
        metrics["micro_auroc"] = np.nan
    try:
        metrics["macro_auroc"] = roc_auc_score(y_true, proba, average="macro")
    except Exception:
        metrics["macro_auroc"] = np.nan
    metrics["brier"] = float(np.mean((proba - y_true) ** 2))
    ece, _ = expected_calibration_error(proba, y_true, n_bins=10)
    metrics["ece"] = ece
    return metrics


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------


@dataclass
class MethodOutput:
    scores_val: np.ndarray
    scores_test: np.ndarray
    extra: dict


def standardize_if_needed(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, descriptor_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Morgan binary fingerprints are usually best left as 0/1 for Tanimoto/SEA-like.
    if descriptor_name.lower() == "morgan":
        return X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr = scaler.fit_transform(X_train)
    return scaler.transform(X_train).astype(np.float32), scaler.transform(X_val).astype(np.float32), scaler.transform(X_test).astype(np.float32)


def shape_distribution_scores(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    n_clusters: int,
    seed: int,
) -> MethodOutput:
    k = int(min(max(2, n_clusters), max(2, len(X_train) - 1)))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=min(4096, max(100, len(X_train))))
    cluster_train = km.fit_predict(X_train)
    centers = km.cluster_centers_
    # Target distribution over clusters. Multi-label ligands contribute to every active target.
    n_targets = Y_train.shape[1]
    counts = np.ones((n_targets, k), dtype=np.float64) * 1e-3
    for i, c in enumerate(cluster_train):
        active = np.flatnonzero(Y_train[i] > 0)
        counts[active, c] += 1.0
    P = counts / counts.sum(axis=1, keepdims=True)

    def query_soft_membership(X: np.ndarray) -> np.ndarray:
        d = cdist(X, centers, metric="euclidean")
        # Temperature = median nearest-center distance; prevents over-sharp memberships.
        tau = float(np.median(np.min(d, axis=1))) if len(d) else 1.0
        tau = max(tau, 1e-6)
        return softmax(-d / tau, axis=1)

    Q_val = query_soft_membership(X_val)
    Q_test = query_soft_membership(X_test)
    D_val = js_distance_matrix_to_targets(Q_val, P)
    D_test = js_distance_matrix_to_targets(Q_test, P)
    # Higher score = more likely target.
    scores_val = -D_val
    scores_test = -D_test
    return MethodOutput(scores_val=scores_val, scores_test=scores_test, extra={"n_clusters": k})


def knn_scores(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    k: int,
    metric: str,
) -> MethodOutput:
    k = int(min(k, len(X_train)))
    if metric == "cosine":
        nn = NearestNeighbors(n_neighbors=k, metric="cosine")
        nn.fit(X_train)
        d_val, idx_val = nn.kneighbors(X_val)
        d_test, idx_test = nn.kneighbors(X_test)
        w_val = 1.0 / (d_val + 1e-6)
        w_test = 1.0 / (d_test + 1e-6)
    else:
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(X_train)
        d_val, idx_val = nn.kneighbors(X_val)
        d_test, idx_test = nn.kneighbors(X_test)
        w_val = 1.0 / (d_val + 1e-6)
        w_test = 1.0 / (d_test + 1e-6)

    def transfer(idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
        out = np.zeros((idx.shape[0], Y_train.shape[1]), dtype=np.float64)
        for i in range(idx.shape[0]):
            labels = Y_train[idx[i]]
            w = weights[i][:, None]
            out[i] = (labels * w).sum(axis=0) / max(float(w.sum()), 1e-12)
        return out

    return MethodOutput(scores_val=transfer(idx_val, w_val), scores_test=transfer(idx_test, w_test), extra={"k": k})


def tanimoto_matrix(A: np.ndarray, B: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    # A[n,b], B[m,b] binary or nonnegative. Tanimoto = dot/(|a|+|b|-dot).
    A = (A > 0).astype(np.float32)
    B = (B > 0).astype(np.float32)
    B_sum = B.sum(axis=1)[None, :]
    rows = []
    for start in range(0, len(A), batch_size):
        X = A[start:start + batch_size]
        dot = X @ B.T
        denom = X.sum(axis=1, keepdims=True) + B_sum - dot
        rows.append(dot / np.maximum(denom, 1e-12))
    return np.vstack(rows)


def sea_like_scores(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    top_m: int = 10,
) -> MethodOutput:
    # For each target, represent by its active training ligands; query score is mean of top-m Tanimoto similarities.
    target_indices = [np.flatnonzero(Y_train[:, j] > 0) for j in range(Y_train.shape[1])]

    def score_queries(Xq: np.ndarray) -> np.ndarray:
        out = np.zeros((len(Xq), Y_train.shape[1]), dtype=np.float32)
        for j, idx in enumerate(target_indices):
            if len(idx) == 0:
                continue
            sims = tanimoto_matrix(Xq, X_train[idx])
            m = min(top_m, sims.shape[1])
            if m == 1:
                out[:, j] = sims.max(axis=1)
            else:
                part = np.partition(sims, -m, axis=1)[:, -m:]
                out[:, j] = part.mean(axis=1)
        return out

    return MethodOutput(scores_val=score_queries(X_val), scores_test=score_queries(X_test), extra={"top_m": top_m})


# ---------------------------------------------------------------------------
# Evaluation loop and plotting
# ---------------------------------------------------------------------------


def filter_targets_by_train_support(
    Y: np.ndarray,
    split: dict[str, np.ndarray],
    target_names: list[str],
    min_train_ligands: int,
) -> tuple[np.ndarray, list[str]]:
    support = Y[split["train"]].sum(axis=0)
    keep = support >= min_train_ligands
    if keep.sum() < 2:
        raise ValueError(
            f"Fewer than two targets have >= {min_train_ligands} training ligands. "
            "Lower --min-train-ligands or use more data."
        )
    return keep, [t for t, ok in zip(target_names, keep) if ok]


def evaluate_one(
    descriptor_name: str,
    X: np.ndarray,
    Y: np.ndarray,
    ligand_df: pd.DataFrame,
    target_names: list[str],
    split_name: str,
    split: dict[str, np.ndarray],
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    keep_targets, kept_target_names = filter_targets_by_train_support(Y, split, target_names, args.min_train_ligands)
    Y2 = Y[:, keep_targets]

    train, val, test = split["train"], split["val"], split["test"]
    X_train_raw, X_val_raw, X_test_raw = X[train], X[val], X[test]
    Y_train, Y_val, Y_test = Y2[train], Y2[val], Y2[test]
    X_train, X_val, X_test = standardize_if_needed(X_train_raw, X_val_raw, X_test_raw, descriptor_name)

    methods: list[tuple[str, Callable[[], MethodOutput]]] = [
        ("shape_distribution", lambda: shape_distribution_scores(X_train, X_val, X_test, Y_train, args.n_clusters, args.seed)),
        ("knn", lambda: knn_scores(X_train, X_val, X_test, Y_train, args.knn_k, args.knn_metric)),
    ]
    if descriptor_name.lower() == "morgan" or args.allow_sea_like_for_all_descriptors:
        methods.append(("sea_like", lambda: sea_like_scores(X_train_raw, X_val_raw, X_test_raw, Y_train, args.sea_top_m)))

    metric_rows: list[dict] = []
    pred_tables: list[pd.DataFrame] = []
    cal_tables: list[pd.DataFrame] = []

    for method_name, runner in methods:
        try:
            mo = runner()
            cal = fit_global_platt(mo.scores_val, Y_val)
            proba_val = apply_global_platt(cal, mo.scores_val)
            proba_test = apply_global_platt(cal, mo.scores_test)
            threshold = choose_threshold(proba_val, Y_val)
            metrics = compute_metrics(Y_test, proba_test, threshold=threshold)
            row = {
                "split": split_name,
                "descriptor": descriptor_name,
                "method": method_name,
                "n_train_ligands": int(len(train)),
                "n_validation_ligands": int(len(val)),
                "n_test_ligands": int(len(test)),
                "n_targets": int(len(kept_target_names)),
                "calibration": "global_platt" if cal is not None else "fallback_sigmoid",
                **mo.extra,
                **metrics,
            }
            metric_rows.append(row)

            # Long-format predictions.
            pred_df = ligand_df.iloc[test][["ligand_id", "canonical_smiles", "document_year", "n_targets"]].reset_index(drop=True)
            for j, t in enumerate(kept_target_names):
                pred_df[f"true__{t}"] = Y_test[:, j]
                pred_df[f"prob__{t}"] = proba_test[:, j]
            top_order = np.argsort(-proba_test, axis=1)
            pred_df["top1_predicted_target"] = [kept_target_names[i[0]] for i in top_order]
            pred_df["top3_predicted_targets"] = [";".join(kept_target_names[j] for j in row[:min(3, len(kept_target_names))]) for row in top_order]
            pred_df["true_targets"] = [";".join(kept_target_names[j] for j in np.flatnonzero(Y_test[i] > 0)) for i in range(len(Y_test))]
            pred_df["split"] = split_name
            pred_df["descriptor"] = descriptor_name
            pred_df["method"] = method_name
            pred_tables.append(pred_df)

            ece, bins = expected_calibration_error(proba_test, Y_test, n_bins=10)
            bins["split"] = split_name
            bins["descriptor"] = descriptor_name
            bins["method"] = method_name
            cal_tables.append(bins)
        except Exception as exc:
            warnings.warn(f"Evaluation failed for {split_name}/{descriptor_name}/{method_name}: {exc}")
            metric_rows.append({
                "split": split_name,
                "descriptor": descriptor_name,
                "method": method_name,
                "status": f"failed: {type(exc).__name__}: {exc}",
            })
    return metric_rows, pred_tables, cal_tables


def save_metrics_heatmap(metrics: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if metrics.empty or "micro_f1" not in metrics:
        return
    clean = metrics.dropna(subset=["micro_f1"]).copy()
    if clean.empty:
        return
    clean["model"] = clean["split"].astype(str) + " | " + clean["descriptor"].astype(str) + " | " + clean["method"].astype(str)
    cols = [c for c in ["top1_accuracy", "top3_accuracy", "micro_f1", "macro_f1", "micro_average_precision", "brier", "ece"] if c in clean.columns]
    mat = clean.set_index("model")[cols]
    fig_h = max(4.5, 0.35 * len(mat) + 1.5)
    fig_w = max(7.5, 1.1 * len(cols) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("_", " ") for c in cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(mat)))
    ax.set_yticklabels(mat.index)
    ax.set_title("E3 assignment metrics across descriptors and methods")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("Metric value")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"fig_e3_assignment_metrics_heatmap.{fmt}", bbox_inches="tight")
    plt.close(fig)


def save_metric_bar(metrics: pd.DataFrame, metric: str, out_dir: Path, formats: list[str]) -> None:
    if metrics.empty or metric not in metrics:
        return
    clean = metrics.dropna(subset=[metric]).copy()
    if clean.empty:
        return
    clean["label"] = clean["split"].astype(str) + " | " + clean["descriptor"].astype(str) + " | " + clean["method"].astype(str)
    clean = clean.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(8.0, max(4.0, 0.36 * len(clean) + 1.4)))
    ax.barh(clean["label"], clean[metric])
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_ylabel("Split | descriptor | method")
    ax.set_title(f"E3 target-assignment benchmark: {metric.replace('_', ' ')}")
    for i, v in enumerate(clean[metric].values):
        ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"fig_e3_{metric}_by_method.{fmt}", bbox_inches="tight")
    plt.close(fig)


def save_reliability_plot(calibration: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path, formats: list[str]) -> None:
    if calibration.empty:
        return
    # Plot the best micro-F1 model per split to keep the figure readable.
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plotted = 0
    for split, g in metrics.dropna(subset=["micro_f1"]).groupby("split"):
        best = g.sort_values("micro_f1", ascending=False).iloc[0]
        m = (
            (calibration["split"] == best["split"]) &
            (calibration["descriptor"] == best["descriptor"]) &
            (calibration["method"] == best["method"])
        )
        bins = calibration.loc[m].dropna(subset=["mean_probability", "empirical_frequency"])
        if bins.empty:
            continue
        ax.plot(bins["mean_probability"], bins["empirical_frequency"], marker="o", label=f"{split}: {best['descriptor']} / {best['method']}")
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed positive frequency")
    ax.set_title("Calibration of multi-label target probabilities")
    ax.legend(frameon=False)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"fig_e3_reliability_diagram.{fmt}", bbox_inches="tight")
    plt.close(fig)


def save_target_distribution(ligand_df: pd.DataFrame, label_sets: list[list[str]], out_dir: Path, formats: list[str]) -> pd.DataFrame:
    rows = []
    for labels in label_sets:
        for t in labels:
            rows.append({"target": t})
    counts = pd.DataFrame(rows)["target"].value_counts().sort_values(ascending=True)
    counts_df = counts.rename_axis("target").reset_index(name="n_ligands")
    counts_df.to_csv(out_dir / "target_label_counts.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.35 * len(counts) + 1.4)))
    ax.barh(counts.index.astype(str), counts.values)
    ax.set_xlabel("Number of active ligands")
    ax.set_ylabel("Target")
    ax.set_title("Target-label coverage in the assignment dataset")
    for i, v in enumerate(counts.values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(out_dir / f"fig_e3_target_label_counts.{fmt}", bbox_inches="tight")
    plt.close(fig)
    return counts_df


def save_review_pdf(out_dir: Path) -> None:
    pngs = [
        "fig_e3_target_label_counts.png",
        "fig_e3_micro_f1_by_method.png",
        "fig_e3_top3_accuracy_by_method.png",
        "fig_e3_assignment_metrics_heatmap.png",
        "fig_e3_reliability_diagram.png",
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run E3 pocket-free multi-label target assignment benchmark.")
    p.add_argument("--metadata-csv", required=True, help="Ligand-target metadata CSV, e.g. full_sdfs.csv or activities.csv.")
    p.add_argument("--morgan-fp-csv", default=None, help="Optional row-aligned Morgan fingerprint CSV, e.g. processed_sdfs-2.csv.")
    p.add_argument("--precomputed-descriptor", action="append", default=[], help="Additional descriptor in form name:path.csv. Can be repeated.")
    p.add_argument("--descriptors", nargs="+", default=["morgan"], help="Descriptors to evaluate: morgan usr usrcat bdc/tsc_bdc plus precomputed names.")
    p.add_argument("--out", default="e3_assignment_results", help="Output directory.")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"], help="Figure formats.")

    p.add_argument("--ligand-col", default=None)
    p.add_argument("--smiles-col", default=None)
    p.add_argument("--target-col", default=None)
    p.add_argument("--year-col", default=None)
    p.add_argument("--activity-col", default=None)
    p.add_argument("--min-activity-value", type=float, default=None, help="Optional filter for pChEMBL-like activity columns.")

    p.add_argument("--morgan-radius", type=int, default=2)
    p.add_argument("--morgan-bits", type=int, default=1024)
    p.add_argument("--tsc-core-path", default=None, help="Local path to cloned TSC/core directory.")
    p.add_argument("--tsc-function", default=None, help="Descriptor function as module:function, e.g. descriptors.bdc:compute_bdc_descriptor.")

    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--min-train-ligands", type=int, default=5)
    p.add_argument("--n-clusters", type=int, default=32)
    p.add_argument("--knn-k", type=int, default=15)
    p.add_argument("--knn-metric", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--sea-top-m", type=int, default=10)
    p.add_argument("--allow-sea-like-for-all-descriptors", action="store_true")
    p.add_argument("--skip-time-split", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(Path(args.metadata_csv), args)
    ligand_df, label_sets = collapse_to_ligands(meta)
    ligand_df.to_csv(out_dir / "ligands_unique.csv", index=False)
    meta.to_csv(out_dir / "ligand_target_labels_long.csv", index=False)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(label_sets).astype(int)
    target_names = list(mlb.classes_)
    pd.DataFrame(Y, columns=target_names).assign(ligand_id=ligand_df["ligand_id"].values).to_csv(out_dir / "ligand_target_matrix.csv", index=False)
    save_target_distribution(ligand_df, label_sets, out_dir, args.formats)

    descriptors = load_descriptors(args, ligand_df)
    desc_summary = []
    for name, X in descriptors.items():
        if len(X) != len(ligand_df):
            raise ValueError(f"Descriptor {name} has {len(X)} rows, expected {len(ligand_df)}")
        desc_summary.append({
            "descriptor": name,
            "n_ligands": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "finite_fraction": float(np.isfinite(X).mean()),
            "zero_row_fraction": float((np.abs(X).sum(axis=1) == 0).mean()),
        })
    pd.DataFrame(desc_summary).to_csv(out_dir / "descriptor_summary.csv", index=False)

    splits: dict[str, dict[str, np.ndarray]] = {}
    splits["scaffold"] = split_by_groups(ligand_df, label_sets, args.test_size, args.val_size, args.seed)
    if not args.skip_time_split:
        ts = split_by_time(ligand_df, args.test_size, args.val_size)
        if ts is not None:
            splits["time"] = ts

    split_rows = []
    for split_name, sp in splits.items():
        for part in ["train", "val", "test"]:
            idx = sp[part]
            split_rows.append({
                "split": split_name,
                "partition": part,
                "n_ligands": int(len(idx)),
                "n_label_assignments": int(Y[idx].sum()),
                "n_targets_present": int((Y[idx].sum(axis=0) > 0).sum()),
                "min_year": float(np.nanmin(ligand_df.iloc[idx]["document_year"])) if ligand_df.iloc[idx]["document_year"].notna().any() else np.nan,
                "max_year": float(np.nanmax(ligand_df.iloc[idx]["document_year"])) if ligand_df.iloc[idx]["document_year"].notna().any() else np.nan,
            })
    pd.DataFrame(split_rows).to_csv(out_dir / "split_summary.csv", index=False)

    metric_rows: list[dict] = []
    pred_tables: list[pd.DataFrame] = []
    cal_tables: list[pd.DataFrame] = []
    for split_name, split in splits.items():
        for descriptor_name, X in descriptors.items():
            rows, preds, cals = evaluate_one(
                descriptor_name=descriptor_name,
                X=X,
                Y=Y,
                ligand_df=ligand_df,
                target_names=target_names,
                split_name=split_name,
                split=split,
                args=args,
                out_dir=out_dir,
            )
            metric_rows.extend(rows)
            pred_tables.extend(preds)
            cal_tables.extend(cals)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out_dir / "e3_assignment_metrics.csv", index=False)
    if pred_tables:
        pd.concat(pred_tables, ignore_index=True).to_csv(out_dir / "e3_test_predictions.csv", index=False)
    if cal_tables:
        calibration = pd.concat(cal_tables, ignore_index=True)
        calibration.to_csv(out_dir / "e3_calibration_bins.csv", index=False)
    else:
        calibration = pd.DataFrame()

    save_metric_bar(metrics, "micro_f1", out_dir, args.formats)
    save_metric_bar(metrics, "top3_accuracy", out_dir, args.formats)
    save_metric_bar(metrics, "micro_average_precision", out_dir, args.formats)
    save_metrics_heatmap(metrics, out_dir, args.formats)
    save_reliability_plot(calibration, metrics, out_dir, args.formats)
    save_review_pdf(out_dir)

    # Manuscript-ready summary and LaTeX table stub.
    good = metrics.dropna(subset=["micro_f1"]) if "micro_f1" in metrics else pd.DataFrame()
    lines = ["# E3 pocket-free multi-label target-assignment benchmark", ""]
    lines.append("## What was evaluated")
    lines.append("This benchmark predicts active targets for unseen ligands without using pocket structures at prediction time. It compares the proposed target-distribution assignment method against kNN and a SEA-like ligand-set baseline. SwissTargetPrediction is intentionally excluded in this run.")
    lines.append("")
    lines.append("## Input coverage")
    lines.append(f"- Unique ligands: {len(ligand_df):,}")
    lines.append(f"- Targets: {len(target_names):,}")
    lines.append(f"- Multi-target ligands: {(ligand_df['n_targets'] > 1).sum():,}")
    lines.append(f"- Descriptors: {', '.join(descriptors.keys())}")
    lines.append(f"- Splits: {', '.join(splits.keys())}")
    lines.append("")
    lines.append("## Best observed models")
    if not good.empty:
        for split_name, g in good.groupby("split"):
            best = g.sort_values("micro_f1", ascending=False).iloc[0]
            lines.append(
                f"- {split_name}: best micro-F1 = {best['micro_f1']:.3f} using "
                f"{best['descriptor']} / {best['method']} "
                f"(Top-1={best.get('top1_accuracy', np.nan):.3f}, Top-3={best.get('top3_accuracy', np.nan):.3f}, "
                f"ECE={best.get('ece', np.nan):.3f})."
            )
    else:
        lines.append("No successful model metrics were produced. Check warnings and input columns.")
    lines.append("")
    lines.append("## Main files")
    for fn in [
        "e3_assignment_metrics.csv",
        "e3_test_predictions.csv",
        "e3_calibration_bins.csv",
        "split_summary.csv",
        "descriptor_summary.csv",
        "selected_figures_review.pdf",
    ]:
        lines.append(f"- `{fn}`")
    (out_dir / "analysis_report.md").write_text("\n".join(lines))

    latex = r"""
% E3 quantitative summary table. Fill values from e3_assignment_metrics.csv.
\begin{table}[ht]
\centering
\caption{Pocket-free multi-label target-assignment performance under scaffold and time splits. The proposed shape-distribution assignment is compared with kNN and a SEA-like ligand-set baseline. SwissTargetPrediction was not included in this run.}
\label{tab:e3-assignment-metrics}
\begin{tabular}{llrrrrrr}
\hline
Split & Method & Top-1 & Top-3 & Micro-F1 & Macro-F1 & mAP & ECE \\
\hline
% scaffold & shape-distribution & ... & ... & ... & ... & ... & ... \\
% scaffold & kNN & ... & ... & ... & ... & ... & ... \\
% scaffold & SEA-like & ... & ... & ... & ... & ... & ... \\
% time & shape-distribution & ... & ... & ... & ... & ... & ... \\
% time & kNN & ... & ... & ... & ... & ... & ... \\
% time & SEA-like & ... & ... & ... & ... & ... & ... \\
\hline
\end{tabular}
\end{table}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.85\linewidth]{figures/fig_e3_micro_f1_by_method.pdf}
    \caption{Pocket-free target assignment under scaffold and time splits. Bars report micro-F1 for the proposed target-distribution assignment and ligand-neighbor baselines.}
    \label{fig:e3-assignment-performance}
\end{figure}

\begin{figure}[ht]
    \centering
    \includegraphics[width=0.65\linewidth]{figures/fig_e3_reliability_diagram.pdf}
    \caption{Calibration of multi-label target probabilities. The diagonal indicates perfect calibration.}
    \label{fig:e3-calibration}
\end{figure}
"""
    (out_dir / "latex_e3_snippet.tex").write_text(latex)

    print(f"Done. E3 results saved to: {out_dir}")
    print(f"Metrics: {out_dir / 'e3_assignment_metrics.csv'}")
    print(f"Report:  {out_dir / 'analysis_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
