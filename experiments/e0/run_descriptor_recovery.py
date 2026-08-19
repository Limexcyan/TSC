#!/usr/bin/env python3
"""
E0 descriptor-recovery table for TaSC.

Builds the table requested in E0:

descriptor | distance metric | pocket Pearson r | pocket Spearman rho | Mantel rho | assignment top-1 | assignment macro-F1

Inputs
------
- metadata CSV with ligand, SMILES, target, and sequence columns. Project default: full_sdfs.csv
- optional row-aligned Morgan fingerprint CSV. Project default: processed_sdfs-2.csv
- optional precomputed descriptor CSVs in format name:path.csv, with ligand_id column or row-aligned to metadata
- optional TSC BDC function, if you want to compute BDC through local TSC/core

The script computes:
1. per-target ligand-cluster distributions for each descriptor;
2. target-target ligand-distribution distances, e.g. JS divergence;
3. pocket distances from target sequences using pairwise global alignment identity;
4. Pearson/Spearman and Mantel correlations between descriptor-derived target distances and pocket distances;
5. scaffold-split assignment metrics for the same descriptors using nearest target-distribution probabilities.

Notes
-----
- USR/USRCAT/PCA-SH require RDKit and generated conformers.
- PCA-SH here is a lightweight PCA-aligned spherical-harmonic descriptor from RDKit conformers.
- BDC requires a local TSC function or a precomputed BDC descriptor matrix.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import math
import os
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import pearsonr, spearmanr
from scipy.special import softmax, sph_harm_y
from scipy.optimize import linear_sum_assignment

from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

try:
    from Bio import Align
except Exception:
    Align = None


def first_existing(columns, candidates):
    cols = list(columns)
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None


def detect_columns(df):
    return {
        "ligand": first_existing(df.columns, ["ligand_id", "chembl_id", "molecule_chembl_id", "zinc_id", "compound_id", "ID", "id"]),
        "smiles": first_existing(df.columns, ["canonical_smiles", "smiles", "SMILES", "canonical_smile"]),
        "target": first_existing(df.columns, ["target", "target_id", "target_name", "receptor"]),
        "seq": first_existing(df.columns, ["seq", "sequence", "protein_sequence", "target_sequence"]),
    }


def normalize_target(x):
    if pd.isna(x): return ""
    s = str(x).strip()
    repl = {"5H2A":"5HT2A", "5-HT1A":"5HT1A", "5-HT1B":"5HT1B", "5-HT1D":"5HT1D", "5-HT2A":"5HT2A"}
    return repl.get(s, s)


def load_metadata(path, args):
    df = pd.read_csv(path)
    det = detect_columns(df)
    ligand_col = args.ligand_col or det["ligand"]
    smiles_col = args.smiles_col or det["smiles"]
    target_col = args.target_col or det["target"]
    seq_col = args.sequence_col or det["seq"]
    if ligand_col is None or smiles_col is None or target_col is None:
        raise ValueError(f"Could not detect ligand/smiles/target columns. Available: {list(df.columns)}")
    out = pd.DataFrame({
        "ligand_id": df[ligand_col].astype(str),
        "canonical_smiles": df[smiles_col].astype(str),
        "target": df[target_col].map(normalize_target),
    })
    out["sequence"] = df[seq_col].astype(str) if seq_col else ""
    out = out[(out["target"] != "") & out["ligand_id"].notna()].drop_duplicates(["ligand_id", "target"])
    return out, df, ligand_col


def collapse_ligands(meta):
    rows, labels = [], []
    for lig, g in meta.groupby("ligand_id", sort=False):
        smiles = next((s for s in g["canonical_smiles"].astype(str) if s and s.lower() != "nan"), "")
        rows.append({"ligand_id": lig, "canonical_smiles": smiles, "n_targets": g["target"].nunique()})
        labels.append(sorted(g["target"].unique()))
    return pd.DataFrame(rows), labels


def import_rdkit():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, rdMolDescriptors
    from rdkit.Chem.Scaffolds import MurckoScaffold
    return Chem, AllChem, rdMolDescriptors, MurckoScaffold, DataStructs


def load_morgan_csv(path, raw_df, ligand_col, ligand_df):
    Xraw = pd.read_csv(path).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if len(Xraw) == len(ligand_df):
        return Xraw.to_numpy(dtype=np.float32)
    if len(Xraw) != len(raw_df):
        raise ValueError(f"Morgan CSV rows {len(Xraw)} do not match metadata rows {len(raw_df)} or unique ligands {len(ligand_df)}")
    Xraw["ligand_id"] = raw_df[ligand_col].astype(str).values
    Xagg = Xraw.groupby("ligand_id", sort=False).mean()
    X = ligand_df[["ligand_id"]].merge(Xagg, left_on="ligand_id", right_index=True, how="left").drop(columns=["ligand_id"])
    return X.fillna(0.0).to_numpy(dtype=np.float32)


def morgan_from_smiles(smiles, radius=2, nbits=1024):
    Chem, AllChem, _, _, DataStructs = import_rdkit()
    X = np.zeros((len(smiles), nbits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None: continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros((nbits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X


def mol_with_conformer(smi, seed=1):
    Chem, AllChem, _, _, _ = import_rdkit()
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
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=150)
    except Exception:
        pass
    return mol


def pca_sh_descriptor_from_mol(mol, lmax=4):
    # Lightweight PCA-aligned spherical harmonic descriptor.
    conf = mol.GetConformer()
    atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    idx = [a.GetIdx() for a in atoms]
    if len(idx) < 3:
        return np.zeros(((lmax + 1) ** 2 * 2,), dtype=float)
    coords = np.asarray([conf.GetAtomPosition(i) for i in idx], dtype=float)
    coords -= coords.mean(axis=0, keepdims=True)
    # PCA alignment.
    try:
        _, _, vt = np.linalg.svd(coords, full_matrices=False)
        coords = coords @ vt.T
    except Exception:
        pass
    r = np.linalg.norm(coords, axis=1)
    r_safe = np.maximum(r, 1e-8)
    theta = np.arccos(np.clip(coords[:, 2] / r_safe, -1.0, 1.0))
    phi = np.arctan2(coords[:, 1], coords[:, 0])
    weights = r_safe / max(np.mean(r_safe), 1e-8)
    feats = []
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            y = sph_harm_y(l, m, theta, phi)
            val = np.mean(weights * y)
            feats.extend([float(np.real(val)), float(np.imag(val))])
    return np.asarray(feats, dtype=float)


def compute_one_3d(args_tuple):
    smi, seed, requested = args_tuple
    _, _, rdMolDescriptors, _, _ = import_rdkit()
    mol = mol_with_conformer(smi, seed=seed)
    out = {}
    if mol is None:
        for r in requested: out[r] = None
        return out
    for name in requested:
        try:
            if name == "usr":
                out[name] = np.asarray(rdMolDescriptors.GetUSR(mol), dtype=np.float32)
            elif name == "usrcat":
                out[name] = np.asarray(rdMolDescriptors.GetUSRCAT(mol), dtype=np.float32)
            elif name in ("pca_sh", "sh", "pca-sh"):
                out[name] = pca_sh_descriptor_from_mol(mol).astype(np.float32)
        except Exception:
            out[name] = None
    return out


def compute_3d_descriptors(smiles, requested, seed=1, n_jobs=1):
    requested = [r.lower() for r in requested]
    rows_by_name = {r: [] for r in requested}
    tasks = [(s, seed, requested) for s in smiles]
    if n_jobs and n_jobs > 1:
        with Pool(processes=n_jobs) as pool:
            iterator = pool.imap(compute_one_3d, tasks, chunksize=64)
            for out in iterator:
                for r in requested: rows_by_name[r].append(out.get(r))
    else:
        for t in tasks:
            out = compute_one_3d(t)
            for r in requested: rows_by_name[r].append(out.get(r))
    matrices = {}
    for name, rows in rows_by_name.items():
        dim = max((len(x) for x in rows if x is not None), default=0)
        if dim == 0:
            warnings.warn(f"No {name} descriptors were computed.")
            continue
        X = np.zeros((len(rows), dim), dtype=np.float32)
        failed = 0
        for i, x in enumerate(rows):
            if x is None or len(x) == 0:
                failed += 1
                continue
            arr = np.asarray(x, dtype=np.float32).ravel()
            X[i, :min(dim, len(arr))] = arr[:dim]
        if failed:
            warnings.warn(f"{name}: {failed} molecules failed and were encoded as zero vectors.")
        matrices[name] = X
    return matrices


def load_precomputed_descriptor(spec, raw_df, ligand_col, ligand_df):
    # spec = name:path.csv
    name, path = spec.split(":", 1)
    df = pd.read_csv(path)
    id_col = first_existing(df.columns, ["ligand_id", "chembl_id", "ID", "zinc_id"])
    if id_col:
        desc_cols = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
        X = ligand_df[["ligand_id"]].merge(df[[id_col] + desc_cols], left_on="ligand_id", right_on=id_col, how="left")
        return name, X[desc_cols].fillna(0.0).to_numpy(dtype=np.float32)
    X = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if len(X) == len(ligand_df):
        return name, X.to_numpy(dtype=np.float32)
    if len(X) == len(raw_df):
        X["ligand_id"] = raw_df[ligand_col].astype(str).values
        Xagg = X.groupby("ligand_id", sort=False).mean()
        out = ligand_df[["ligand_id"]].merge(Xagg, left_on="ligand_id", right_index=True, how="left").drop(columns=["ligand_id"])
        return name, out.fillna(0.0).to_numpy(dtype=np.float32)
    raise ValueError(f"Cannot align descriptor {name}:{path}")


def sequence_distance_matrix(meta):
    seqs = meta.groupby("target")["sequence"].first().to_dict()
    targets = sorted(seqs)
    n = len(targets)
    D = np.zeros((n, n), dtype=float)
    if Align is None:
        warnings.warn("Biopython unavailable: using simple length-normalized mismatch distance")
    else:
        aligner = Align.PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 1.0
        aligner.mismatch_score = 0.0
        aligner.open_gap_score = -1.0
        aligner.extend_gap_score = -0.1
    for i in range(n):
        for j in range(i + 1, n):
            a = str(seqs[targets[i]]).replace("\n", "")
            b = str(seqs[targets[j]]).replace("\n", "")
            if Align is not None:
                score = aligner.score(a, b)
                identity_like = score / max(len(a), len(b), 1)
                dist = 1.0 - max(0.0, min(1.0, identity_like))
            else:
                m = min(len(a), len(b))
                ident = sum(aa == bb for aa, bb in zip(a[:m], b[:m])) / max(max(len(a), len(b)), 1)
                dist = 1.0 - ident
            D[i, j] = D[j, i] = dist
    return targets, D


def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))


def wasserstein_discrete(p, q, centers):
    # Approximate 1-Wasserstein/EMD between cluster distributions using center distances.
    p = np.asarray(p, dtype=float); q = np.asarray(q, dtype=float)
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12)
    # Greedy transport after sorting clusters along first principal coordinate.
    if centers.shape[0] < 2:
        return 0.0
    try:
        _, _, vt = np.linalg.svd(centers - centers.mean(axis=0), full_matrices=False)
        coord = (centers - centers.mean(axis=0)) @ vt[0]
    except Exception:
        coord = centers[:, 0]
    order = np.argsort(coord)
    x = coord[order]
    pp = p[order].copy(); qq = q[order].copy()
    # 1D Wasserstein formula for discrete histograms along PCA axis.
    cdf = np.cumsum(pp - qq)
    dx = np.diff(x, prepend=x[0])
    return float(np.sum(np.abs(cdf) * np.abs(dx)))


def target_distribution_distances(X, ligand_df, label_sets, target_names, n_clusters=32, seed=42):
    Xs = X.astype(np.float32)
    # standardize continuous descriptors; keep binary Morgan as is if only 0/1-ish.
    uniq_vals = np.unique(Xs[:min(1000, len(Xs))])
    binaryish = len(uniq_vals) <= 3 and set(np.round(uniq_vals, 3)).issubset({0.0, 1.0})
    if not binaryish:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xs).astype(np.float32)
    k = int(min(max(2, n_clusters), max(2, len(Xs) - 1)))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=min(4096, max(100, len(Xs))))
    clusters = km.fit_predict(Xs)
    target_index = {t: i for i, t in enumerate(target_names)}
    counts = np.ones((len(target_names), k), dtype=float) * 1e-3
    for i, labs in enumerate(label_sets):
        for t in labs:
            if t in target_index:
                counts[target_index[t], clusters[i]] += 1.0
    P = counts / counts.sum(axis=1, keepdims=True)
    D_js = np.zeros((len(target_names), len(target_names)), dtype=float)
    D_w = np.zeros_like(D_js)
    for i in range(len(target_names)):
        for j in range(i+1, len(target_names)):
            D_js[i,j] = D_js[j,i] = js_divergence(P[i], P[j])
            D_w[i,j] = D_w[j,i] = wasserstein_discrete(P[i], P[j], km.cluster_centers_)
    return {"JS": D_js, "Wasserstein1D": D_w}, {"clusters": clusters, "centers": km.cluster_centers_, "target_distributions": P}


def mantel_spearman(A, B, n_perm=999, seed=42):
    idx = np.triu_indices_from(A, k=1)
    a = A[idx]; b = B[idx]
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]; b = b[mask]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan, np.nan
    rho = spearmanr(a, b).statistic
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n)
        bp = B[perm][:, perm]
        bb = bp[idx][mask]
        rr = spearmanr(a, bb).statistic
        if abs(rr) >= abs(rho):
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return float(rho), float(p)


def pair_correlations(D_desc, D_pocket):
    idx = np.triu_indices_from(D_desc, k=1)
    x = D_desc[idx]
    y = D_pocket[idx]
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan, np.nan, np.nan
    pr = pearsonr(x, y)
    sr = spearmanr(x, y)
    return float(pr.statistic), float(pr.pvalue), float(sr.statistic), float(sr.pvalue)


def murcko_scaffolds(smiles):
    Chem, _, _, MurckoScaffold, _ = import_rdkit()
    out = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            out.append(f"invalid_{i}"); continue
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        out.append(Chem.MolToSmiles(scaf, isomericSmiles=False) if scaf is not None else f"empty_{i}")
    return np.asarray(out)


def split_scaffold(ligand_df, test_size=0.15, val_size=0.15, seed=42):
    idx = np.arange(len(ligand_df))
    groups = murcko_scaffolds(ligand_df["canonical_smiles"].tolist())
    if len(np.unique(groups)) < 3:
        trv, te = train_test_split(idx, test_size=test_size, random_state=seed)
        tr, va = train_test_split(trv, test_size=val_size/(1-test_size), random_state=seed+1)
        return {"train": tr, "val": va, "test": te}
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trv, te = next(gss.split(idx, groups=groups))
    groups_trv = groups[trv]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=seed+1)
    tr_rel, va_rel = next(gss2.split(trv, groups=groups_trv))
    return {"train": trv[tr_rel], "val": trv[va_rel], "test": te}


def fit_platt(scores_val, Y_val):
    x = scores_val.reshape(-1, 1); y = Y_val.reshape(-1)
    mask = np.isfinite(x[:, 0])
    x = x[mask]; y = y[mask]
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(x, y)
    return clf


def apply_platt(model, scores):
    if model is not None:
        return model.predict_proba(scores.reshape(-1,1))[:,1].reshape(scores.shape)
    s = scores.astype(float)
    lo, hi = np.nanpercentile(s, [5,95])
    if hi <= lo: return np.full_like(s, 0.5, dtype=float)
    z = (s - lo) / (hi - lo)
    return 1 / (1 + np.exp(-6*(z - 0.5)))


def assign_shape_distribution(X, label_sets, target_names, split, n_clusters=32, seed=42):
    mlb = MultiLabelBinarizer(classes=target_names)
    Y = mlb.fit_transform(label_sets).astype(int)
    tr, va, te = split["train"], split["val"], split["test"]
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    # standardize if continuous
    binaryish = set(np.unique(Xtr[:min(1000,len(Xtr))]).round(3)).issubset({0.0,1.0})
    if not binaryish:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr); Xva = scaler.transform(Xva); Xte = scaler.transform(Xte)
    k = int(min(max(2, n_clusters), max(2, len(Xtr)-1)))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=min(4096, max(100, len(Xtr))))
    cl_tr = km.fit_predict(Xtr)
    counts = np.ones((len(target_names), k), dtype=float) * 1e-3
    for ii, c in enumerate(cl_tr):
        for t in np.flatnonzero(Y[tr][ii] > 0):
            counts[t, c] += 1.0
    P = counts / counts.sum(axis=1, keepdims=True)
    def q_membership(Xq):
        d = cdist(Xq, km.cluster_centers_)
        tau = max(float(np.median(np.min(d, axis=1))), 1e-6)
        return softmax(-d/tau, axis=1)
    def score(Xq):
        Q = q_membership(Xq)
        D = np.zeros((len(Xq), len(target_names)))
        for i in range(len(Xq)):
            for j in range(len(target_names)):
                D[i,j] = js_divergence(Q[i], P[j])
        return -D
    Sva, Ste = score(Xva), score(Xte)
    platt = fit_platt(Sva, Y[va])
    Pte = apply_platt(platt, Ste)
    Yte = Y[te]
    top1 = np.mean([Yte[i, np.argmax(Pte[i])] > 0 for i in range(len(te)) if Yte[i].sum() > 0])
    # threshold chosen on val
    Pva = apply_platt(platt, Sva)
    best_t, best_f = 0.5, -1
    for th in np.linspace(0.05, 0.95, 19):
        f = f1_score(Y[va], (Pva >= th).astype(int), average="micro", zero_division=0)
        if f > best_f:
            best_f, best_t = f, float(th)
    macro_f1 = f1_score(Yte, (Pte >= best_t).astype(int), average="macro", zero_division=0)
    return float(top1), float(macro_f1), float(best_t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", default="full_sdfs.csv")
    ap.add_argument("--morgan-fp-csv", default=None)
    ap.add_argument("--precomputed-descriptor", action="append", default=[], help="name:path.csv; may be repeated")
    ap.add_argument("--descriptors", nargs="+", default=["morgan", "usr", "usrcat", "pca_sh"], help="morgan usr usrcat pca_sh plus precomputed names")
    ap.add_argument("--ligand-col", default=None)
    ap.add_argument("--smiles-col", default=None)
    ap.add_argument("--target-col", default=None)
    ap.add_argument("--sequence-col", default=None)
    ap.add_argument("--out", default="e0_descriptor_recovery_results")
    ap.add_argument("--n-clusters", type=int, default=32)
    ap.add_argument("--mantel-permutations", type=int, default=999)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--max-ligands", type=int, default=None, help="Optional debugging subsample of unique ligands")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta, raw_df, ligand_col = load_metadata(Path(args.metadata_csv), args)
    ligand_df, label_sets = collapse_ligands(meta)
    if args.max_ligands is not None and args.max_ligands < len(ligand_df):
        rng = np.random.default_rng(args.seed)
        keep_idx = np.sort(rng.choice(len(ligand_df), size=args.max_ligands, replace=False))
        keep_ligs = set(ligand_df.iloc[keep_idx]["ligand_id"])
        ligand_df = ligand_df.iloc[keep_idx].reset_index(drop=True)
        label_sets = [labs for lig, labs in zip(collapse_ligands(meta)[0]["ligand_id"], collapse_ligands(meta)[1]) if lig in keep_ligs]
    target_names = sorted(meta["target"].unique())

    # Match sequence matrix target order and target_names.
    seq_targets, Dp_all = sequence_distance_matrix(meta)
    target_names = [t for t in target_names if t in seq_targets]
    idx_seq = [seq_targets.index(t) for t in target_names]
    D_pocket = Dp_all[np.ix_(idx_seq, idx_seq)]

    descriptors = {}
    requested = [d.lower() for d in args.descriptors]
    if "morgan" in requested:
        if args.morgan_fp_csv:
            descriptors["Morgan"] = load_morgan_csv(Path(args.morgan_fp_csv), raw_df, ligand_col, ligand_df)
        else:
            descriptors["Morgan"] = morgan_from_smiles(ligand_df["canonical_smiles"].tolist())
    rdkit_requested = [d for d in requested if d in {"usr", "usrcat", "pca_sh", "sh", "pca-sh"}]
    if rdkit_requested:
        mats = compute_3d_descriptors(ligand_df["canonical_smiles"].tolist(), rdkit_requested, seed=args.seed, n_jobs=args.n_jobs)
        for k, X in mats.items():
            name = {"usr":"USR", "usrcat":"USRCAT", "pca_sh":"SH/PCA-SH", "sh":"SH/PCA-SH", "pca-sh":"SH/PCA-SH"}.get(k, k)
            descriptors[name] = X
    for spec in args.precomputed_descriptor:
        name, X = load_precomputed_descriptor(spec, raw_df, ligand_col, ligand_df)
        descriptors[name] = X

    # Filter label sets to targets present in sequence matrix.
    label_sets2 = [[t for t in labs if t in target_names] for labs in label_sets]
    keep = [i for i, labs in enumerate(label_sets2) if labs]
    if len(keep) < len(ligand_df):
        ligand_df = ligand_df.iloc[keep].reset_index(drop=True)
        label_sets2 = [label_sets2[i] for i in keep]
        descriptors = {name: X[keep] for name, X in descriptors.items()}

    split = split_scaffold(ligand_df, seed=args.seed)
    rows = []
    pair_rows = []
    for desc_name, X in descriptors.items():
        dists, extra = target_distribution_distances(X, ligand_df, label_sets2, target_names, n_clusters=args.n_clusters, seed=args.seed)
        top1, macro_f1, threshold = assign_shape_distribution(X, label_sets2, target_names, split, n_clusters=args.n_clusters, seed=args.seed)
        for metric_name, Ddesc in dists.items():
            pr, pp, sr, sp = pair_correlations(Ddesc, D_pocket)
            mr, mp = mantel_spearman(Ddesc, D_pocket, n_perm=args.mantel_permutations, seed=args.seed)
            rows.append({
                "descriptor": desc_name,
                "distance_metric": metric_name,
                "pocket_pearson_r": pr,
                "pocket_pearson_p": pp,
                "pocket_spearman_rho": sr,
                "pocket_spearman_p": sp,
                "mantel_rho": mr,
                "mantel_p": mp,
                "assignment_top1": top1,
                "assignment_macro_f1": macro_f1,
                "assignment_threshold": threshold,
                "n_targets": len(target_names),
                "n_ligands": len(ligand_df),
                "n_pairs": int(len(target_names)*(len(target_names)-1)/2),
            })
            idx = np.triu_indices_from(Ddesc, k=1)
            for a,b,dd,dp in zip(np.array(target_names)[idx[0]], np.array(target_names)[idx[1]], Ddesc[idx], D_pocket[idx]):
                pair_rows.append({"descriptor": desc_name, "distance_metric": metric_name, "target_1": a, "target_2": b, "descriptor_distance": dd, "pocket_distance": dp})

    table = pd.DataFrame(rows)
    order = ["Morgan", "USR", "USRCAT", "BDC", "SH/PCA-SH"]
    table["descriptor_order"] = table["descriptor"].map({d:i for i,d in enumerate(order)}).fillna(99)
    table = table.sort_values(["descriptor_order", "distance_metric"]).drop(columns=["descriptor_order"])
    table.to_csv(out / "e0_descriptor_recovery_table.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(out / "e0_pairwise_descriptor_vs_pocket_distances.csv", index=False)
    ligand_df.to_csv(out / "e0_ligands_used.csv", index=False)
    pd.DataFrame({"target": target_names}).to_csv(out / "e0_targets_used.csv", index=False)

    # LaTeX table with core requested columns.
    latex_cols = ["descriptor", "distance_metric", "pocket_pearson_r", "pocket_spearman_rho", "mantel_rho", "assignment_top1", "assignment_macro_f1"]
    latex_df = table[latex_cols].copy()
    for c in latex_cols[2:]:
        latex_df[c] = latex_df[c].map(lambda x: "--" if pd.isna(x) else f"{x:.3f}")
    latex = latex_df.to_latex(index=False, escape=False, column_format="llrrrrr")
    (out / "e0_descriptor_recovery_table.tex").write_text(latex)

    lines = ["# E0 descriptor recovery analysis", ""]
    lines.append(f"Unique ligands used: {len(ligand_df):,}")
    lines.append(f"Targets used: {len(target_names):,} ({', '.join(target_names)})")
    lines.append(f"Target pairs: {int(len(target_names)*(len(target_names)-1)/2)}")
    lines.append("")
    lines.append("## Main table")
    lines.append(table[latex_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("Positive correlations mean that targets farther apart in pocket/sequence space are also farther apart in ligand-distribution descriptor space. Assignment metrics are scaffold-split shape-distribution assignment scores, not kNN/SEA-like baselines.")
    if not any(table["descriptor"].str.contains("BDC", case=False, na=False)):
        lines.append("")
        lines.append("BDC was not computed in this run. Add `--precomputed-descriptor BDC:e0_bdc_descriptors.csv` or run locally with a TSC-generated BDC descriptor matrix to fill the BDC pocket-correlation row.")
    (out / "analysis_report.md").write_text("\n".join(lines))
    print(f"Done. Results saved to: {out}")
    print(out / "e0_descriptor_recovery_table.csv")

if __name__ == "__main__":
    main()
