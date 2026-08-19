#!/usr/bin/env python3
"""
E1 Mantel grid: ligand-distribution distances vs pocket/sequence distances.

This script builds the missing E1 table:

    descriptor | distribution distance | Mantel rho | partial Mantel rho | p-value

It computes target-level ligand distributions from descriptor matrices and compares
pairwise target distances to pocket/sequence distances. Partial Mantel correlation
controls for a chosen control distance, by default Morgan/JS topology distance.

Designed for the current TaSC data:
    full_sdfs.csv        columns like chembl_id, ID, smiles, target, seq
    processed_sdfs-2.csv row-aligned Morgan fingerprints

Example:
    python run_e1_mantel_grid.py \
      --metadata-csv full_sdfs.csv \
      --morgan-fp-csv processed_sdfs-2.csv \
      --descriptors morgan usr usrcat pca_sh \
      --distance-metrics js wasserstein mmd \
      --out e1_mantel_grid_results \
      --mantel-permutations 999

Add BDC from a precomputed descriptor table:
    python run_e1_mantel_grid.py \
      --metadata-csv full_sdfs.csv \
      --morgan-fp-csv processed_sdfs-2.csv \
      --precomputed-descriptor BDC:e0_bdc_descriptors.csv \
      --descriptors morgan usr usrcat pca_sh \
      --distance-metrics js wasserstein mmd \
      --out e1_mantel_grid_results_all
"""
from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def first_existing(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    for cand in candidates:
        pat = re.compile(re.escape(cand), re.I)
        for c in cols:
            if pat.search(c):
                return c
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Run E1 descriptor x distance Mantel grid.")
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--morgan-fp-csv", default=None)
    p.add_argument("--precomputed-descriptor", action="append", default=[], help="name:path.csv, can be repeated")
    p.add_argument("--descriptors", nargs="+", default=["morgan"], help="morgan usr usrcat pca_sh plus precomputed descriptor names")
    p.add_argument("--distance-metrics", nargs="+", default=["js", "wasserstein", "mmd"], choices=["js", "bhattacharyya", "wasserstein", "mmd"])
    p.add_argument("--out", default="e1_mantel_grid_results")
    p.add_argument("--target-col", default=None)
    p.add_argument("--smiles-col", default=None)
    p.add_argument("--ligand-col", default=None)
    p.add_argument("--seq-col", default=None)
    p.add_argument("--pocket-distance-csv", default=None, help="Optional pair table with target_1,target_2,pocket distance")
    p.add_argument("--pocket-distance-col", default=None, help="Column in --pocket-distance-csv; auto-detects alignment_distance/pocket_distance")
    p.add_argument("--min-target-ligands", type=int, default=50)
    p.add_argument("--n-clusters", type=int, default=64)
    p.add_argument("--pca-components", type=int, default=10)
    p.add_argument("--max-samples-per-target", type=int, default=3000)
    p.add_argument("--mantel-permutations", type=int, default=999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args()


def load_metadata(path: Path, args) -> pd.DataFrame:
    df = pd.read_csv(path)
    ligand_col = args.ligand_col or first_existing(df.columns, ["ligand_id", "chembl_id", "ID", "zinc_id", "compound_id", "molecule_chembl_id"])
    smiles_col = args.smiles_col or first_existing(df.columns, ["canonical_smiles", "smiles", "SMILES", "isomeric_smiles"])
    target_col = args.target_col or first_existing(df.columns, ["target", "target_id", "target_name", "receptor", "protein"])
    seq_col = args.seq_col or first_existing(df.columns, ["seq", "sequence", "protein_sequence", "target_sequence"])
    if target_col is None:
        raise ValueError(f"Could not detect target column. Columns: {list(df.columns)}")
    if ligand_col is None:
        ligand_col = "__row_id"
        df[ligand_col] = [f"row_{i}" for i in range(len(df))]
    if smiles_col is None:
        df["__empty_smiles"] = ""
        smiles_col = "__empty_smiles"
    if seq_col is None and args.pocket_distance_csv is None:
        raise ValueError("Need a sequence column in metadata or --pocket-distance-csv.")
    out = pd.DataFrame({
        "ligand_id": df[ligand_col].astype(str),
        "canonical_smiles": df[smiles_col].astype(str),
        "target": df[target_col].astype(str),
    })
    if seq_col:
        out["seq"] = df[seq_col].astype(str)
    else:
        out["seq"] = ""
    return out.reset_index(drop=True)


def import_rdkit():
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolDescriptors
        from rdkit import DataStructs
        return Chem, AllChem, rdMolDescriptors, DataStructs
    except Exception as exc:
        raise ImportError("RDKit required for this descriptor. Install via conda-forge rdkit.") from exc


def mol_with_conformer(smi: str, seed: int = 1):
    Chem, AllChem, _, _ = import_rdkit()
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
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    return mol


def morgan_from_smiles(smiles, n_bits=1024, radius=2):
    Chem, AllChem, _, DataStructs = import_rdkit()
    X = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr
    return X


def rdkit_3d(smiles, kind: str, seed: int):
    _, _, rdMolDescriptors, _ = import_rdkit()
    rows = []
    bad = 0
    for i, smi in enumerate(smiles):
        mol = mol_with_conformer(str(smi), seed=seed)
        if mol is None:
            rows.append(None); bad += 1; continue
        try:
            if kind == "usr":
                v = rdMolDescriptors.GetUSR(mol)
            elif kind == "usrcat":
                v = rdMolDescriptors.GetUSRCAT(mol)
            else:
                raise ValueError(kind)
            rows.append(np.asarray(v, dtype=np.float32))
        except Exception:
            rows.append(None); bad += 1
    dim = max((len(x) for x in rows if x is not None), default=0)
    if dim == 0:
        raise ValueError(f"No {kind} descriptors could be computed.")
    X = np.zeros((len(rows), dim), dtype=np.float32)
    for i, x in enumerate(rows):
        if x is not None:
            X[i, :len(x)] = x
    if bad:
        warnings.warn(f"{kind}: {bad} molecules failed; encoded as zero vectors")
    return X


def pca_sh_descriptor(smiles, seed: int, max_l: int = 6):
    """Simple PCA-normalised radial spherical-harmonic-like descriptor.

    This is a lightweight stand-in for SH/PCA-SH: generate a conformer, align heavy-atom
    coordinates by PCA, then summarize radial distributions along PCA axes and radial
    moments. It is not a full surface spherical-harmonics implementation, but it gives a
    reproducible PCA-shape baseline row.
    """
    Chem, _, _, _ = import_rdkit()
    rows = []
    bad = 0
    for smi in smiles:
        mol = mol_with_conformer(str(smi), seed=seed)
        if mol is None:
            rows.append(None); bad += 1; continue
        try:
            conf = mol.GetConformer()
            coords = np.asarray(conf.GetPositions(), dtype=float)
            heavy = [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1]
            coords = coords[heavy] if heavy else coords
            coords = coords - coords.mean(axis=0, keepdims=True)
            if len(coords) >= 3:
                _, _, vt = np.linalg.svd(coords, full_matrices=False)
                coords = coords @ vt.T
            r = np.linalg.norm(coords, axis=1)
            feats = []
            # global radial moments
            for arr in [r, np.abs(coords[:,0]), np.abs(coords[:,1]), np.abs(coords[:,2])]:
                feats += [float(np.mean(arr)), float(np.std(arr)), float(np.min(arr)), float(np.max(arr))]
                feats += [float(np.percentile(arr, q)) for q in [10,25,50,75,90]]
            # signed axis moments
            for ax in range(3):
                x = coords[:, ax]
                feats += [float(np.mean(x)), float(np.std(x)), float(np.mean(x**2)), float(np.mean(x**3)), float(np.mean(x**4))]
            rows.append(np.asarray(feats, dtype=np.float32))
        except Exception:
            rows.append(None); bad += 1
    dim = max((len(x) for x in rows if x is not None), default=0)
    if dim == 0:
        raise ValueError("No pca_sh descriptors could be computed.")
    X = np.zeros((len(rows), dim), dtype=np.float32)
    for i, x in enumerate(rows):
        if x is not None:
            X[i, :len(x)] = x
    if bad:
        warnings.warn(f"pca_sh: {bad} molecules failed; encoded as zero vectors")
    return X


def load_descriptor_csv(path: Path, meta: pd.DataFrame, ligand_id_col="ligand_id"):
    df = pd.read_csv(path)
    id_col = first_existing(df.columns, ["ligand_id", "chembl_id", "ID", "zinc_id"])
    if id_col:
        num_cols = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
        if not num_cols:
            tmp = df.drop(columns=[id_col]).apply(pd.to_numeric, errors="coerce")
            num_cols = tmp.columns.tolist()
            df = pd.concat([df[[id_col]], tmp], axis=1)
        # average duplicate IDs, then map onto row-level meta
        agg = df[[id_col] + num_cols].groupby(id_col, sort=False).mean()
        X = meta[["ligand_id"]].merge(agg, left_on="ligand_id", right_index=True, how="left").drop(columns=["ligand_id"])
        return X.fillna(0.0).to_numpy(dtype=np.float32)
    else:
        num = df.apply(pd.to_numeric, errors="coerce")
        if len(num) != len(meta):
            raise ValueError(f"Descriptor {path} has {len(num)} rows, metadata has {len(meta)} rows, and no ligand_id column.")
        return num.fillna(0.0).to_numpy(dtype=np.float32)


def load_descriptors(args, meta: pd.DataFrame):
    requested = [x.lower() for x in args.descriptors]
    out = {}
    if "morgan" in requested:
        if args.morgan_fp_csv:
            X = pd.read_csv(args.morgan_fp_csv)
            X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            if len(X) != len(meta):
                raise ValueError(f"Morgan file rows {len(X)} != metadata rows {len(meta)}")
            out["Morgan"] = X
        else:
            out["Morgan"] = morgan_from_smiles(meta["canonical_smiles"].tolist())
    if "usr" in requested:
        out["USR"] = rdkit_3d(meta["canonical_smiles"].tolist(), "usr", args.seed)
    if "usrcat" in requested:
        out["USRCAT"] = rdkit_3d(meta["canonical_smiles"].tolist(), "usrcat", args.seed)
    if "pca_sh" in requested or "sh" in requested or "pca-sh" in requested:
        out["SH/PCA-SH"] = pca_sh_descriptor(meta["canonical_smiles"].tolist(), args.seed)
    for spec in args.precomputed_descriptor:
        if ":" not in spec:
            raise ValueError("--precomputed-descriptor must be name:path.csv")
        name, pth = spec.split(":", 1)
        out[name] = load_descriptor_csv(Path(pth), meta)
    return out


def target_filter(meta, min_target_ligands: int):
    counts = meta["target"].value_counts()
    keep = counts[counts >= min_target_ligands].index.tolist()
    if len(keep) < 3:
        raise ValueError(f"Need at least 3 targets with >= {min_target_ligands} ligands; got {len(keep)}")
    mask = meta["target"].isin(keep).to_numpy()
    return meta.loc[mask].reset_index(drop=True), mask, keep


def sequence_distance_matrix(meta: pd.DataFrame, targets: list[str]):
    seq_map = meta[["target", "seq"]].drop_duplicates().set_index("target")["seq"].to_dict()
    D = np.zeros((len(targets), len(targets)), dtype=float)
    try:
        from Bio.Align import PairwiseAligner
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 1.0
        aligner.mismatch_score = 0.0
        aligner.open_gap_score = 0.0
        aligner.extend_gap_score = 0.0
        for i, t1 in enumerate(targets):
            for j, t2 in enumerate(targets):
                if j <= i: continue
                s1, s2 = seq_map[t1], seq_map[t2]
                score = aligner.score(s1, s2)
                denom = max(len(s1), len(s2), 1)
                D[i, j] = D[j, i] = 1.0 - score / denom
    except Exception:
        warnings.warn("Biopython unavailable: using simple length-normalized mismatch distance")
        for i, t1 in enumerate(targets):
            for j, t2 in enumerate(targets):
                if j <= i: continue
                s1, s2 = str(seq_map[t1]), str(seq_map[t2])
                L = max(len(s1), len(s2), 1)
                s1p = s1.ljust(L, "-")
                s2p = s2.ljust(L, "-")
                ident = sum(a == b for a, b in zip(s1p, s2p)) / L
                D[i, j] = D[j, i] = 1.0 - ident
    return D


def load_pocket_distance_csv(path: Path, targets: list[str], distance_col: Optional[str]):
    df = pd.read_csv(path)
    c1 = first_existing(df.columns, ["target_1", "target1", "Sample_1", "sample_1"])
    c2 = first_existing(df.columns, ["target_2", "target2", "Sample_2", "sample_2"])
    dc = distance_col or first_existing(df.columns, ["pocket_distance", "alignment_distance", "distance", "Alignment_Distance"])
    if not (c1 and c2 and dc):
        raise ValueError(f"Need target pair cols and distance col in {path}; columns={list(df.columns)}")
    idx = {t: i for i, t in enumerate(targets)}
    D = np.full((len(targets), len(targets)), np.nan, dtype=float)
    np.fill_diagonal(D, 0.0)
    for _, row in df.iterrows():
        t1, t2 = str(row[c1]), str(row[c2])
        if t1 in idx and t2 in idx:
            D[idx[t1], idx[t2]] = D[idx[t2], idx[t1]] = float(row[dc])
    if np.isnan(D).any():
        raise ValueError("Pocket distance CSV does not cover all retained target pairs.")
    return D


def preprocess_for_distances(X: np.ndarray, pca_components: int):
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    # Remove all-zero variance columns cheaply
    std = X.std(axis=0)
    keep = std > 1e-12
    if keep.sum() >= 2:
        X = X[:, keep]
    # scale/PCA for continuous metrics
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    n_comp = int(min(pca_components, Xs.shape[1], Xs.shape[0] - 1))
    if n_comp >= 2:
        Xp = PCA(n_components=n_comp, random_state=42).fit_transform(Xs)
    else:
        Xp = Xs
    return np.asarray(Xp, dtype=np.float32)


def sample_indices_by_target(meta, targets, max_samples, seed):
    rng = np.random.default_rng(seed)
    groups = []
    for t in targets:
        idx = np.flatnonzero((meta["target"].to_numpy() == t))
        if len(idx) > max_samples:
            idx = rng.choice(idx, size=max_samples, replace=False)
        groups.append(np.asarray(idx, dtype=int))
    return groups


def js_or_bhatt_matrix(X, meta, targets, n_clusters, seed, metric="js"):
    k = int(min(max(2, n_clusters), max(2, len(X)-1)))
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=min(4096, max(256, len(X)//10)))
    lab = km.fit_predict(X)
    P = []
    for t in targets:
        idx = np.flatnonzero(meta["target"].to_numpy() == t)
        hist = np.bincount(lab[idx], minlength=k).astype(float) + 1e-6
        hist /= hist.sum()
        P.append(hist)
    P = np.vstack(P)
    D = np.zeros((len(targets), len(targets)), dtype=float)
    for i in range(len(targets)):
        for j in range(i+1, len(targets)):
            if metric == "js":
                d = float(jensenshannon(P[i], P[j], base=2.0) ** 2)
            else:
                bc = float(np.sum(np.sqrt(P[i] * P[j])))
                d = -math.log(max(bc, 1e-12))
            D[i, j] = D[j, i] = d
    return D


def wasserstein_matrix(Xp, groups, targets):
    D = np.zeros((len(targets), len(targets)), dtype=float)
    for i in range(len(targets)):
        Xi = Xp[groups[i]]
        for j in range(i+1, len(targets)):
            Xj = Xp[groups[j]]
            vals = [wasserstein_distance(Xi[:, c], Xj[:, c]) for c in range(Xp.shape[1])]
            D[i, j] = D[j, i] = float(np.mean(vals))
    return D


def mmd_matrix(Xp, groups, targets, seed):
    # Median bandwidth from a small sample
    rng = np.random.default_rng(seed)
    sample_idx = np.concatenate([g[:min(len(g), 200)] for g in groups])
    if len(sample_idx) > 1000:
        sample_idx = rng.choice(sample_idx, size=1000, replace=False)
    S = Xp[sample_idx]
    # squared distances sample
    dif = S[:min(500, len(S)), None, :] - S[None, :min(500, len(S)), :]
    d2 = np.sum(dif*dif, axis=2)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / max(2.0 * med, 1e-9)
    def kmean(A, B):
        # block simple, sampled groups small
        d2 = np.sum((A[:, None, :] - B[None, :, :])**2, axis=2)
        return float(np.exp(-gamma * d2).mean())
    D = np.zeros((len(targets), len(targets)), dtype=float)
    for i in range(len(targets)):
        Xi = Xp[groups[i]]
        for j in range(i+1, len(targets)):
            Xj = Xp[groups[j]]
            mmd2 = kmean(Xi, Xi) + kmean(Xj, Xj) - 2 * kmean(Xi, Xj)
            D[i, j] = D[j, i] = float(math.sqrt(max(mmd2, 0.0)))
    return D


def upper_vec(M):
    M = np.asarray(M, dtype=float)
    iu = np.triu_indices_from(M, k=1)
    return M[iu]


def mantel_spearman(XD, YD, permutations=999, seed=42):
    x = upper_vec(XD)
    y = upper_vec(YD)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    rho = float(spearmanr(x, y).statistic) if len(x) >= 3 else np.nan
    if not np.isfinite(rho) or permutations <= 0:
        return rho, np.nan
    rng = np.random.default_rng(seed)
    n = XD.shape[0]
    more = 0
    valid = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        xp = upper_vec(XD[np.ix_(perm, perm)])[mask]
        rp = spearmanr(xp, y).statistic
        if np.isfinite(rp):
            valid += 1
            if abs(rp) >= abs(rho):
                more += 1
    p = (more + 1.0) / (valid + 1.0) if valid else np.nan
    return rho, float(p)


def partial_corr(x, y, z):
    # rank-based partial correlation through residuals
    from scipy.stats import rankdata
    x = rankdata(x); y = rankdata(y); z = rankdata(z)
    Z = np.column_stack([np.ones_like(z), z])
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]
    by = np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ bx
    ry = y - Z @ by
    return float(pearsonr(rx, ry).statistic)


def partial_mantel(XD, YD, ZD, permutations=999, seed=42):
    x = upper_vec(XD); y = upper_vec(YD); z = upper_vec(ZD)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 4 or np.std(z) < 1e-12:
        return np.nan, np.nan
    rho = partial_corr(x, y, z)
    if not np.isfinite(rho) or permutations <= 0:
        return rho, np.nan
    rng = np.random.default_rng(seed)
    n = XD.shape[0]
    more = 0; valid = 0
    for _ in range(permutations):
        perm = rng.permutation(n)
        xp = upper_vec(XD[np.ix_(perm, perm)])[mask]
        rp = partial_corr(xp, y, z)
        if np.isfinite(rp):
            valid += 1
            if abs(rp) >= abs(rho):
                more += 1
    p = (more + 1.0) / (valid + 1.0) if valid else np.nan
    return float(rho), float(p)


def pearson_spearman_vec(XD, YD):
    x, y = upper_vec(XD), upper_vec(YD)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    pr = pearsonr(x, y).statistic if len(x) >= 3 else np.nan
    sr = spearmanr(x, y).statistic if len(x) >= 3 else np.nan
    pp = pearsonr(x, y).pvalue if len(x) >= 3 else np.nan
    sp = spearmanr(x, y).pvalue if len(x) >= 3 else np.nan
    return float(pr), float(pp), float(sr), float(sp), len(x)


def save_heatmap(M, targets, title, path):
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(max(5, 0.55*len(targets)+2), max(4, 0.5*len(targets)+1.5)))
    im = ax.imshow(M, aspect="auto")
    ax.set_xticks(range(len(targets))); ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_yticks(range(len(targets))); ax.set_yticklabels(targets)
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_scatter(XD, YD, title, outbase, formats):
    if plt is None:
        return
    x, y = upper_vec(XD), upper_vec(YD)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return
    pr, pp, sr, sp, _ = pearson_spearman_vec(XD, YD)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(x, y, alpha=0.8)
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, m*xx + b)
    ax.set_xlabel("Ligand-distribution distance")
    ax.set_ylabel("Pocket / sequence distance")
    ax.set_title(title + f"\nPearson r={pr:.3f}, Spearman ρ={sr:.3f}")
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(f"{outbase}.{fmt}", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    meta0 = load_metadata(Path(args.metadata_csv), args)
    meta, mask, targets = target_filter(meta0, args.min_target_ligands)
    targets = sorted(targets)
    # important: sort target list and keep meta order. Descriptor arrays are row-aligned then masked.
    descriptors0 = load_descriptors(args, meta0)
    descriptors = {name: X[mask] for name, X in descriptors0.items()}

    if args.pocket_distance_csv:
        Dpocket = load_pocket_distance_csv(Path(args.pocket_distance_csv), targets, args.pocket_distance_col)
        pocket_source = str(args.pocket_distance_csv)
    else:
        Dpocket = sequence_distance_matrix(meta, targets)
        pocket_source = "metadata sequence column"

    np.savetxt(out_dir / "pocket_distance_matrix.csv", Dpocket, delimiter=",")
    pd.DataFrame(Dpocket, index=targets, columns=targets).to_csv(out_dir / "pocket_distance_matrix_labeled.csv")

    all_distance_mats = {}
    # precompute Morgan JS as control if available
    control = None
    if "Morgan" in descriptors:
        control = js_or_bhatt_matrix(descriptors["Morgan"], meta, targets, args.n_clusters, args.seed, metric="js")
        pd.DataFrame(control, index=targets, columns=targets).to_csv(out_dir / "control_morgan_js_distance_matrix.csv")

    rows = []
    for desc_name, Xraw in descriptors.items():
        print(f"Descriptor: {desc_name} shape={Xraw.shape}")
        Xp = preprocess_for_distances(Xraw, args.pca_components)
        groups = sample_indices_by_target(meta, targets, args.max_samples_per_target, args.seed)
        for metric in args.distance_metrics:
            print(f"  metric: {metric}")
            if desc_name == "Morgan" and metric == "js" and control is not None:
                XD = control
            elif metric in ("js", "bhattacharyya"):
                XD = js_or_bhatt_matrix(Xraw, meta, targets, args.n_clusters, args.seed, metric=metric)
            elif metric == "wasserstein":
                XD = wasserstein_matrix(Xp, groups, targets)
            elif metric == "mmd":
                XD = mmd_matrix(Xp, groups, targets, args.seed)
            else:
                raise ValueError(metric)
            key = f"{desc_name}__{metric}"
            all_distance_mats[key] = XD
            pd.DataFrame(XD, index=targets, columns=targets).to_csv(out_dir / f"distance_matrix_{desc_name.replace('/', '_')}_{metric}.csv")

            pear, pear_p, spear, spear_p, n_pairs = pearson_spearman_vec(XD, Dpocket)
            mantel_rho, mantel_p = mantel_spearman(XD, Dpocket, args.mantel_permutations, args.seed)
            if control is not None and not (desc_name == "Morgan" and metric == "js"):
                partial_rho, partial_p = partial_mantel(XD, Dpocket, control, args.mantel_permutations, args.seed)
                partial_control = "Morgan JS"
            else:
                partial_rho, partial_p = np.nan, np.nan
                partial_control = ""
            rows.append({
                "descriptor": desc_name,
                "distribution_distance": metric,
                "n_targets": len(targets),
                "n_pairs": n_pairs,
                "pocket_pearson_r": pear,
                "pocket_pearson_p": pear_p,
                "pocket_spearman_rho": spear,
                "pocket_spearman_p": spear_p,
                "mantel_rho": mantel_rho,
                "mantel_p_value": mantel_p,
                "partial_mantel_rho": partial_rho,
                "partial_mantel_p_value": partial_p,
                "partial_control": partial_control,
            })
            if not args.skip_plots:
                save_scatter(XD, Dpocket, f"E1: {desc_name} / {metric}", out_dir / f"fig_e1_{desc_name.replace('/', '_')}_{metric}_vs_pocket", args.formats)

    table = pd.DataFrame(rows)
    # Add manuscript-focused column aliases too
    manuscript = table[["descriptor", "distribution_distance", "mantel_rho", "partial_mantel_rho", "mantel_p_value", "n_targets", "n_pairs"]].copy()
    manuscript = manuscript.rename(columns={
        "distribution_distance": "distribution distance",
        "mantel_rho": "Mantel rho",
        "partial_mantel_rho": "partial Mantel rho",
        "mantel_p_value": "p-value",
    })
    table.to_csv(out_dir / "e1_mantel_grid_full.csv", index=False)
    manuscript.to_csv(out_dir / "e1_mantel_grid_table.csv", index=False)
    try:
        manuscript.to_latex(out_dir / "e1_mantel_grid_table.tex", index=False, float_format="%.3f", na_rep="--")
    except Exception:
        pass

    # Save a compact best-per-descriptor table by absolute Mantel rho.
    compact = table.sort_values("mantel_rho", key=lambda s: s.abs(), ascending=False).groupby("descriptor", as_index=False).head(1)
    compact.to_csv(out_dir / "e1_best_distance_per_descriptor.csv", index=False)

    report = []
    report.append("# E1 Mantel grid report")
    report.append("")
    report.append(f"Pocket distance source: {pocket_source}")
    report.append(f"Retained targets: {', '.join(targets)}")
    report.append(f"Minimum ligands per target: {args.min_target_ligands}")
    report.append(f"Mantel permutations: {args.mantel_permutations}")
    report.append("")
    report.append("## Best row per descriptor")
    for _, r in compact.iterrows():
        report.append(f"- {r['descriptor']} / {r['distribution_distance']}: Mantel rho={r['mantel_rho']:.3f}, p={r['mantel_p_value']:.3g}, partial={r['partial_mantel_rho'] if pd.notna(r['partial_mantel_rho']) else 'NA'}")
    report.append("")
    report.append("## Interpretation")
    report.append("Positive Mantel rho means descriptor-derived ligand distribution distances increase with pocket/sequence distances. Partial Mantel rho controls for Morgan/JS topology distance when available; use it to ask whether a 3D descriptor adds signal beyond 2D topology.")
    (out_dir / "analysis_report.md").write_text("\n".join(report))

    meta_summary = meta.groupby("target").size().rename("n_ligand_rows").reset_index()
    meta_summary.to_csv(out_dir / "target_coverage.csv", index=False)
    print(f"Done. E1 Mantel grid saved to: {out_dir}")
    print(f"Main table: {out_dir / 'e1_mantel_grid_table.csv'}")
    print(f"Full table: {out_dir / 'e1_mantel_grid_full.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
