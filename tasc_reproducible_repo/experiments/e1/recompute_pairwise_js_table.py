import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import BisectingKMeans

def infer_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def get_fp_matrix(fp):
    id_like = {"ligand_id", "molecule_chembl_id", "canonical_smiles", "smiles", "target", "target_name"}
    cols = []
    for c in fp.columns:
        if c.lower() in id_like:
            continue
        if pd.api.types.is_numeric_dtype(fp[c]):
            cols.append(c)
    if not cols:
        raise SystemExit("No numeric fingerprint columns found.")
    return fp[cols].fillna(0).to_numpy(dtype=float)

def js_divergence_bits(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.sum() <= 0 or q.sum() <= 0:
        return np.nan

    p = p / p.sum()
    q = q / q.sum()

    # scipy returns sqrt(JS divergence), so square it.
    d = jensenshannon(p, q, base=2.0)
    return float(d ** 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata-csv", default="full_sdfs.csv")
    ap.add_argument("--morgan-fp-csv", default="processed_sdfs-2.csv")
    ap.add_argument("--out", default="corrected_pairwise_js_table.csv")
    ap.add_argument("--n-clusters", type=int, default=32)
    ap.add_argument("--target-filter", default=None, help="optional substring filter, e.g. opioid")
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata_csv)
    fp = pd.read_csv(args.morgan_fp_csv)

    target_col = infer_col(meta, ["target", "target_name", "target_id", "pref_name", "receptor"])
    ligand_col = infer_col(meta, ["ligand_id", "molecule_chembl_id", "chembl_id", "ID"])

    if target_col is None:
        raise SystemExit(f"Could not infer target column. Columns: {list(meta.columns)}")

    if ligand_col is None:
        ligand_col = "_row_ligand_id"
        meta[ligand_col] = np.arange(len(meta)).astype(str)

    if len(meta) != len(fp):
        raise SystemExit(f"Row mismatch: metadata={len(meta)}, fingerprints={len(fp)}")

    df = meta[[ligand_col, target_col]].copy()
    df[target_col] = df[target_col].astype(str)

    if args.target_filter:
        mask = df[target_col].str.contains(args.target_filter, case=False, na=False)
        df = df.loc[mask].copy()
        fp = fp.loc[mask].copy()

    X = get_fp_matrix(fp)

    km = BisectingKMeans(n_clusters=args.n_clusters, random_state=42)
    clusters = km.fit_predict(X)

    df["cluster"] = clusters

    targets = sorted(df[target_col].dropna().unique())
    if len(targets) < 2:
        raise SystemExit("Need at least two targets after filtering.")

    distributions = {}
    for t in targets:
        counts = (
            df.loc[df[target_col] == t, "cluster"]
            .value_counts()
            .reindex(range(args.n_clusters), fill_value=0)
            .to_numpy(dtype=float)
        )
        p = counts / counts.sum()
        distributions[t] = p

        if not np.isclose(p.sum(), 1.0):
            raise RuntimeError(f"Distribution for {t} does not sum to 1: {p.sum()}")

    rows = []
    for i, t1 in enumerate(targets):
        for t2 in targets[i+1:]:
            js = js_divergence_bits(distributions[t1], distributions[t2])
            rows.append({
                "target_1": t1,
                "target_2": t2,
                "js_divergence_bits": js,
                "js_distance_sqrt": np.sqrt(js),
            })

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)

    print("saved:", args.out)
    print("n_targets:", len(targets))
    print("n_pairs:", len(out))
    print("JS min:", out["js_divergence_bits"].min())
    print("JS max:", out["js_divergence_bits"].max())
    print(out.to_string(index=False))

    if out["js_divergence_bits"].max() > 1.000001:
        raise RuntimeError("JS divergence > 1: still wrong.")

if __name__ == "__main__":
    main()
