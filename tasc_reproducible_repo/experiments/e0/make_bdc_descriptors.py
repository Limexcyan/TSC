#!/usr/bin/env python3
"""Generate a precomputed BDC descriptor matrix for E0 from full_sdfs.csv.

This script is meant to be run locally in the same environment where your TSC BDC
wrapper already works. It writes a CSV that can be passed to run_e0_descriptor_recovery.py:

python make_e0_bdc_descriptors.py \
  --metadata-csv full_sdfs.csv \
  --tsc-core-path ../TSC-restore-3d-desc/core \
  --wrapper-module tsc_bdc_wrapper \
  --out e0_bdc_descriptors.csv

Then:

python run_e0_descriptor_recovery.py \
  --metadata-csv full_sdfs.csv \
  --morgan-fp-csv processed_sdfs-2.csv \
  --descriptors morgan usr usrcat pca_sh \
  --precomputed-descriptor BDC:e0_bdc_descriptors.csv \
  --out e0_descriptor_recovery_results \
  --n-jobs 8
"""
from __future__ import annotations
import argparse
import importlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def first_existing(cols, candidates):
    low={c.lower():c for c in cols}
    for c in candidates:
        if c in cols: return c
        if c.lower() in low: return low[c.lower()]
    return None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--metadata-csv', default='full_sdfs.csv')
    ap.add_argument('--ligand-col', default=None)
    ap.add_argument('--smiles-col', default=None)
    ap.add_argument('--tsc-core-path', default='../TSC-restore-3d-desc/core')
    ap.add_argument('--wrapper-module', default='tsc_bdc_wrapper')
    ap.add_argument('--wrapper-function', default='compute_bdc_descriptor')
    ap.add_argument('--out', default='e0_bdc_descriptors.csv')
    ap.add_argument('--errors-out', default='e0_bdc_descriptor_errors.csv')
    args=ap.parse_args()

    sys.path.insert(0, str(Path(args.tsc_core_path).resolve()))
    sys.path.insert(0, str(Path('.').resolve()))
    mod=importlib.import_module(args.wrapper_module)
    fn=getattr(mod, args.wrapper_function)

    df=pd.read_csv(args.metadata_csv)
    ligand_col=args.ligand_col or first_existing(df.columns, ['ligand_id','chembl_id','molecule_chembl_id','zinc_id','ID'])
    smiles_col=args.smiles_col or first_existing(df.columns, ['canonical_smiles','smiles','SMILES'])
    if ligand_col is None or smiles_col is None:
        raise SystemExit(f'Could not find ligand/smiles columns in {list(df.columns)}')
    ligs=(df[[ligand_col, smiles_col]].dropna().drop_duplicates(ligand_col).rename(columns={ligand_col:'ligand_id', smiles_col:'canonical_smiles'}).reset_index(drop=True))

    rows=[]; errors=[]; dim=0
    tmp=[]
    for i,r in ligs.iterrows():
        ligand_id=str(r['ligand_id']); smi=str(r['canonical_smiles'])
        try:
            v=np.asarray(fn(smi), dtype=float).ravel()
            if v.size==0 or not np.isfinite(v).all():
                raise ValueError('empty or non-finite BDC vector')
            tmp.append((ligand_id, v)); dim=max(dim, len(v))
        except Exception as e:
            tmp.append((ligand_id, np.array([], dtype=float)))
            errors.append({'ligand_id':ligand_id, 'error':repr(e)})
        if (i+1)%100==0:
            print(f'{i+1}/{len(ligs)} done; errors={len(errors)}; dim={dim}', flush=True)
    if dim==0:
        raise SystemExit('No valid BDC vectors were produced.')
    out_rows=[]
    for ligand_id,v in tmp:
        arr=np.zeros(dim, dtype=float)
        if len(v): arr[:min(dim,len(v))]=v[:dim]
        out_rows.append([ligand_id]+arr.tolist())
    cols=['ligand_id']+[f'desc_{i}' for i in range(dim)]
    pd.DataFrame(out_rows, columns=cols).to_csv(args.out, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(args.errors_out, index=False)
    print('saved', args.out)
    print('ligands', len(out_rows), 'dim', dim, 'errors', len(errors))

if __name__=='__main__': main()
