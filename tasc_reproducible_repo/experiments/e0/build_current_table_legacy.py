#!/usr/bin/env python3
"""Build the current E0 descriptor-recovery table from available project outputs.

This is the lightweight companion to run_e0_descriptor_recovery.py. It uses:
- full_sdfs.csv + processed_sdfs-2.csv to compute the current Morgan pocket-recovery row;
- E3 assignment metrics CSVs to fill assignment top-1 and macro-F1 for Morgan and BDC;
- placeholders for descriptors whose full descriptor matrices are not present yet.

Use run_e0_descriptor_recovery.py to generate exact rows for USR/USRCAT/BDC/SH once
those descriptor matrices are available.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
try:
    from Bio import Align
except Exception:
    Align = None


def seq_distance_matrix(meta, target_col='target', seq_col='seq'):
    targets = sorted(meta[target_col].dropna().unique())
    seqs = meta.groupby(target_col)[seq_col].first().astype(str).str.replace('\n','', regex=False).to_dict()
    n=len(targets); D=np.zeros((n,n),float)
    if Align:
        aligner=Align.PairwiseAligner(); aligner.mode='global'; aligner.match_score=1.0; aligner.mismatch_score=0.0; aligner.open_gap_score=-1.0; aligner.extend_gap_score=-0.1
    for i,a in enumerate(targets):
        for j,b in enumerate(targets[i+1:], i+1):
            sa, sb = seqs[a], seqs[b]
            if Align:
                score = aligner.score(sa, sb)
                ident = max(0.0, min(1.0, score / max(len(sa), len(sb), 1)))
            else:
                m=min(len(sa),len(sb)); ident=sum(x==y for x,y in zip(sa[:m],sb[:m]))/max(len(sa),len(sb),1)
            D[i,j]=D[j,i]=1.0-ident
    return targets,D


def js(p,q,eps=1e-9):
    p=np.asarray(p,float)+eps; q=np.asarray(q,float)+eps
    p=p/p.sum(); q=q/q.sum(); m=.5*(p+q)
    return .5*np.sum(p*np.log(p/m))+.5*np.sum(q*np.log(q/m))


def mantel_rho(Dx, Dy, n_perm=999, seed=42):
    idx=np.triu_indices_from(Dx,1)
    x=Dx[idx]; y=Dy[idx]
    mask=np.isfinite(x)&np.isfinite(y)
    x=x[mask]; y=y[mask]
    rho=spearmanr(x,y).statistic
    rng=np.random.default_rng(seed); ge=0; n=Dx.shape[0]
    for _ in range(n_perm):
        p=rng.permutation(n)
        yy=Dy[p][:,p][idx][mask]
        rr=spearmanr(x, yy).statistic
        if abs(rr) >= abs(rho): ge += 1
    return float(rho), float((ge+1)/(n_perm+1))


def compute_morgan_recovery(meta_csv, morgan_csv, n_perm=999):
    meta=pd.read_csv(meta_csv)
    X=pd.read_csv(morgan_csv).apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(float)
    targets,Dp=seq_distance_matrix(meta)
    # mean bit-frequency distribution per target
    P=[]
    for t in targets:
        mask=(meta['target']==t).values
        v=X[mask].mean(axis=0)+1e-6
        v=v/v.sum()
        P.append(v)
    P=np.vstack(P)
    Dj=np.zeros_like(Dp)
    for i in range(len(targets)):
        for j in range(i+1,len(targets)):
            Dj[i,j]=Dj[j,i]=js(P[i],P[j])
    idx=np.triu_indices_from(Dj,1)
    x=Dj[idx]; y=Dp[idx]
    pr=pearsonr(x,y); sr=spearmanr(x,y); mr,mp=mantel_rho(Dj,Dp,n_perm=n_perm)
    return {
        'pocket_pearson_r': float(pr.statistic), 'pocket_pearson_p': float(pr.pvalue),
        'pocket_spearman_rho': float(sr.statistic), 'pocket_spearman_p': float(sr.pvalue),
        'mantel_rho': mr, 'mantel_p': mp,
        'n_targets': len(targets), 'n_pairs': int(len(targets)*(len(targets)-1)/2),
        'source_note': 'computed from full_sdfs.csv + processed_sdfs-2.csv using target-level Morgan bit-frequency JS divergence vs sequence distance'
    }


def read_assignment(e3_csv):
    if not e3_csv or not Path(e3_csv).exists():
        return {}
    df=pd.read_csv(e3_csv)
    out={}
    # Prefer shape_distribution rows to measure the paper's target-distribution method.
    for _,r in df.iterrows():
        desc=str(r.get('descriptor','')).lower()
        method=str(r.get('method','')).lower()
        if method == 'shape_distribution':
            out[desc]={'assignment_top1': float(r.get('top1_accuracy', np.nan)), 'assignment_macro_f1': float(r.get('macro_f1', np.nan))}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--metadata-csv', default='full_sdfs.csv')
    ap.add_argument('--morgan-fp-csv', default='processed_sdfs-2.csv')
    ap.add_argument('--e3-metrics-csv', default='e3_morgan_vs_bdc_metrics.csv')
    ap.add_argument('--out', default='e0_descriptor_recovery_current')
    ap.add_argument('--mantel-permutations', type=int, default=999)
    args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    assign=read_assignment(args.e3_metrics_csv)
    rows=[]
    m=compute_morgan_recovery(args.metadata_csv, args.morgan_fp_csv, args.mantel_permutations)
    rows.append({'descriptor':'Morgan','distance_metric':'JS', **m, **assign.get('morgan',{}), 'status':'computed'})
    for desc in ['USR','USRCAT','BDC','SH/PCA-SH']:
        row={'descriptor':desc,'distance_metric':'JS','pocket_pearson_r':np.nan,'pocket_pearson_p':np.nan,'pocket_spearman_rho':np.nan,'pocket_spearman_p':np.nan,'mantel_rho':np.nan,'mantel_p':np.nan,'assignment_top1':np.nan,'assignment_macro_f1':np.nan,'n_targets':np.nan,'n_pairs':np.nan}
        if desc=='BDC' and 'bdc' in assign:
            row.update(assign['bdc'])
            row['status']='assignment_done_pocket_missing'
            row['source_note']='BDC assignment imported from E3; pocket-recovery requires BDC descriptor matrix or local TSC run.'
        else:
            row['status']='missing_descriptor_or_assignment'
            row['source_note']='Run run_e0_descriptor_recovery.py with this descriptor or provide a precomputed descriptor CSV.'
        rows.append(row)
    table=pd.DataFrame(rows)
    table.to_csv(out/'e0_descriptor_recovery_table_current_with_status.csv', index=False)
    core_cols=['descriptor','distance_metric','pocket_pearson_r','pocket_spearman_rho','mantel_rho','assignment_top1','assignment_macro_f1']
    core=table[core_cols]
    core.to_csv(out/'e0_descriptor_recovery_table_current.csv', index=False)
    tex=core.copy()
    for c in core_cols[2:]: tex[c]=tex[c].map(lambda v:'--' if pd.isna(v) else f'{v:.3f}')
    (out/'e0_descriptor_recovery_table_current.tex').write_text(tex.to_latex(index=False, escape=False, column_format='llrrrrr'))
    lines=['# E0 current descriptor-recovery table', '', 'This table combines values that are already computable from the current project data with explicit placeholders for missing descriptor rows.', '', '## Current table', core.to_markdown(index=False), '', '## What is still missing', '- USR, USRCAT and SH/PCA-SH rows require running descriptor generation/assignment with `run_e0_descriptor_recovery.py`.', '- BDC has E3 assignment values available, but the pocket-recovery row requires a BDC descriptor matrix or local TSC BDC computation.', '- The Morgan pocket-recovery row is computed from opioid-family `full_sdfs.csv`/`processed_sdfs-2.csv` using JS divergence over target-level Morgan bit-frequency distributions and sequence-distance pocket proxy.']
    (out/'analysis_report.md').write_text('\n'.join(lines))
    print('Done:', out/'e0_descriptor_recovery_table_current.csv')

if __name__=='__main__': main()
