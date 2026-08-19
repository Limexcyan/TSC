import re
import numpy as np
import pandas as pd
from pathlib import Path
from Bio import pairwise2

META = Path("full_sdfs.csv")
MAPPING = Path("target_mapping_opioids.csv")
JS_TABLE = Path("corrected_pairwise_js_table.csv")

def clean_seq(x):
    return re.sub(r"\s+", "", str(x)).upper()

def pick_sequence(meta, target):
    rows = meta.loc[meta["target"].astype(str) == str(target), "seq"].dropna().map(clean_seq)
    rows = rows[rows.str.len() > 0]
    if rows.empty:
        raise ValueError(f"No sequence found for target={target}")

    counts = rows.value_counts()
    if len(counts) > 1:
        print(f"WARNING: multiple sequences for {target}; using most frequent one.")
        print(counts.head())

    return counts.index[0]

def globalxx_alignment(seq1, seq2):
    aln = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    raw_score = float(aln.score)  # number of exact matches under globalxx
    aligned_len = len(aln.seqA)
    similarity = raw_score / aligned_len if aligned_len else np.nan
    distance = 1.0 - similarity if np.isfinite(similarity) else np.nan
    return raw_score, similarity, distance, aligned_len

meta = pd.read_csv(META)
mapping = pd.read_csv(MAPPING)
js = pd.read_csv(JS_TABLE)

# Add NOP fallback if not already in mapping.
if "chembl2014" not in set(mapping["target"].astype(str)):
    mapping = pd.concat([
        mapping,
        pd.DataFrame([{
            "target": "chembl2014",
            "target_chembl_id": "CHEMBL2014",
            "target_name": "NOP receptor / Nociceptin receptor"
        }])
    ], ignore_index=True)

raw_to_name = dict(zip(mapping["target"].astype(str), mapping["target_name"].astype(str)))

# Remove unmapped low-count target automatically.
js["target_1_name"] = js["target_1"].astype(str).map(raw_to_name)
js["target_2_name"] = js["target_2"].astype(str).map(raw_to_name)
clean = js.dropna(subset=["target_1_name", "target_2_name"]).copy()

seqs = {}
for t in sorted(set(clean["target_1"]).union(set(clean["target_2"]))):
    seqs[str(t)] = pick_sequence(meta, str(t))

rows = []
for _, r in clean.iterrows():
    t1 = str(r["target_1"])
    t2 = str(r["target_2"])

    score, sim, dist, aln_len = globalxx_alignment(seqs[t1], seqs[t2])

    rows.append({
        "target_1": r["target_1_name"],
        "target_2": r["target_2_name"],
        "js_divergence_bits": r["js_divergence_bits"],
        "js_distance_sqrt": r["js_distance_sqrt"],
        "alignment_score_globalxx": score,
        "alignment_similarity": sim,
        "alignment_distance": dist,
        "aligned_length": aln_len,
    })

out = pd.DataFrame(rows).sort_values("js_divergence_bits")

# Main opioid table: delta, kappa, mu, NOP only.
main_targets = {
    "Delta opioid receptor",
    "Kappa opioid receptor",
    "Mu opioid receptor",
    "NOP receptor / Nociceptin receptor",
}
main = out[
    out["target_1"].isin(main_targets) &
    out["target_2"].isin(main_targets)
].copy()

out.to_csv("corrected_js_alignment_table_all_mapped.csv", index=False)
main.to_csv("corrected_js_alignment_table_main_opioid.csv", index=False)

print("\n=== MAIN OPIOID TABLE ===")
print(main.to_string(index=False))

print("\n=== ALL MAPPED, INCLUDING SIGMA ===")
print(out.to_string(index=False))

# LaTeX versions
paper = main[[
    "target_1",
    "target_2",
    "js_divergence_bits",
    "alignment_score_globalxx",
    "alignment_similarity",
]].copy()

paper["js_divergence_bits"] = paper["js_divergence_bits"].map(lambda x: f"{x:.3f}")
paper["alignment_score_globalxx"] = paper["alignment_score_globalxx"].map(lambda x: f"{x:.0f}")
paper["alignment_similarity"] = paper["alignment_similarity"].map(lambda x: f"{x:.3f}")

tex = paper.to_latex(
    index=False,
    escape=False,
    column_format="llrrr",
    caption=(
        "Corrected ligand-distribution Jensen--Shannon divergence and protein sequence "
        "alignment similarity for the opioid receptor set."
    ),
    label="tab:corrected-js-alignment-opioid",
)

Path("corrected_js_alignment_table_main_opioid.tex").write_text(tex)

print("\n=== LaTeX written to corrected_js_alignment_table_main_opioid.tex ===")
print(tex)
