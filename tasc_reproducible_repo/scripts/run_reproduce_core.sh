#!/usr/bin/env bash
set -euo pipefail

# Minimal reproducibility driver. Edit paths if your data are not in repo root.
META=${META:-full_sdfs.csv}
MORGAN=${MORGAN:-processed_sdfs-2.csv}
TSC_CORE=${TSC_CORE:-../TSC-restore-3d-desc/core}
N_JOBS=${N_JOBS:-8}

mkdir -p results

python experiments/e0/make_usr_usrcat_descriptors.py \
  --metadata-csv "$META" \
  --out results/e0_shape_descriptors \
  --descriptors usr usrcat \
  --n-jobs "$N_JOBS" \
  --resume

PYTHONPATH="experiments/wrappers:${TSC_CORE}:." python experiments/e0/make_bdc_descriptors.py \
  --metadata-csv "$META" \
  --tsc-core-path "$TSC_CORE" \
  --wrapper-module tsc_bdc_wrapper \
  --out results/e0_bdc_descriptors.csv || echo "BDC generation failed; continue with precomputed BDC if available."

python experiments/e0/run_descriptor_recovery.py \
  --metadata-csv "$META" \
  --morgan-fp-csv "$MORGAN" \
  --descriptors morgan usr usrcat pca_sh \
  --precomputed-descriptor bdc:results/e0_bdc_descriptors.csv \
  --out results/e0_descriptor_recovery_results_final \
  --n-jobs "$N_JOBS" \
  --mantel-permutations 999

python experiments/e1/run_mantel_grid.py \
  --metadata-csv "$META" \
  --morgan-fp-csv "$MORGAN" \
  --descriptors morgan pca_sh \
  --precomputed-descriptor usr:results/e0_shape_descriptors/e0_usr_descriptors.csv \
  --precomputed-descriptor usrcat:results/e0_shape_descriptors/e0_usrcat_descriptors.csv \
  --precomputed-descriptor bdc:results/e0_bdc_descriptors.csv \
  --distance-metrics js wasserstein mmd \
  --out results/e1_mantel_grid_results_final \
  --formats png pdf

python experiments/e1/recompute_pairwise_js_table.py \
  --metadata-csv "$META" \
  --morgan-fp-csv "$MORGAN" \
  --out corrected_pairwise_js_table.csv \
  --n-clusters 32
python experiments/e1/make_corrected_js_alignment_table.py
mv corrected_pairwise_js_table.csv results/ || true
mv corrected_js_alignment_table_*.csv results/ || true
mv corrected_js_alignment_table_*.tex results/ || true

PYTHONPATH="experiments/wrappers:${TSC_CORE}:." python experiments/e3/run_target_assignment_v2.py \
  --metadata-csv "$META" \
  --morgan-fp-csv "$MORGAN" \
  --descriptors morgan usr usrcat bdc \
  --tsc-core-path "$TSC_CORE" \
  --tsc-function tsc_bdc_wrapper:compute_bdc_descriptor \
  --out results/e3_assignment_results_all_descriptors \
  --formats png pdf
