# Reproducing the TaSC paper experiments

All commands below assume the working directory is the repository root and that `full_sdfs.csv` and `processed_sdfs-2.csv` are available either in the root or under `data/raw/`.

## E0: descriptor recovery

Generate USR/USRCAT descriptors if needed:

```bash
python experiments/e0/make_usr_usrcat_descriptors.py \
  --metadata-csv full_sdfs.csv \
  --out results/e0_shape_descriptors \
  --descriptors usr usrcat \
  --n-jobs 8 \
  --resume
```

Generate BDC descriptors through the original TSC code:

```bash
PYTHONPATH="experiments/wrappers:../TSC-restore-3d-desc/core:." \
python experiments/e0/make_bdc_descriptors.py \
  --metadata-csv full_sdfs.csv \
  --tsc-core-path ../TSC-restore-3d-desc/core \
  --wrapper-module tsc_bdc_wrapper \
  --out results/e0_bdc_descriptors.csv
```

Run E0:

```bash
python experiments/e0/run_descriptor_recovery.py \
  --metadata-csv full_sdfs.csv \
  --morgan-fp-csv processed_sdfs-2.csv \
  --descriptors morgan usr usrcat pca_sh \
  --precomputed-descriptor bdc:results/e0_bdc_descriptors.csv \
  --out results/e0_descriptor_recovery_results_final \
  --n-jobs 8 \
  --mantel-permutations 999
```

## E1: Mantel grid

```bash
python experiments/e1/run_mantel_grid.py \
  --metadata-csv full_sdfs.csv \
  --morgan-fp-csv processed_sdfs-2.csv \
  --descriptors morgan pca_sh \
  --precomputed-descriptor usr:results/e0_shape_descriptors/e0_usr_descriptors.csv \
  --precomputed-descriptor usrcat:results/e0_shape_descriptors/e0_usrcat_descriptors.csv \
  --precomputed-descriptor bdc:results/e0_bdc_descriptors.csv \
  --distance-metrics js wasserstein mmd \
  --out results/e1_mantel_grid_results_final \
  --formats png pdf
```

Correct the illustrative JS/alignment table:

```bash
python experiments/e1/recompute_pairwise_js_table.py \
  --metadata-csv full_sdfs.csv \
  --morgan-fp-csv processed_sdfs-2.csv \
  --out results/corrected_pairwise_js_table.csv \
  --n-clusters 32

cp results/corrected_pairwise_js_table.csv corrected_pairwise_js_table.csv
python experiments/e1/make_corrected_js_alignment_table.py
mv corrected_js_alignment_table_*.csv results/ 2>/dev/null || true
mv corrected_js_alignment_table_*.tex results/ 2>/dev/null || true
```

## E2: docking/alignment diagnostics

The E2 scripts are in `experiments/e2/`. They expect the prepared docking/alignment inputs used in the paper and write figures/tables to output folders passed in their CLIs.

## E3: pocket-free target assignment

```bash
PYTHONPATH="experiments/wrappers:../TSC-restore-3d-desc/core:." \
python experiments/e3/run_target_assignment_v2.py \
  --metadata-csv full_sdfs.csv \
  --morgan-fp-csv processed_sdfs-2.csv \
  --descriptors morgan usr usrcat bdc \
  --tsc-core-path ../TSC-restore-3d-desc/core \
  --tsc-function tsc_bdc_wrapper:compute_bdc_descriptor \
  --out results/e3_assignment_results_all_descriptors \
  --formats png pdf
```

Use precomputed descriptors when available to avoid recomputing 3D descriptors.

## E4: boundary/selectivity validation

First build or provide the ChEMBL Ki table:

```bash
python experiments/e4/build_chembl_ki_table.py \
  --metadata-csv full_sdfs.csv \
  --target-map data/raw/target_mapping_opioids.csv \
  --out results/chembl_opioid_ki
```

Then run E4 using an E3 predictions file with `prob__TARGET` columns:

```bash
python experiments/e4/run_boundary_selectivity_v2.py \
  --predictions-csv results/e3_assignment_results_all_descriptors/e3_test_predictions.csv \
  --ki-csv results/chembl_opioid_ki/chembl_ki_activities.csv \
  --out results/e4_boundary_selectivity_results \
  --formats png pdf
```

## E5: one-class benchmark

Build the cross-target decoy dataset and run the one-class benchmark with available descriptors:

```bash
python experiments/e5/build_e5_data_from_scratch.py --help
python experiments/e5/make_e5_shape_descriptors.py --help
python experiments/e5/run_e5_oneclass_benchmark.py --help
```

Confirmed-Ki dataset builder:

```bash
python experiments/e5/build_confirmed_ki_dataset.py \
  --ki-csv results/chembl_opioid_ki/chembl_ki_activities.csv \
  --out results/e5_confirmed_ki
```

Summarize the shape-claim table:

```bash
python experiments/e5/summarize_e5_shape_claim.py --help
```
