# E3 target-assignment benchmark

This package contains `run_e3_target_assignment.py`, a complete benchmark for the E3 experiment:

> Pocket-free, multi-label target assignment from ligand descriptors under scaffold and time splits.

It compares:

1. `shape_distribution`: target-level ligand-cluster distributions with Jensen-Shannon distance;
2. `knn`: k-nearest ligand-label transfer;
3. `sea_like`: a SEA-inspired target ligand-set baseline, without SwissTargetPrediction.

## Run with previous ZINC/prepared data

From the directory containing `full_sdfs.csv` and `processed_sdfs-2.csv`:

```bash
python run_e3_target_assignment.py \
  --metadata-csv full_sdfs.csv \
  --morgan-fp-csv processed_sdfs-2.csv \
  --descriptors morgan \
  --out e3_assignment_results \
  --formats png pdf
```

This uses the previous Morgan fingerprints directly, so it does not recompute Morgan descriptors.

## Run with TSC 3D descriptors

Clone the TSC repository locally and pass the local `core` path. Because function names can change, the robust mode is to provide the exact descriptor function:

```bash
python run_e3_target_assignment.py \
  --metadata-csv full_sdfs.csv \
  --descriptors bdc \
  --tsc-core-path /path/to/TSC/core \
  --tsc-function descriptors.bdc:compute_bdc_descriptor \
  --out e3_assignment_results_bdc \
  --formats png pdf
```

If the exact function name is different, change `--tsc-function module:function`.

## Run with RDKit 3D descriptors

```bash
python run_e3_target_assignment.py \
  --metadata-csv full_sdfs.csv \
  --descriptors morgan usr usrcat \
  --out e3_assignment_results_rdkit3d \
  --formats png pdf
```

USR/USRCAT require RDKit and generate conformers from SMILES, which may take time for large datasets.

## Main outputs

- `e3_assignment_metrics.csv` — main paper table.
- `e3_test_predictions.csv` — per-ligand predicted probabilities and top targets.
- `e3_calibration_bins.csv` — reliability diagram data.
- `split_summary.csv` — scaffold/time split sizes.
- `descriptor_summary.csv` — descriptor dimensions and coverage.
- `fig_e3_micro_f1_by_method.*` — performance figure.
- `fig_e3_top3_accuracy_by_method.*` — top-k assignment figure.
- `fig_e3_reliability_diagram.*` — calibration figure.
- `latex_e3_snippet.tex` — manuscript snippet.

## Required columns

The script auto-detects common names, but the safest metadata CSV has:

```text
ligand_id,canonical_smiles,target,document_year,pchembl_value
```

`document_year` is optional. Without it, the time split is skipped and scaffold split is still run.
