# TaSC paper reproducibility package

This repository overlay collects the original TaSC 3D-descriptor code path with the paper-specific scripts used for experiments E0--E5.

The original descriptor source used in the project was:

```bash
git clone --branch restore-3d-desc https://github.com/Limexcyan/TSC.git TSC-restore-3d-desc
```

The scripts in `experiments/` are intentionally kept close to the versions used during the paper work, so that old command lines remain reproducible. Shared guardrails and small utilities live in `src/tasc_paper/`.

## Layout

```text
src/tasc_paper/                 small shared utilities, especially corrected JS divergence
experiments/e0/                 descriptor-recovery tables and descriptor generation
experiments/e1/                 Mantel grid and corrected JS/alignment table
experiments/e2/                 docking/alignment and smina--Glide concordance utilities
experiments/e3/                 pocket-free target assignment benchmark
experiments/e4/                 boundary/selectivity Ki validation
experiments/e5/                 one-class binder/outlier benchmarks
experiments/wrappers/           wrapper around the original TSC BDC descriptor
configs/                        place for run configs, currently command-driven
data/raw/                       input CSVs/SDFs go here; large data are gitignored
results/                        generated outputs; gitignored except placeholder
manuscript/                     optional manuscript files, e.g. main_cleaned.tex
```

## Install

Preferred conda setup:

```bash
conda env create -f environment.yml
conda activate tasc-paper
pip install -e .
```

For BDC descriptors, clone the original TSC branch next to this repository:

```bash
git clone --branch restore-3d-desc https://github.com/Limexcyan/TSC.git ../TSC-restore-3d-desc
```

Then use the wrapper with:

```bash
PYTHONPATH="experiments/wrappers:../TSC-restore-3d-desc/core:." python ... \
  --tsc-core-path ../TSC-restore-3d-desc/core \
  --tsc-function tsc_bdc_wrapper:compute_bdc_descriptor
```

## Required input files

The main opioid/assignment runs expect these project files, which are not committed because they are generated/large:

```text
full_sdfs.csv
processed_sdfs-2.csv
```

Put them either in the repository root or in `data/raw/` and pass explicit paths in the commands below.

## Quick smoke test

```bash
pip install -e .[test]
pytest
```

## Reproduce paper experiments

A concise command list is in `docs/reproduce_paper.md`.

## Important methodological note

Use `src/tasc_paper/distances.py` for future JS calculations. `scipy.spatial.distance.jensenshannon` returns the Jensen--Shannon **distance**, i.e. `sqrt(JSD)`. The paper tables should report base-2 Jensen--Shannon **divergence**, computed as `jensenshannon(p, q, base=2.0) ** 2`, which is bounded by `[0, 1]`.
