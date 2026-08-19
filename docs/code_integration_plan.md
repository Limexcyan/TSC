# Code integration plan

This package is organized as a clean overlay on top of the original TSC branch `restore-3d-desc`.

## What was preserved

The paper scripts remain command-line scripts under `experiments/` rather than being aggressively rewritten. This keeps exact reproducibility of the analysis runs.

## What was normalized

1. Filename suffixes such as `(1)`, `(2)`, `(3)` were removed.
2. Scripts were grouped by experiment E0--E5.
3. The BDC wrapper was placed both at repository root and under `experiments/wrappers/` for backwards-compatible imports.
4. Shared JS-divergence utilities were added under `src/tasc_paper/distances.py`.
5. Generated outputs and large input data are gitignored.

## Recommended future refactor

After the paper is submitted, move repeated helper functions from the experiment scripts into reusable modules:

```text
src/tasc_paper/io.py              column detection and metadata loading
src/tasc_paper/descriptors.py     Morgan, USR, USRCAT, PCA-SH, BDC wrapper calls
src/tasc_paper/distributions.py   cluster distributions and target-level histograms
src/tasc_paper/distances.py       JS, Wasserstein, MMD, Mantel helpers
src/tasc_paper/assignment.py      shape-distribution, kNN, SEA-like assignment
src/tasc_paper/selectivity.py     Ki conversion, boundary scores, selectivity metrics
src/tasc_paper/oneclass.py        centroid and Isolation Forest scoring
```

Do not do that deep refactor before final paper checks unless necessary, because it can silently change numerical results.

## Pull-request workflow

1. Clone original TSC branch.
2. Copy this overlay into the repository root.
3. Commit as `paper-reproducibility-scripts`.
4. Run `pytest` and one smoke command on synthetic test data.
5. Run the full paper commands only on the local machine with the large CSV/SDF inputs.
