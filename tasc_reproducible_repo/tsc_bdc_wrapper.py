"""
Wrapper for using TSC/core desc3d.bend_deformed_cylinder_descriptors
from run_e3_target_assignment_v2.py.

Place this file next to run_e3_target_assignment_v2.py and call:

python run_e3_target_assignment_v2.py \
  --metadata-csv full_sdfs.csv \
  --descriptors bdc \
  --tsc-core-path ../TSC-restore-3d-desc/core/ \
  --tsc-function tsc_bdc_wrapper:compute_bdc_descriptor \
  --out e3_assignment_results_bdc \
  --formats png pdf

Requires RDKit, because BDC needs 3D coordinates, not just a SMILES string.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numbers
import numpy as np


def _flatten_numeric(value):
    """Convert nested numeric outputs to a 1D float numpy array."""
    if value is None:
        return np.array([], dtype=float)
    if isinstance(value, np.ndarray):
        try:
            arr = value.astype(float).ravel()
            return arr[np.isfinite(arr)]
        except Exception:
            pass
    if isinstance(value, numbers.Number):
        return np.array([float(value)], dtype=float)
    if isinstance(value, Mapping):
        parts = [_flatten_numeric(v) for v in value.values()]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    if isinstance(value, (str, bytes)):
        return np.array([], dtype=float)
    if isinstance(value, Sequence):
        parts = [_flatten_numeric(v) for v in value]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.array([], dtype=float)
    try:
        arr = np.asarray(value, dtype=float).ravel()
        return arr[np.isfinite(arr)]
    except Exception:
        return np.array([], dtype=float)


def _mol_from_smiles(smiles: str, seed: int = 1):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useRandomCoords = True
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(mol, randomSeed=int(seed), useRandomCoords=True)
    if status != 0:
        raise ValueError(f"RDKit cannot generate conformer for SMILES: {smiles!r}")
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=250)
    except Exception:
        pass
    return mol


def _mol_arrays(mol):
    conf = mol.GetConformer()
    coords = np.asarray(conf.GetPositions(), dtype=float)
    atomic_numbers = np.asarray([a.GetAtomicNum() for a in mol.GetAtoms()], dtype=int)
    atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]

    heavy_idx = [i for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1]
    heavy_coords = coords[heavy_idx]
    heavy_atomic_numbers = atomic_numbers[heavy_idx]
    heavy_atom_symbols = [atom_symbols[i] for i in heavy_idx]

    return coords, atomic_numbers, atom_symbols, heavy_coords, heavy_atomic_numbers, heavy_atom_symbols


def compute_bdc_descriptor(smiles: str, seed: int = 1):
    """Return a numeric BDC descriptor for one SMILES using TSC desc3d.

    The underlying TSC function is tried with several common calling conventions:
    RDKit Mol, coordinate array, coordinate array + atom labels, and keyword forms.
    """
    import desc3d

    fn = desc3d.bend_deformed_cylinder_descriptors
    mol = _mol_from_smiles(smiles, seed=seed)
    coords, nums, syms, hcoords, hnums, hsyms = _mol_arrays(mol)

    attempts = [
        lambda: fn(mol),
        lambda: fn(coords),
        lambda: fn(hcoords),
        lambda: fn(coords, nums),
        lambda: fn(hcoords, hnums),
        lambda: fn(coords, syms),
        lambda: fn(hcoords, hsyms),
        lambda: fn(mol, coords),
        lambda: fn(coordinates=coords),
        lambda: fn(coords=coords),
        lambda: fn(points=coords),
        lambda: fn(atom_positions=coords),
        lambda: fn(mol=mol),
        lambda: fn(coordinates=coords, atomic_numbers=nums),
        lambda: fn(coords=coords, atomic_numbers=nums),
        lambda: fn(coordinates=hcoords, atomic_numbers=hnums),
        lambda: fn(coords=hcoords, atomic_numbers=hnums),
        lambda: fn(coordinates=coords, atom_symbols=syms),
        lambda: fn(coords=coords, atom_symbols=syms),
    ]

    errors = []
    for attempt in attempts:
        try:
            out = attempt()
            arr = _flatten_numeric(out)
            if arr.size and np.all(np.isfinite(arr)):
                return arr.astype(float)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    raise ValueError(
        "TSC desc3d.bend_deformed_cylinder_descriptors did not return a numeric vector "
        "for any supported calling convention. First errors: " + " | ".join(errors[:5])
    )
