import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem

def optimize_molecule_3d(mol, use_mmff=True, max_iters=200):
    mol = Chem.AddHs(mol)

    # 3D embedding
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    success = AllChem.EmbedMolecule(mol, params)

    if success != 0:
        raise ValueError("Embedding failed for molecule.")

    # Optimization
    if use_mmff:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is None:
            raise ValueError("MMFF properties unavailable.")
        result = AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=max_iters)
    else:
        result = AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)

    if result != 0:
        print("Warning: Optimization may not have fully converged.")

    # Extract 3D coordinates
    conf = mol.GetConformer()
    positions = []
    for atom_idx in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(atom_idx)
        positions.append([pos.x, pos.y, pos.z])

    return mol, positions

df = pd.read_csv("data/full_sdfs.csv")
df = df['smiles']
for i in range(len(df)):
    smiles = df[i+1]
    mol = Chem.MolFromSmiles(smiles)

    optimized_mol, coords = optimize_molecule_3d(mol, use_mmff=True)

    coords_np = np.array(coords)
    print("3D coordinates (shape={}):".format(coords_np.shape))
    print(coords_np)

