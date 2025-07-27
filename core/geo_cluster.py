import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from numpy import linalg as la
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
import scipy


# --- Spherical Harmonics ---
def rsh(m, l, ph, th):
    if m < 0:
        tm = scipy.special.sph_harm(-m, l, ph, th).imag
    else:
        tm = scipy.special.sph_harm(m, l, ph, th).real
    if m == 0:
        tm = tm / math.sqrt(2)
    return math.sqrt(2) * pow(-1, m) * tm


def PCA_SH(coords, maxl=3):
    coords = np.array(coords)
    avg = coords.mean(axis=0)
    coords -= avg
    L = len(coords)
    ev = np.transpose(la.eig(np.cov(np.transpose(coords)))[1])

    for i in range(2):
        proj = coords.dot(ev[i])
        if -min(proj) > max(proj):
            ev[i] = -ev[i]
    if la.det(ev) < 0:
        ev[2] = -ev[2]

    coords = coords.dot(np.transpose(ev))

    r, th, ph = np.zeros(L), np.zeros(L), np.zeros(L)
    for i in range(L):
        r[i] = la.norm(coords[i])
        th[i] = math.acos(coords[i, 2] / r[i])
        denom = r[i] * math.sin(th[i])
        ph[i] = math.acos(coords[i, 0] / denom) if denom > 1e-6 else 0

    harm = []
    for l in range(maxl):
        for m in range(-l, l+1):
            fun = np.array([rsh(m, l, ph[i], th[i]) for i in range(L)])
            coeff = r.dot(fun) / np.sum(fun * fun)
            harm.append(coeff)
    return harm


# --- 3D Optimization ---
def optimize_molecule_3d(mol, use_mmff=True, max_iters=200):
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    if use_mmff and AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=max_iters)
    else:
        AllChem.UFFOptimizeMolecule(mol, maxIters=max_iters)

    conf = mol.GetConformer()
    coords = [[conf.GetAtomPosition(i).x,
               conf.GetAtomPosition(i).y,
               conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())]
    return coords


# --- Main Process ---
def cluster_smiles_csv(csv_path, cluster_count=5, max_mols=1000):
    df = pd.read_csv(csv_path)
    if 'smiles' not in df.columns:
        raise ValueError("Expected a 'smiles' column in CSV.")

    features, names = [], []
    for i, row in df.iterrows():
        # if i >= max_mols:
        #     break
        smiles = row['smiles']
        name = row['name'] if 'name' in row else f'mol_{i}'
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        coords = optimize_molecule_3d(mol)
        if coords is None:
            continue
        try:
            sh_desc = PCA_SH(coords, maxl=3)
            features.append(sh_desc)
            names.append(name)
        except Exception as e:
            print(f"[WARN] Failed SH for {name}: {e}")

    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=cluster_count, random_state=42)
    labels = kmeans.fit_predict(features_scaled)

    # Visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='rainbow', s=50)
    plt.title("3D SH Descriptor Clustering")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()

    return pd.DataFrame({'name': names, 'cluster': labels})


# --- Example Usage ---
if __name__ == "__main__":
    result_df = cluster_smiles_csv("data/full_sdfs.csv", cluster_count=5, max_mols=100)
    print(result_df.head())
