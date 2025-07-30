import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from desc3d import *

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


def cluster_smiles_csv(csv_path, cluster_count=5, max_mols=100):
    df = pd.read_csv(csv_path)
    if 'smiles' not in df.columns:
        raise ValueError("Expected a 'smiles' column in CSV.")

    features, names = [], []
    for i, row in df.iterrows():
        if int(i) >= max_mols:
            break
        smiles = row['smiles']
        name = row['name'] if 'name' in row else f'mol_{i}'
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        coords = optimize_molecule_3d(mol)
        if coords is None:
            continue
        try:
            coords = np.array(coords)
            sh_desc = pca_spherical_harmonics(coords)
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

def rif_cluster_smiles_csv(csv_path, cluster_count=5, max_mols=100):
    df = pd.read_csv(csv_path)
    if 'smiles' not in df.columns:
        raise ValueError("Expected a 'smiles' column in CSV.")

    features, names = [], []
    for i, row in df.iterrows():
        if int(i) >= max_mols:
            break
        smiles = row['smiles']
        name = row['name'] if 'name' in row else f'mol_{i}'
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        coords = optimize_molecule_3d(mol)
        if coords is None:
            continue
        try:
            coords = np.array(coords)
            sh_desc = rif_sh_descriptors(coords)
            features.append(sh_desc)
            names.append(name)
        except Exception as e:
            print(f"[WARN] Failed descriptor for {name}: {e}")

    features = np.array(features)
    if features.shape[0] == 0:
        raise ValueError("No valid molecules/descriptors found.")

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=cluster_count, random_state=42)
    labels = kmeans.fit_predict(features_scaled)

    # Visualization
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features_scaled)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='rainbow', s=50)
    plt.title("RIF-SH Descriptor Clustering")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)
    plt.show()

    return pd.DataFrame({'name': names, 'cluster': labels})

if __name__ == "__main__":
    result_df = rif_cluster_smiles_csv("data/proper_sdf/molecule_tree/full_sdfs.csv", cluster_count=5, max_mols=100)
    print(result_df.head())
