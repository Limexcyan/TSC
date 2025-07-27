import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import BisectingKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import math
from numpy import linalg as la
import scipy
import matplotlib
from utils import *
matplotlib.use('TkAgg')

# ----------------- Spherical Harmonics Descriptor ----------------- #

def rsh(m, l, ph, th):
    if (m < 0):
        tm = scipy.special.sph_harm(-m,l,ph,th).imag
    else:
        tm = scipy.special.sph_harm(m,l,ph,th).real
    if m == 0:
        tm /= math.sqrt(2)
    return math.sqrt(2) * pow(-1, m) * tm


def PCA_SH(coords, maxl=3):
    coords = np.array(coords)
    avg = coords.mean(axis=0)
    coords -= avg
    L = len(coords)
    ev = np.transpose(la.eig(np.cov(coords.T))[1])
    for i in range(2):
        proj = coords @ ev[i]
        if -min(proj) > max(proj):
            ev[i] = -ev[i]
    if la.det(ev) < 0:
        ev[2] = -ev[2]
    coords = coords @ ev.T

    r, th, ph = np.zeros(L), np.zeros(L), np.zeros(L)
    for i in range(L):
        r[i] = la.norm(coords[i])
        th[i] = math.acos(coords[i, 2] / r[i]) if r[i] > 1e-5 else 0
        denom = r[i] * math.sin(th[i])
        ph[i] = math.acos(coords[i, 0] / denom) if denom > 1e-6 else 0

    harm = []
    for l in range(maxl):
        for m in range(-l, l+1):
            fun = np.array([rsh(m, l, ph[i], th[i]) for i in range(L)])
            coeff = r @ fun / np.sum(fun * fun)
            harm.append(coeff)
    return harm


# ----------------- 3D Optimization ----------------- #

def optimize_molecule_3d(mol):
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        return None
    if AllChem.MMFFHasAllMoleculeParams(mol):
        AllChem.MMFFOptimizeMolecule(mol)
    else:
        AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    return [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
            for i in range(mol.GetNumAtoms())]


# ----------------- Clustering & Analysis ----------------- #

def cluster_SH(df, cluster_count=5, max_mols=100, maxl=3):
    features, names, smiles = [], [], []

    for i, row in df.iterrows():
        if i >= max_mols:
            break
        smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        coords = optimize_molecule_3d(mol)
        if coords is None:
            continue
        try:
            desc = PCA_SH(coords, maxl=maxl)
            features.append(desc)
            names.append(row.get('target', f'mol_{i}'))
            smiles.append(smi)
        except Exception as e:
            print(f"Skipped {i}: {e}")

    X = StandardScaler().fit_transform(features)
    clustering = BisectingKMeans(n_clusters=cluster_count, random_state=42)
    dist = clustering.fit_transform(X)
    labels = clustering.labels_
    dists = dist[np.arange(len(X)), labels]

    df_clustered = pd.DataFrame(X)
    df_clustered['centroid'] = labels
    df_clustered['distance'] = dists
    df_clustered['target'] = names
    df_clustered['smiles'] = smiles

    return df_clustered


def visualize_clusters_3D(df_clustered):
    pca = PCA(n_components=3)
    pcadata = pd.DataFrame(pca.fit_transform(df_clustered.drop(columns=["centroid", "distance", "target", "smiles"])))
    pcadata.columns = ["pc1", "pc2", "pc3"]
    df_clustered = df_clustered.reset_index(drop=True)
    pcadata['centroid'] = df_clustered['centroid']
    pcadata['distance'] = df_clustered['distance']

    sizes = np.power(10, 6) / np.power(pcadata["distance"] + 1e-4, 7)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(projection='3d')
    ax.azim = 35
    ax.elev = 35
    sc = ax.scatter(pcadata["pc1"], pcadata["pc2"], pcadata["pc3"],
                    c=pcadata["centroid"], cmap='gist_rainbow', s=sizes)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.title("Spherical Harmonics Clustering (PCA 3D)")
    plt.show()
    return pcadata


def cluster_stats(df_clustered, cluster_count):
    counts = df_clustered['centroid'].value_counts().sort_index()
    plt.bar(range(cluster_count), counts)
    plt.title("Cluster Sizes")
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.show()

    dist_sums = df_clustered.groupby('centroid')['distance'].sum()
    plt.scatter(counts.values, dist_sums.values)
    plt.title("Cluster Size vs Total Distance")
    plt.xlabel("Count")
    plt.ylabel("Total Distance")
    plt.show()


# ----------------- Usage Example ----------------- #

if __name__ == "__main__":
    df = pd.read_csv("data/proper_sdf/receptors_tree/full_sdfs.csv")  # expects columns: smiles, target
    clustered_df = cluster_SH(df, cluster_count=20, max_mols=200)
    export_data(clustered_df, "herehere")
    #pcadata = visualize_clusters_3D(clustered_df)
    #cluster_stats(clustered_df, cluster_count=20)
