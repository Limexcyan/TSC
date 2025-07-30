import pandas as pd

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools as pdt
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import rdFMCS as fmcs

from matplotlib import pyplot as plt

from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import ipywidgets

def cluster(dataset,n):
    clustering = BisectingKMeans(n_clusters=n)
    clusterdist = clustering.fit_transform(dataset)
    clusterid = clustering.labels_
    clusterdist = clusterdist[np.arange(len(dataset)),clusterid]
    processed_data = dataset.copy()
    processed_data['centroid'] = clusterid
    processed_data['distance'] = clusterdist
    pca = PCA(n_components=3)
    pcdata3d = pd.DataFrame(pca.fit_transform(processed_data.drop(["centroid","distance"], axis=1)))
    pcdata3d.columns = ["pc1", "pc2", "pc3"]

    sizes = np.power(10,6)/np.power(processed_data["distance"],7)
    fig = plt.figure(figsize=(11,9))
    ax = fig.add_subplot(projection='3d')
    ax.azim = 35
    ax.elev = 35
    sc = ax.scatter(pcdata3d["pc1"],pcdata3d["pc2"],pcdata3d["pc3"],cmap=plt.cm.gist_rainbow,c=processed_data["centroid"],s=sizes)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.show()
    return clustering, processed_data, pcdata3d

def showcluster(cid,dat,pcdat):
    cdata = dat[dat["centroid"]==cid]['distance']
    pickcluster = pcdat.loc[dat["centroid"]==cid]
    sizes = np.power(10,6)/np.power(cdata,7)
    print(sizes.shape)
    print(pickcluster.shape)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(projection='3d')
    ax.azim = 35
    ax.elev = 35
    sc = ax.scatter(pickcluster["pc1"],pickcluster["pc2"],pickcluster["pc3"],cmap=plt.cm.viridis,s=sizes)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)


def getAtoms(df):
    mols = []
    for row in df:
        smiles = row
        mol = Chem.MolFromSmiles(smiles)
        mols.append(mol)
    return mols

def findmcs(cid,dat,ogdat):
    centroid = dat.loc[dat['centroid'] == cid, :]
    smilesss = ogdat.loc[dat['centroid']==cid, :]
    act = smilesss['std_value']
    dst = centroid['distance']
    mol = smilesss['smiles'][:10]
    if (mol.empty or mol.size < 2):
        print("cluster did not converge close enough")
        #plt.scatter(dst,act)
        #plt.show()
    else:
        mol = mol.reset_index(drop=True)
        #print(mol)
        mcs = Chem.MolFromSmarts(fmcs.FindMCS(getAtoms(mol),timeout=1).smartsString)
        #plt.scatter(dst,act)
        #plt.show()
        img = Chem.Draw.MolToImage(mcs)
        return img

def show_molecule(smiles=''):
    if not smiles:
        return
    mol = Chem.MolFromSmiles(smiles)
    display(mol)
    if not mol:
        return
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    mblock = Chem.MolToMolBlock(mol, kekulize=False)

    # 3D VIEW:
    #view = py3Dmol.view(width=800, height=500)
    #view.addModel(Chem.MolToMolBlock(mol, kekulize=True), 'mol')
    #view.setStyle({'model': -1}, {"stick": {'color': 'white', 'radius': 0.15}, "sphere": {'radius': .4}})
    #view.zoomTo({'model': -1})
    #view.show()


def examine_cluster(dat,dat_sorted,cid,eid):
    centroid = dat_sorted.loc[dat['centroid'] == cid, :]
    print(eid,"/",len(centroid))
    print(centroid['distance'].iloc[eid])
    mol = centroid['smiles']
    #mol = mol.reset_index()
    print(centroid['target'].iloc[eid])
    show_molecule(mol.iloc[eid])

def centroidanalysis(dat,ccount):
    dsts = []
    count = []
    for i in range(ccount):
        centroid = dat[dat['centroid']==i]
        dsts.append(np.sum(centroid['distance']))
        count.append(len(centroid))
    plt.scatter(count,dsts)
    plt.title("centroid count vs total distance")
    plt.xlabel("count")
    plt.ylabel("distance")
    plt.show()

def cal_centroidanalysis(dat,ccount):
    dsts = []
    count = []
    for i in range(ccount):
        centroid = dat[dat['centroid']==i]
        dsts.append(np.sum(centroid['distance']))
        count.append(len(centroid))
    return dsts, count

def ccount(dat,cs):
    count = []
    for i in range(cs):
        centroid = dat[dat['centroid']==i]
        count.append(len(centroid))
    plt.bar(range(cs),count)
    plt.title("centroid count")
    plt.xlabel("centroid")
    plt.ylabel("amount")
    plt.show()

def cal_ccount(dat,cs):
    count = []
    for i in range(cs):
        centroid = dat[dat['centroid']==i]
        count.append(len(centroid))
    return count

def perform_analysis(dat,n,og):
    cl, processed, pcadata = cluster(dat,n)
    ccount(processed,n)
    centroidanalysis(processed,n)
    dat_sorted = processed.copy()
    dat_sorted['smiles'] = og['smiles']
    dat_sorted['target'] = og['target']
    dat_sorted = dat_sorted.sort_values(by=['distance'])
    return dat_sorted, processed

