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

def st_cluster(dataset,n):
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
    return clustering, processed_data, pcdata3d, fig


def cal_centroidanalysis(dat,ccount):
    dsts = []
    count = []
    for i in range(ccount):
        centroid = dat[dat['centroid']==i]
        dsts.append(np.sum(centroid['distance']))
        count.append(len(centroid))
    return dsts, count

