import pandas as pd

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools as pdt
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import rdFMCS as fmcs

import seaborn as sns

from matplotlib import pyplot as plt

from sklearn.cluster import BisectingKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils import *
from featurizer import *
from clusters_features import *

if __name__ == "__main__":
    dropped_columns = ['CdId', 'Mol Weight', 'Formula', 'target_id', 'target_name', 'organism', 'target_type', 'ID',
                       'std_unit', 'std_type']

    data_path = f'data/proper_sdf'

    sdfs = ['data/proper_sdf/CHEMBL2147.sdf', 'data/proper_sdf/CHEMBL2343.sdf', 'data/proper_sdf/CHEMBL5794.sdf']
    userinputs = [[('target', 'threonine-protein kinase PIM1')], [('target', 'threonine-protein kinase Sgk1')],
                  [('target', 'threonine-protein kinase Sgk2')]]

    full_df = consolidate_data(sdfs, userinputs=userinputs)

    export_data(full_df, "data/full_sdfs.csv")

    feature_count = 1024
    quantile_count = 100
    featurizer = ClusteringFeaturizer(activity_dist='quantiles', qcount=quantile_count, length=feature_count)
    feature_count = (feature_count) * 1.0
    datas = featurizer(full_df)

    export_data(datas, "data/processed_sdfs.csv")

    a1, a2, a3, a4 = perform_analysis(datas,50,full_df)


