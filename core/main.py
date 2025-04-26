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

from scipy.special import rel_entr, kl_div
from Bio import pairwise2


from utils import *
from featurizer import *
from clusters_features import *
from analysis import *
from config import *

if __name__ == "__main__":

    full_df = consolidate_data(sdfs, userinputs=userinputs)

    export_data(full_df, "data/full_sdfs.csv")

    featurizer = ClusteringPreprocessFeaturizer(preprocess_type='quantiles', quant_count=quantile_count, morganfp_size=feature_count)
    feature_count = feature_count * 1.0
    datas = featurizer(full_df)

    export_data(datas, "data/processed_sdfs.csv")

    out_data_full, out_data_processed = perform_analysis(datas,50,full_df)

    targettocolor(out_data_full, 50)
    summary(out_data_full, 1e-20)




