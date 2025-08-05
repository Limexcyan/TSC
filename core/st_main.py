import streamlit as st
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
from st_clusters_features import *
from analysis import *
from config import *
import os

upload_metadata = []
upload_files = []
target_count = 0
targets = []

consolidated_data = None
upload_consolidated = None

process_mfp_count = 0
process_mfp_radius = 0
featurized_data = None
upload_featurized = None

cluster_count = 0

upload_clustered = None
clustered_data = None


st_page_preprocess = st.Page("st_pages/st_page_preprocess.py", title="1. Preprocess data")
st_page_featurize = st.Page("st_pages/st_page_featurize.py", title="2. Featurize dataset")
st_page_cluster = st.Page("st_pages/st_page_cluster.py", title="3. Run clustering")
st_page_analyze = st.Page("st_pages/st_page_analyze.py", title="Analyze results")

pg = st.navigation(
        {
            "Clustering": [st_page_preprocess, st_page_featurize, st_page_cluster],
            "Analysis": [st_page_analyze]
        }
)


st.set_page_config(
        page_title = "TSC Project",
        layout = "wide"

)

pg.run()
