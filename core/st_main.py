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
from clusters_features import *
from analysis import *
from config import *
import os

st.title('TSC Project')
st.header('Step 1. Import CHEMBL files to be processed')
upload = st.file_uploader('CHEMBL file upload',type='.sdf',accept_multiple_files=True)
metadata = []
consolidated_sdfs = None
st.subheader('After upload, assign metadata to files:')
if len(upload) <= 0:
    st.markdown('*Upload sdf files first*')
else:
    for file in upload:
        st.markdown('**File: ' + file.name + '**')
        d_name = st.text_input('Full target name',key=file.name + 'name')
        d_seq = st.text_input('Enzyme sequence',key=file.name + 'seq')
        metadata.append([('file',file),('target',d_name),('seq',d_seq)])
st.subheader('If everything is ready, consolidate the SDFs')
clicked_consolidate = st.button('Consolidate')
if clicked_consolidate:
    consolidated_sdfs = st_consolidate_data(metadata)
    export_data(consolidated_sdfs,"../data/full_sdfs.csv")
    st.download_button('Download consolidated .csv', consolidated_sdfs.to_csv(index=False).encode('utf-8'), 'consolidated.csv')

st.header('Step 2. Featurize the consolidated dataset')
st.markdown('**Optional** Upload a consolidated dataset (do this if you have skipped step 1)')
upload_consolidated = st.file_uploader('CSV file upload',type='.csv',accept_multiple_files=False)
if upload_consolidated is not None:
    consolidated_sdfs = pd.read_csv(upload_consolidated,index_col=False)
st.subheader('Select featurizer arguments:')
mfp_count = st.select_slider('Morgan fingerprint count: ', [512,1024,2048],value = 1024)
mfp_radius = st.select_slider('Morgan fingerprint radius: ', [0,1,2,3,4],value = 2)
st.subheader('If everything is ready, featurize the data')
clicked_featurize = st.button('Featurize')
featurize_status = 0
def featurize_update(status):
    featurize_status = status
    print(featurize_status)
if clicked_featurize:
    featurizer = ClusteringPreprocessFeaturizer(preprocess_type='quantiles', quant_count=100, morganfp_size=mfp_count, morganfp_radius=mfp_radius)
    featurized_data = featurizer(consolidated_sdfs,update_callback=featurize_update)
    export_data(featurized_data,"../data/processed_sdfs.csv")
    st.download_button('Download featurized .csv', featurized_data.to_csv(index=False).encode('utf-8'), 'featurized.csv')

st.header('Step 3. Run clustering on featurized dataset')
st.markdown('**Optional** Upload a featurized dataset (do this if you have skipped step 2)')
upload_featurized = st.file_uploader('CSV file upload',type='.csv',accept_multiple_files=False,key='featurized_upload')
if upload_featurized is not None:
    featurized_data = pd.read_csv(upload_featurized,index_col=False)
st.markdown('**Optional** Upload a consolidated dataset (do this if you have skipped step 1)')
upload_consolidated = st.file_uploader('CSV file upload',type='.csv',accept_multiple_files=False,key='consolidated_upload')
if upload_consolidated is not None:
    consolidated_sdfs = pd.read_csv(upload_consolidated,index_col=False)
st.subheader('Select clustering arguments:')
cluster_count = st.select_slider('Cluster count: ', range(20,200,1),value = 50)
st.subheader('If everything is ready, run the clustering algorithm')
clicked_cluster = st.button('Cluster')
if clicked_cluster:
    cl, processed, pcdata, fig = st_cluster(featurized_data,cluster_count)
    st.pyplot(fig)
    
