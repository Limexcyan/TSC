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

st.title('TSC Project')
st.markdown('additional information here')
st.divider()

st.header('Import CHEMBL files to be processed')
upload_files = st.file_uploader('CHEMBL file upload',type=".sdf",accept_multiple_files=True,key='upload1')
st.subheader('After upload, assign metadata to files')
if len(upload_files) <= 0:
    st.markdown('*Please upload files first*')
else:
    for file in upload_files:
        st.markdown('**File: ' + file.name + '**')
        _metadata_target_name = st.text_input('Full target name',key='meta_target_' + file.name)
        _metadata_target_seq = st.text_input('Enzyme sequence',key='meta_sequence_' + file.name)
        upload_metadata.append([('data',file),('target',_metadata_target_name),('seq',_metadata_target_seq)])
        target_count = target_count + 1

st.subheader('If everything is ready, consolidate uploaded files and download them')
if st.button('Consolidate'):
    st.info('Consolidating...')
    consolidated_data = st_consolidate_data(upload_metadata)
    st.success('Done! Preparing file for download...') 
    st.download_button('Download consolidated .csv', consolidated_data.to_csv(index=False).encode('utf-8'), 'consolidated.csv')
st.divider()


st.header('Featurize the consolidated dataset')
st.markdown('Upload the dataset you have consolidated for featurizing')
upload_consolidated = st.file_uploader('CSV file upload',type='.csv',accept_multiple_files=False,key='upload2')
if upload_consolidated is not None:
    consolidated_data = pd.read_csv(upload_consolidated, index_col=False)
st.subheader('Choose featurizer arguments:')
process_mfp_count = st.select_slider('Morgan fingerprint count:',[512,1024,2048],value=1024)
process_mfp_radius = st.select_slider('Morgan fingerprint radius:',[0,1,2,3,4],value=2)
st.subheader('If everything is ready, featurize the consolidated dataset and download it')
if st.button('Featurize'):
    st.info('Please wait, this may take a while...')
    featurizer = ClusteringPreprocessFeaturizer(preprocess_type='quantiles',quant_count=100,morganfp_size=process_mfp_count,morganfp_radius=process_mfp_radius)
    featurized_data = featurizer(consolidated_data)
    st.success('Done! Preparing file for download...')

    st.download_button('Download featurized .csv',featurized_data.to_csv(index=False).encode('utf-8'),'featurized.csv')
st.divider()


st.header('Run clustering on featurized dataset')
st.markdown('Upload the dataset you have featurized for clustering')
upload_featurized = st.file_uploader('upload featurized CSV',type='.csv',accept_multiple_files=False,key='upload3')
if upload_featurized is not None:
    featurized_data = pd.read_csv(upload_featurized,index_col=False)
st.markdown('Upload the respective consolidated dataset to allow for analisys of results')
upload_consolidated = st.file_uploader('upload consolidated CSV',type='.csv',accept_multiple_files=False,key='upload4')
if upload_consolidated is not None:
    consolidated_data = pd.read_csv(upload_consolidated,index_col=False)
st.subheader('Select clustering parameters')
cluster_count = st.select_slider('Cluster count: ', range(20,200,1),value = 50)
st.subheader('If everything is ready, run the clustering algorithm and download the clustering results for analysis')
if st.button('Cluster'):
    cluster_object, clustered_data, _unused_pca_data_3d, _unused_pca_fig = st_cluster(featurized_data,cluster_count)
    # st.pyplot(_unused_pca_fig)
    # st.divider()
    # st.header('Clustering results')
    clustered_data['smiles'] = consolidated_data['smiles']
    clustered_data['target'] = consolidated_data['target']
    st.success('Done! Preparing file for download...')
    # clustered_data = clustered_data.sort_values(by=['distance'])
    # compound_distance = []
    # compound_clusters = []
    # aggregate_compound_clusters = [[0 for j in range()]]

    st.download_button('Download clustering results .csv',clustered_data.to_csv(index=False).encode('utf-8'),'clustering.csv')
st.divider()

st.header('Run analysis on your results')
st.markdown('Feel free to analyze the clustering results on your own, the tools provided within this application are just a simple overview of the clustering')
upload_clustered = st.file_uploader('Upload clustering results CSV',type='.csv',accept_multiple_files=False,key='upload5')
if upload_clustered is not None:
    clustered_data = pd.read_csv(upload_clustered,index_col = False)
    # for index, row in clustered_data.iterrows():
    #     _mol = Chem.MolFromSmiles(row['smiles'])
    #     _mol = Chem.AddHs(_mol)
    #     AllChem.EmbedMolecule(_mol)
    #     AllChem.MMFFOptimizeMolecule(_mol)
    #     _mblock = Chem.MolToMolBlock(_mol)
    #     row['mol_img'] = 

if clustered_data is not None:
    target_count = clustered_data['target'].nunique()
    targets = clustered_data['target'].unique()
    cluster_count = clustered_data['centroid'].nunique()
    st.dataframe(
            clustered_data[['target','centroid','distance','smiles']],
            column_config={
                'target': 'Target',
                'smiles': 'Molecule Smiles',
                'centroid': 'Assigned cluster ID',
                'distance': 'Distance from cluster center'
            },
            hide_index=True
    )
    display_only = st.multiselect('Show target distribution over clusters graph for target:',targets,default=targets)
    if len(display_only) > 0:
        compound_distance = []
        compound_clusters = []
        aggregate_compound_clusters = [[0 for j in range(cluster_count)] for i in range(len(display_only))]
        compound_norm = []
        for i in range(len(display_only)):
            _temp_d, _temp_c = cal_centroidanalysis(clustered_data[clustered_data['target'] == display_only[i]],cluster_count)
            compound_distance.append(_temp_d)
            compound_norm.append(_temp_c / np.sum(_temp_c))
            compound_clusters.append(_temp_c)
            for j in range(i,len(display_only)):
                aggregate_compound_clusters[j] = np.add(aggregate_compound_clusters[j], _temp_c)
        display_fig = plt.figure(figsize=(7,7))
        ax = display_fig.add_axes([0,0,1,1])
        _ = ax.set_xlim(-1,cluster_count), ax.set_xticks([i for i in range(0,cluster_count, cluster_count // 25)])
        for i in range(len(display_only)):
            ax.bar(range(cluster_count),aggregate_compound_clusters[len(display_only) - i - 1], label=display_only[len(display_only) - i - 1])
        plt.xlabel('centroid id')
        plt.ylabel('classified amount')
        plt.legend()
        st.pyplot(display_fig)
            
