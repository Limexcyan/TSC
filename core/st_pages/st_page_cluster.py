import streamlit as st
from st_clusters_features import *

st.header('Run clustering on featurized dataset')
st.markdown('Upload the dataset you have featurized for clustering')
volatile_upload_featurized = st.file_uploader('upload featurized CSV',type='.csv',accept_multiple_files=False,key='upload3')
st.session_state['upload_featurized'] = volatile_upload_featurized

st.markdown('Upload the respective consolidated dataset to allow for analisys of results')
volatile_upload_consolidated = st.file_uploader('upload consolidated CSV',type='.csv',accept_multiple_files=False,key='upload4')
st.session_state['upload_consolidated'] = volatile_upload_consolidated

if 'upload_featurized' not in st.session_state or st.session_state['upload_featurized'] == None or 'upload_consolidated' not in st.session_state or st.session_state['upload_consolidated'] == None:
    st.markdown('*Please upload featurized and consolidated datasets first*')
else:
    st.session_state['featurized_data'] = pd.read_csv(st.session_state['upload_featurized'],index_col=False)
    st.session_state['consolidated_data'] = pd.read_csv(st.session_state['upload_consolidated'],index_col=False)

    st.subheader('Select clustering parameters')
    cluster_count = st.select_slider('Cluster count: ', range(20,200,1),value = 50)

    st.subheader('Select clustered data export options')
    keep_fingerprints = st.checkbox('Keep morgan fingerprints in output', value = False)
    keep_smiles = st.checkbox('Keep smiles molecule representation in output', value = True)
    keep_target = st.checkbox('Keep full target names in output', value = True)

    st.subheader('If everything is ready, run the clustering algorithm and download the clustering results for analysis')
    if st.button('Cluster'):
        st.session_state['is_cluster_done'] = None
        volatile_cluster_object, volatile_clustered_data, _unused_pca_data_3d, _unused_pca_fig = st_cluster(st.session_state['featurized_data'],cluster_count)
        st.session_state['cluster_object'] = volatile_cluster_object
        st.success('Done! Preparing file for download...')
        if not keep_fingerprints:
            _fpncc = st.session_state['featurized_data'].shape[1]
            _fpncs = ["mfp_" + str(i) for i in range(_fpncc)]
            print(_fpncc)
            volatile_clustered_data = volatile_clustered_data.drop(_fpncs, axis=1)
        if keep_smiles:
            volatile_clustered_data['smiles'] = st.session_state['consolidated_data']['smiles']
        if keep_target:
            volatile_clustered_data['target'] = st.session_state['consolidated_data']['target']
        st.session_state['is_cluster_done'] = 1
        st.session_state['clustered_data'] = volatile_clustered_data

    if 'is_cluster_done' not in st.session_state or st.session_state['is_cluster_done'] == None:
        _ = None
        #do nothing
    else:
        st.download_button('Download clustering results .csv',st.session_state['clustered_data'].to_csv(index=False).encode('utf-8'),'clustering.csv')
