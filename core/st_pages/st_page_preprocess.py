import streamlit as st
from utils import *

volatile_upload_metadata = []

st.header('Import CHEMBL files to be processed')
volatile_upload_files = st.file_uploader('CHEMBL file upload',type=".sdf",accept_multiple_files=True,key='upload1')
st.session_state['upload_files'] = volatile_upload_files

st.subheader('After upload, assign metadata to files')

if 'upload_files' not in st.session_state or st.session_state['upload_files'] == None:
    st.markdown('*Please upload files first*')
else:
    for file in st.session_state['upload_files']:
        st.markdown('**File: ' + file.name + '**')
        _metadata_target_name = st.text_input('Full target name',key='meta_target_' + file.name)
        # _metadata_target_seq = st.text_input('Enzyme sequence',key='meta_sequence_' + file.name)
        volatile_upload_metadata.append([('data',file),('target',_metadata_target_name),('seq',0)])

st.subheader('If everything is ready, consolidate uploaded files and download them')
if st.button('Consolidate'):
    st.session_state['is_preprocess_complete'] = None
    st.info('Consolidating...')
    consolidated_data = st_consolidate_data(volatile_upload_metadata)
    st.session_state['post_preprocess_data'] = consolidated_data
    st.session_state['is_preprocess_complete'] = 1
    st.success('Done! Preparing file for download...')
if 'is_preprocess_complete' not in st.session_state or st.session_state['is_preprocess_complete'] == None:
    _ = None
    #do nothing
else:
    st.download_button('Download consolidated .csv', st.session_state['post_preprocess_data'].to_csv(index=False).encode('utf-8'), 'consolidated.csv')
