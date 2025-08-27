import streamlit as st
from featurizer import *

st.header('Featurize the consolidated dataset')
st.markdown('Upload the dataset you have consolidated for featurizing')
volatile_upload_consolidated = st.file_uploader('CSV file upload',type='.csv',accept_multiple_files=False,key='upload2')
st.session_state['upload_consolidated'] = volatile_upload_consolidated

if 'upload_consolidated' not in st.session_state or st.session_state['upload_consolidated'] == None:
    st.markdown('*Please upload consolidated dataset first*')
else:
    consolidated_data = pd.read_csv(st.session_state['upload_consolidated'], index_col=False)
    st.subheader('Choose featurizer arguments:')
    process_mfp_count = st.select_slider('Morgan fingerprint count:',[512,1024,2048],value=1024)
    process_mfp_radius = st.select_slider('Morgan fingerprint radius:',[0,1,2,3],value=2)
    st.subheader('If everything is ready, featurize the consolidated dataset and download it')
    if st.button('Featurize'):
        st.session_state['is_featurizer_complete'] = None
        st.info('Please wait, this may take a while...')
        featurizer = ClusteringPreprocessFeaturizer(preprocess_type='quantiles',quant_count=100,morganfp_size=process_mfp_count,morganfp_radius=process_mfp_radius)
        featurized_data = featurizer(consolidated_data)
        st.session_state['post_featurized_data'] = featurized_data
        st.session_state['is_featurizer_complete'] = 1
        st.success('Done! Preparing file for download...')
    
    if 'is_featurizer_complete' not in st.session_state or st.session_state['is_featurizer_complete'] == None:
        _ = None
        #do nothing
    else:
        st.download_button('Download featurized .csv',st.session_state['post_featurized_data'].to_csv(index=False).encode('utf-8'),'featurized.csv')
