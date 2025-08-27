import streamlit as st
import pandas as pd
import numpy as np

from analysis import *

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.palettes import viridis

st.header('Run analysis on your results')
st.markdown('Feel free to analyze the clustering results on your own, the tools provided within this application are just a simple overview of the clustering')
volatile_upload_clustered = st.file_uploader('Upload clustering results CSV',type='.csv',accept_multiple_files=False,key='upload5')
st.session_state['upload_clustered'] = volatile_upload_clustered

if 'upload_clustered' not in st.session_state or st.session_state['upload_clustered'] == None:
    st.markdown("*Please upload clustered data for analysis*")
else:
    clustered_data = pd.read_csv(st.session_state['upload_clustered'],index_col = False)
    if 'target' not in clustered_data.columns:
        st.markdown("**Cannot run analysis on clustered data with unknown targets. Export the clustered data while keeping full target names in output**")
    else:
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
        graph = pd.DataFrame(columns=np.append(targets, np.array(['centroid'])))
        graph['centroid'] = range(cluster_count)
        for target in targets:
            for cid in range(cluster_count):
                count = clustered_data[(clustered_data['target'] == target) & (clustered_data['centroid']==cid)].count()[0]
                graph.at[cid, target] = count
        p = figure(x_range = (-1,cluster_count), x_axis_label="centroid id", title="molecule count by centroid id", tools="hover,box_zoom,box_select,reset", tooltips="$name: @$name")
        p.vbar_stack(targets, x='centroid', source=ColumnDataSource(graph), width=0.9, color=viridis(len(targets)), legend_label=targets.tolist())
        st.bokeh_chart(p)
        # display_only = st.multiselect('Show target distribution over clusters graph for target:',targets,default=targets)
        # if len(display_only) > 0:
        #     compound_distance = []
        #     compound_clusters = []
        #     aggregate_compound_clusters = [[0 for j in range(cluster_count)] for i in range(len(display_only))]
        #     compound_norm = []
        #     for i in range(len(display_only)):
        #         _temp_d, _temp_c = cal_centroidanalysis(clustered_data[clustered_data['target'] == display_only[i]],cluster_count)
        #         compound_distance.append(_temp_d)
        #         compound_norm.append(_temp_c / np.sum(_temp_c))
        #         compound_clusters.append(_temp_c)
        #         for j in range(i,len(display_only)):
        #             aggregate_compound_clusters[j] = np.add(aggregate_compound_clusters[j], _temp_c)
        #     display_fig = plt.figure(figsize=(7,7))
        #     ax = display_fig.add_axes([0,0,1,1])
        #     _ = ax.set_xlim(-1,cluster_count), ax.set_xticks([i for i in range(0,cluster_count, cluster_count // 25)])
        #     for i in range(len(display_only)):
        #         ax.bar(range(cluster_count),aggregate_compound_clusters[len(display_only) - i - 1], label=display_only[len(display_only) - i - 1])
        #     plt.xlabel('centroid id')
        #     plt.ylabel('classified amount')
        #     plt.legend()
        #     st.pyplot(display_fig)
