import pandas as pd
import plotly.express as px
from plotly.io import to_html
from sklearn.decomposition import PCA

class Analysis():
    def __init__(self, csv_data):
        self.data = pd.read_csv(csv_data)
        self.cluster_count = self.data['centroid'].nunique()
        self.names = self.data['origin'].unique()
        self.feature_cols = self.data.filter(like="mfp",axis=1).columns.tolist()
        self.bar_stacked_clusters = None
    
    def __call__(self):
        bar_stacked_clusters_dataframe = pd.DataFrame(
            {
                "name" : [name for name in self.names for _ in range(self.cluster_count)],
                "cluster id" : [i for _ in self.names for i in range(self.cluster_count)],
                "compound count" : [len(self.data.query(f'origin == "{name}" and centroid == {i}')) for name in self.names for i in range(self.cluster_count)]
            }
        )
        self.bar_stacked_clusters = px.bar(bar_stacked_clusters_dataframe, x="cluster id", y="compound count", color="name", title="Distribution of compounds per cluster")

        return self

    def get_bar_stacked_clusters(self):
        return to_html(self.bar_stacked_clusters,full_html=False,include_plotlyjs='cdn')
