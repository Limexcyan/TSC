from pathlib import Path
import polars as pl
import plotly.express as px
from plotly.io import to_html

def preliminary_analysis(input_file: Path):
    data = pl.read_csv(input_file)
    cluster_count = data.get_column("centroid").n_unique()
    names = data.get_column("filename").unique()

    bar_stacked_clusters_dataframe = pl.DataFrame(
            {
                "name" : [name for name in names for _ in range(cluster_count)],
                "cluster id" : [i for _ in names for i in range(cluster_count)],
                "compound count" : [len(data.filter(pl.col("filename") == name, pl.col("centroid")==i)) for name in names for i in range(cluster_count)]
            }
        )
    return px.bar(bar_stacked_clusters_dataframe,x="cluster id", y="compound count", color="name", title="Distribution of compounds per cluster")