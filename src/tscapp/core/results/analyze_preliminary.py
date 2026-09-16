from plotly.io import to_html
from pathlib import Path

import plotly.express as px
import polars as pl

def preliminary_analysis(input_file: Path, output_file: Path):
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
    graph = px.bar(bar_stacked_clusters_dataframe,x="cluster id", y="compound count", color="name", title="Distribution of compounds per cluster")
    with open(output_file, "w") as fo:
        fo.write(to_html(graph,full_html=False,include_plotlyjs="cdn"))