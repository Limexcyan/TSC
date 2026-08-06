from enum import Enum
from ..common.settings import ConfigBase
from sklearn.cluster import BisectingKMeans

import polars as pl

import logging

logger = logging.getLogger(__name__)

class ClusteringBuffers(Enum):
    BISECTING_K_MEANS = "bisecting_k_means"

class ClusteringConfig(ConfigBase):
    cluster_count: int = 64
    use_clustering: ClusteringBuffers = ClusteringBuffers.BISECTING_K_MEANS

type ClusteringBuffer = BisectingKMeans

def clustering_get_distances(data: pl.DataFrame, buffer: ClusteringBuffer) -> pl.DataFrame:
    return pl.from_numpy(buffer.transform(data)).select(pl.min_horizontal("*")).to_series()

def clustering_get_centroids(data: pl.DataFrame, buffer: ClusteringBuffer) -> pl.Series:
    return pl.from_numpy(buffer.predict(data)).to_series()

def clustering(data: pl.DataFrame, config: ClusteringConfig) -> ClusteringBuffer:
    match config.use_clustering:
        case ClusteringBuffers.BISECTING_K_MEANS:
            return clustering_bisecting_k_means(data, config)
        case _:
            raise ValueError(f"Clustering buffer {config.use_clustering} does not exist")

def clustering_bisecting_k_means(data: pl.DataFrame, config: ClusteringConfig) -> BisectingKMeans:
    logger.info("clustering with %d clusters on dataset with %d rows", config.cluster_count, data.shape[0])
    clustering_buffer = BisectingKMeans(n_clusters=config.cluster_count)
    clustering_buffer.fit(data)
    return clustering_buffer
