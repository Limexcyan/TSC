from random import getrandbits
from random import random
import logging
from enum import Enum

import polars as pl
from sklearn.cluster import BisectingKMeans

from ..common.settings import ConfigBase

logger = logging.getLogger(__name__)

class ClusteringBuffers(Enum):
    BISECTING_K_MEANS = "bisecting_k_means"

class ClusteringConfig(ConfigBase):
    cluster_count: int = 64
    seed: int | None = None
    use_clustering: ClusteringBuffers = ClusteringBuffers.BISECTING_K_MEANS

type ClusteringBuffer = BisectingKMeans

def clustering_get_distances(data: pl.DataFrame, buffer: ClusteringBuffer) -> pl.DataFrame:
    return pl.from_numpy(buffer.transform(data)).select(pl.min_horizontal("*")).to_series()

def clustering_get_centroids(data: pl.DataFrame, buffer: ClusteringBuffer) -> pl.Series:
    return pl.from_numpy(buffer.predict(data)).to_series()

def clustering(data: pl.DataFrame, config: ClusteringConfig) -> ClusteringBuffer:
    match config.use_clustering:
        case ClusteringBuffers.BISECTING_K_MEANS:
            return clustering_bisecting_k_means(data.select(r"^mfp.*$"), config)
        case _:
            raise ValueError(f"Clustering buffer {config.use_clustering} does not exist")

def clustering_bisecting_k_means(data: pl.DataFrame, config: ClusteringConfig) -> tuple[BisectingKMeans, int]:
    logger.info("clustering with %d clusters on dataset with %d rows", config.cluster_count, data.shape[0])
    if config.seed:
        seed = config.seed
        clustering_buffer = BisectingKMeans(n_clusters=config.cluster_count, random_state=config.seed)
    else:
        seed = getrandbits(32)
        clustering_buffer = BisectingKMeans(n_clusters=config.cluster_count, random_state=seed)
    clustering_buffer.fit(data)
    return (clustering_buffer, seed)
