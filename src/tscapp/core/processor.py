from pathlib import Path
import logging
from ..common.cli import CliRunnerMixin
from ..common.settings import ConfigBase, SettingsBase

from .processor_loader import LoaderConfig, load_stage
from .processor_featurizer import FeaturizerConfig, featurizer_morgan_fingerprint
from .processor_consolidator import consolidate_after_loaded, consolidate_after_featurize
from .processor_clustering import ClusteringConfig, clustering, clustering_get_centroids, clustering_get_distances

logger = logging.getLogger(__name__)

class ProcessorConfig(ConfigBase):
    output_file: Path
    input_dir: Path
    loader_config: LoaderConfig = LoaderConfig()
    featurizer_config: FeaturizerConfig = FeaturizerConfig()
    clustering_config: ClusteringConfig = ClusteringConfig()

class ProcessorCli(CliRunnerMixin, SettingsBase):
    processor: ProcessorConfig

    def cmd(self):
        processor(config=self.processor)

def processor(config: ProcessorConfig):
    loaded = list()
    for file in config.input_dir.glob("*.sdf"):
        chembl_data = load_stage(file, config.loader_config)
        loaded.append(chembl_data)
    consolidated = consolidate_after_loaded(loaded)
    featurized = featurizer_morgan_fingerprint(consolidated, config.featurizer_config)
    clustered = clustering(featurized, config.clustering_config)
    resulting = consolidated.with_columns(
        (clustering_get_centroids(featurized, clustered)).alias("centroid"),
        (clustering_get_distances(featurized, clustered)).alias("distance"),
    )
    resulting.write_csv(config.output_file)
    logger.info("results saved under %s", config.output_file)

if __name__ == "__main__":
    ProcessorCli.run()