import logging
import pickle
from pathlib import Path

from ..common.cli import CliRunnerMixin
from ..common.settings import ConfigBase, SettingsBase
from .processor_clustering import (
    ClusteringConfig,
    clustering,
    clustering_get_centroids,
    clustering_get_distances,
)
from .processor_consolidator import (
    consolidate_after_loaded,
)
from .processor_featurizer import FeaturizerConfig, featurizer_morgan_fingerprint
from .processor_loader import LoaderConfig, load_stage
from .results.analyze import AnalyzeConfig, analyze

logger = logging.getLogger(__name__)

class ProcessorConfig(ConfigBase):
    output_file: Path
    input_files: list[Path]
    save_clustering: bool = True
    loader_config: LoaderConfig = LoaderConfig()
    featurizer_config: FeaturizerConfig = FeaturizerConfig()
    clustering_config: ClusteringConfig = ClusteringConfig()

class ProcessorCli(CliRunnerMixin, SettingsBase):
    processor: ProcessorConfig

    def cmd(self):
        processor(config=self.processor)

def processor(config: ProcessorConfig):
    loaded = []
    for file in config.input_files:
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
    if config.save_clustering:
        logger.info("saving clustering")
        with open(config.output_file.parent / (config.output_file.stem + ".bin.clustered"), "wb") as fo:
            pickle.dump(clustered, fo)
    logger.info("results saved under %s", config.output_file)
    
    logger.info("running analysys...")
    analyze(config=AnalyzeConfig(output_dir=config.output_file.parent,input_csv=config.output_file))

if __name__ == "__main__":
    ProcessorCli.run()