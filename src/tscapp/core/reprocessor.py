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
    consolidate_after_featurize
)
from .processor_featurizer import (
    FeaturizerConfig,
    featurizer_morgan_fingerprint,
)
from .processor_loader import LoaderConfig, load_stage, load_csv
from .results.analyze import AnalyzeConfig, analyze

logger = logging.getLogger(__name__)

class ReProcessorConfig(ConfigBase):
    output_file: Path
    input_files: list[Path]
    resource_path: Path
    save_clustering: bool = True
    loader_config: LoaderConfig = LoaderConfig()
    featurizer_config: FeaturizerConfig = FeaturizerConfig()
    clustering_config: ClusteringConfig = ClusteringConfig()

class ReProcessorCli(CliRunnerMixin, SettingsBase):
    processor: ReProcessorConfig

    def cmd(self):
        reprocessor(config=self.processor)

def reprocessor(config: ReProcessorConfig):
    logger.info("%s",config.resource_path)
    csv_file = config.resource_path.glob("*.csv").__next__()
    clu_file = config.resource_path.glob("*.bin.clustered").__next__()
    existing_data = load_csv(csv_file)
    with open(clu_file, "rb") as fo:
        clustered = pickle.load(fo)
    loaded = []
    for file in config.input_files:
        chembl_data = load_stage(file, config.loader_config)
        featurized = featurizer_morgan_fingerprint(chembl_data, config.featurizer_config)
        loaded.append(consolidate_after_featurize(chembl_data,featurized))
    consolidated = consolidate_after_loaded(loaded)
    #for file in config.input_files:
    #    chembl_data = load_stage(file, config.loader_config)
    #    loaded.append(chembl_data)
    #consolidated = consolidate_after_loaded(loaded)
    #featurized = featurizer_morgan_fingerprint(consolidated, config.featurizer_config)
    featurized = consolidated.select(r"^mfp.*$")
    resulting = consolidated.with_columns(
        (clustering_get_centroids(featurized, clustered)).alias("centroid"),
        (clustering_get_distances(featurized, clustered)).alias("distance"),
    ).drop("^mfp.*$")
    consolidate_after_loaded([existing_data, resulting]).write_csv(config.output_file)
    if config.save_clustering:
        logger.info("saving clustering")
        with open(config.output_file.parent / (config.output_file.stem + ".bin.clustered"), "wb") as fo:
            pickle.dump(clustered, fo)
    logger.info("results saved under %s", config.output_file)
    
    logger.info("running analysys...")
    analyze(config=AnalyzeConfig(output_dir=config.output_file.parent,input_csv=config.output_file))

if __name__ == "__main__":
    ReProcessorCli.run()