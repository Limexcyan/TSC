from pathlib import Path
import logging
from ..common.cli import CliRunnerMixin
from ..common.settings import ConfigBase, SettingsBase

from .processor_loader import LoaderConfig, load_chembl_sdf
from .processor_featurizer import FeaturizerConfig, featurizer_morgan_fingerprint
from .processor_consolidator import consolidate_after_loaded, consolidate_after_featurize

logger = logging.getLogger(__name__)

class ProcessorConfig(ConfigBase):
    output_file: Path
    input_dir: Path
    loader_config: LoaderConfig = LoaderConfig()
    featurizer_config: FeaturizerConfig = FeaturizerConfig()

class ProcessorCli(CliRunnerMixin, SettingsBase):
    processor: ProcessorConfig

    def cmd(self):
        processor(config=self.processor)

def processor(config: ProcessorConfig):
    loaded = list()
    for file in config.input_dir.glob("*.sdf"):
        chembl_data = load_chembl_sdf(file, config.loader_config)
        loaded.append(chembl_data)
    consolidated = consolidate_after_loaded(loaded)
    featurized = featurizer_morgan_fingerprint(consolidated, config.featurizer_config)
    processed = consolidate_after_featurize(consolidated, featurized)
    del consolidated
    del featurized
    processed.write_csv(config.output_file)

if __name__ == "__main__":
    ProcessorCli.run()