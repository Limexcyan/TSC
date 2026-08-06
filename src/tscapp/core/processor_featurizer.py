from ..common.settings import ConfigBase
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

import polars as pl

import logging

logger = logging.getLogger(__name__)

class FeaturizerConfig(ConfigBase):
    mfp_radius: int = 2
    mfp_size: int = 1024

def featurizer_morgan_fingerprint(
    dataframe: pl.DataFrame,
    config: FeaturizerConfig
) -> pl.DataFrame:
    logger.info("featurize r=%d s=%d", config.mfp_radius, config.mfp_size)
    fingerprint_labels = [f"mfp_{i+1}" for i in range(config.mfp_size)]
    preload_fingerprints = list()
    smiles_series = dataframe.get_column("smiles")
    fingerprint_generator = GetMorganGenerator(
        radius=config.mfp_radius,
        fpSize=config.mfp_size
    )
    mols_series = smiles_series.map_elements(MolFromSmiles)
    fingerprints_series = list(map(list,fingerprint_generator.GetFingerprints(mols_series.to_list())))
    #for smiles in smiles_series:
    #    molecule = MolFromSmiles(smiles)
    #    fingerprint = fingerprint_generator.GetFingerprint(molecule)
    #    preload_fingerprints.append(list(fingerprint))
    fingerprints = pl.DataFrame(fingerprints_series, schema=fingerprint_labels, orient="row")
    return fingerprints