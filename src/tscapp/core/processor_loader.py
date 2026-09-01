from enum import Enum, auto
from ..common.settings import ConfigBase
from typing import List
from pathlib import Path
from rdkit.Chem.PandasTools import LoadSDF

import polars as pl

import logging

logger = logging.getLogger(__name__)

class LoaderStage(Enum):
    NONE = auto()
    LOADED = auto()
    CONSOLIDATED = auto()
    FEATURIZED = auto()
    CLUSTERED = auto()


class LoaderConfig(ConfigBase):
    smiles_col: str = "smiles"
    keep_cols: List[str] | str = ["chembl_id","smiles"]
    load_stage: LoaderStage = LoaderStage.NONE

def load_chembl_sdf(
    file: Path,
    config: LoaderConfig
) -> pl.DataFrame:
    logger.info("loading %s columns %s", file, ",".join(config.keep_cols))
    chembl_sdf = pl.from_pandas(
        LoadSDF(
            filename=file.resolve(),
            molColName=None,
            smilesName=config.smiles_col,
            embedProps=True
        )
    )
    return chembl_sdf.select(config.keep_cols)

def load_csv(
    file: Path,
) -> pl.DataFrame:
    logger.info("loading csv %s", file)
    return pl.read_csv(file)

def load_stage(
    file: Path,
    config: LoaderConfig,
) -> pl.DataFrame:
    match config.load_stage:
        case LoaderStage.NONE:
            return load_chembl_sdf(file, config)
        case LoaderStage.LOADED:
            raise NotImplementedError
        case LoaderStage.CONSOLIDATED:
            raise NotImplementedError
        case LoaderStage.FEATURIZED:
            raise NotImplementedError
        case LoaderStage.CLUSTERED:
            raise NotImplementedError
        case _:
            raise ValueError(f"Loader stage {config.load_stage} does not exist")
