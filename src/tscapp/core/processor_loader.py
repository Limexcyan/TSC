from ..common.settings import ConfigBase
from typing import List
from pathlib import Path
from rdkit.Chem.PandasTools import LoadSDF

import polars as pl

import logging

logger = logging.getLogger(__name__)

class LoaderConfig(ConfigBase):
    smiles_col: str = "smiles"
    keep_cols: List[str] | str = ["chembl_id","smiles"]

def load_chembl_sdf(
    file: Path,
    config: LoaderConfig
) -> pl.DataFrame:
    logger.info("loading %s columns %s", file, ",".join(config.keep_cols))
    chembl_sdf = pl.from_pandas(
        LoadSDF(
            filename=file,
            molColName=None,
            smilesName=config.smiles_col,
            embedProps=True
        )
    )
    return chembl_sdf.select(config.keep_cols)