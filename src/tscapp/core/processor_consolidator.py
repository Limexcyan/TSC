import logging

import polars as pl

logger = logging.getLogger(__name__)

def consolidate_after_featurize(featurizer_input: pl.DataFrame, featurizer_output: pl.DataFrame) -> pl.DataFrame:
    logger.info("h-stacking featurizer output with featurizer input")
    return pl.concat([featurizer_input, featurizer_output], how="horizontal")

def consolidate_after_loaded(all_outputs: list[pl.DataFrame]) -> pl.DataFrame:
    logger.info("v-stacking %d consolidation outputs", len(all_outputs))
    return pl.concat(all_outputs, how="vertical")