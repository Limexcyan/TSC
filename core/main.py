import pandas as pd

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools as pdt
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import rdFMCS as fmcs
from utils import *

if __name__ == "__main__":
    dropped_columns = ['CdId', 'Mol Weight', 'Formula', 'target_id', 'target_name', 'organism', 'target_type', 'ID',
                       'std_unit', 'std_type']

    data_path = f'data/proper_sdf'

    sdfs = ['data/proper_sdf/CHEMBL2147.sdf', 'data/proper_sdf/CHEMBL2343.sdf', 'data/proper_sdf/CHEMBL5794.sdf']
    userinputs = [[('target', 'threonine-protein kinase PIM1')], [('target', 'threonine-protein kinase Sgk1')],
                  [('target', 'threonine-protein kinase Sgk2')]]

    full_df = consolidate_data(sdfs, userinputs=userinputs)

    export_data(full_df, "data/full_sdfs.csv")