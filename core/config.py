"""
This file stores configuration of functions
"""

dropped_columns = [
    'CdId',
    'Mol Weight',
    'Formula',
    'target_id',
    'target_name',
    'organism',
    'target_type',
    'ID',
    'std_unit',
    'std_type'
]

sdfs = [
    'data/proper_sdf/CHEMBL2147.sdf',
    'data/proper_sdf/CHEMBL2343.sdf',
    'data/proper_sdf/CHEMBL5794.sdf'
]

data_path = f'data/proper_sdf'

userinputs = [
    [('target', 'threonine-protein kinase PIM1')],
    [('target', 'threonine-protein kinase Sgk1')],
    [('target', 'threonine-protein kinase Sgk2')]
]

feature_count = 1024
quantile_count = 100