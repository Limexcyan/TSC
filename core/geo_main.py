import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from utils import *
from featurizer import *
from clusters_features import *
from analysis import *
from config import *
from geo_cluster_full import *
import os

path = './core/data/proper_sdf/molecule_tree'

if __name__ == "__main__":
    print("Trying to find existing data...")
    full_sdf_path = os.path.join(path, "full_sdfs.csv")
    full_3d_path = os.path.join(path, "full_3ds.csv")

    if os.path.isfile(full_sdf_path):
        print("Found full_sdfs.csv")
        full_df = pd.read_csv(full_sdf_path)
    else:
        print("Creating full_sdfs.csv from input SDFs...")
        full_df = consolidate_data(sdfs, userinputs=userinputs)
        export_data(full_df, full_sdf_path)
        print("Saved full_sdfs.csv")

    print("Trying to find existing data...")
    if os.path.isfile(full_3d_path):
        print("Found full_3ds.csv")
        df_3d = pd.read_csv(full_3d_path)
    else:
        print("Generating 3D coordinates...")
        coords_list = []
        smiles_list = []
        targets_list = []
        df = pd.read_csv(f"{path}/full_sdfs.csv")
        for i, row in df.iterrows():
            smiles = row['smiles']
            target = row['target']
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            coords = optimize_molecule_3d(mol)
            if coords is None:
                continue
            try:
                coords_list.append(coords)
                smiles_list.append(smiles)
                targets_list.append(target)
                print(f"Processed molecule {i}")
            except Exception as e:
                print(f"Skipping molecule {i} due to error: {e}")

        df_3d = pd.DataFrame(coords_list)
        df_3d.insert(0, "smiles", smiles_list)
        df_3d.insert(0, "target", targets_list)

        print("Saved full_3ds.csv")
        export_data(df_3d, full_3d_path)

    #out_data_full, out_data_processed = perform_analysis(datas, cluster_count, full_df)

    # targettocolor(out_data_full, cluster_count)
    #summary(out_data_full, 1e-20)