from utils import *
from analysis import *
from geo_cluster_full import *
import os
from geo_cluster import optimize_molecule_3d
from config import *
from ast import literal_eval

if __name__ == "__main__":
    sdfs = molecules_sdfs
    userinputs = molecules_userinputs
    path = './data/proper_sdf/molecule_tree'

    print("Trying to find existing data...")
    full_sdf_path = os.path.join(path, "full_sdfs.csv")
    full_3d_path = os.path.join(path, "full_3ds.csv")

    if os.path.isfile(full_sdf_path):
        print("Found full_sdfs.csv")
        full_df = pd.read_csv(full_sdf_path)
    else:
        print("Creating full_sdfs.csv from input SDFs...")
        full_df = consolidate_data(sdfs, userinputs=userinputs, dropcols=dropped_columns)
        export_data(full_df, full_sdf_path)
        print("Saved full_sdfs.csv")

    print("Trying to find existing data...")
    if os.path.isfile(full_3d_path):
        print("Found full_3ds.csv")
        df_3d = pd.read_csv(full_3d_path, low_memory=False)
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

    path = "./data/proper_sdf/molecule_tree/full_3ds.csv"
    cluster_count = 20
    max_l = 10
    try:
        dataset = cluster_3d_string_csv(path, cluster_count=cluster_count, max_l=max_l, descriptor='usr')
        og_data = pd.read_csv(path, low_memory=False)[['target', 'smiles']][:1000]
        dat_sorted, processed = perform_analysis(dataset, cluster_count, og_data)
        compound_distance, compound_clusters, compound_norm = summary(dat_sorted, eps=0.1,
                                                                      userinputs=molecules_userinputs,
                                                                      cluster_count=cluster_count)
        if compound_distance is not None:
            print("Compound Distances:", compound_distance)
            print("Compound Clusters:", compound_clusters)
            print("Compound Normalized:", compound_norm)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}")

