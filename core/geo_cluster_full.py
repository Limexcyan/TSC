import ast
from desc3d import *
from core.utils import *
from core.analysis import *

def cluster_3d_string_csv(csv_path, max_l=3, descriptor='sh'):
    df = pd.read_csv(csv_path, low_memory=False)
    features, names, smiles = [], [], []
    no_col = max_l ** 2
    for idx, row in df.iterrows():
        try:
            coords_raw = [ast.literal_eval(c) for c in row[2:].dropna() if isinstance(c, str) and '[' in c and ']' in c]
            coords = [c for c in coords_raw if len(c) == 3 and all(isinstance(x, (int, float)) for x in c)]
            print(idx)
            coords = np.array(coords)
            desc = np.array(coords)
            if descriptor == 'sh':
                desc = pca_spherical_harmonics(coords, max_l=max_l)
            elif descriptor == 'sh-mse':
                desc = pca_spherical_harmonics_mse(coords, max_l=max_l)
            elif descriptor == 'rif':
                desc = rif_sh_descriptor(coords, harmonics=max_l)
                no_col = max_l
            elif descriptor == 'bdc':
                desc = bend_deformed_cylinder_descriptors(coords)
                no_col = 8
            elif descriptor == 'chd':
                da = 1
                dc = 2
                desc = cylindrical_harmonics_descriptor(coords, da=da, dc=dc)
                no_col = (da + 1)*(2 * dc + 1)
            elif descriptor == 'usr':
                desc = ultrafast_shape_recognition_descriptors(coords)
                no_col = 12
            elif desc is None or np.any(np.isnan(desc)):
                print(f"[WARN] Skipping molecule {row['target']}, invalid SH descriptors")
                continue
            features.append(desc)
            # names.append(row['target'])
            # smiles.append(row['smiles'])
        except Exception as e:
            print(f"[ERROR] Failed to process molecule {row['target']}: {e}")
    if not features:
        raise ValueError("No valid molecules processed; check input data.")

    #dataset['target'] = names
    #dataset['smiles'] = smiles

    labels = ['mfp_' + str(n) for n in np.arange(no_col)]
    features = np.array(features)
    dataset = pd.DataFrame(features, columns=labels)
    return dataset

if __name__ == "__main__":
    path = "./data/proper_sdf/molecule_tree/full_3ds.csv"
    userinputs = molecules_userinputs

    cluster_count = 20
    max_l = 4

    #dataset = cluster_3d_string_csv(path, descriptor='chd', max_l=10)
    #export_data(dataset,"./data/proper_sdf/molecule_tree/chd.csv")

    desc_df = pd.read_csv("./data/proper_sdf/molecule_tree/chd.csv")
    #desc_df = desc_df.drop(['mfp_1', 'mfp_2', 'mfp_3', 'mfp_4', 'mfp_5', 'mfp_6', 'mfp_7'], axis=1)

    full_df = pd.read_csv("./data/proper_sdf/molecule_tree/full_sdfs.csv", index_col=False)
    out_data_full, _ = perform_analysis(desc_df, cluster_count, full_df)

    print("SUMMARY:")
    summary(out_data_full, userinputs, 1e-20)