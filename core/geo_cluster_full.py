import pandas as pd
import numpy as np
import math
import scipy.special
from numpy import linalg as la
from sklearn.cluster import BisectingKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import ast
from desc3d import *

# --- SH Clustering ---
def cluster_3d_string_csv(csv_path, cluster_count=10, max_l=3):
    df = pd.read_csv(csv_path, low_memory=False)[:100]
    features, names, smiles = [], [], []

    for idx, row in df.iterrows():
        try:
            coords_raw = [ast.literal_eval(c) for c in row[2:].dropna() if isinstance(c, str) and '[' in c and ']' in c]
            coords = [c for c in coords_raw if len(c) == 3 and all(isinstance(x, (int, float)) for x in c)]
            if len(coords) < 5:
                print(f"[WARN] Skipping molecule {row['target']}, too few atoms ({len(coords)})")
                continue
            coords = np.array(coords)
            sh_desc = pca_spherical_harmonics(coords, max_l=max_l)
            if sh_desc is None or np.any(np.isnan(sh_desc)):
                print(f"[WARN] Skipping molecule {row['target']}, invalid SH descriptors")
                continue
            features.append(sh_desc)
            names.append(row['target'])
            smiles.append(row['smiles'])
        except Exception as e:
            print(f"[ERROR] Failed to process molecule {row['target']}: {e}")

    if not features:
        raise ValueError("No valid molecules processed; check input data.")

    features = np.array(features)
    dataset = pd.DataFrame(features)
    dataset['target'] = names
    dataset['smiles'] = smiles
    return dataset


# --- Clustering and Visualization Functions ---
def cluster(dataset, n):
    features = dataset.drop(['target', 'smiles'], axis=1)
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')  # Impute NaN with mean
    features_imputed = imputer.fit_transform(features)
    features_scaled = scaler.fit_transform(features_imputed)

    clustering = BisectingKMeans(n_clusters=n, random_state=42)
    clusterdist = clustering.fit_transform(features_scaled)
    clusterid = clustering.labels_
    clusterdist = clusterdist[np.arange(len(features_scaled)), clusterid]

    processed_data = dataset.copy()
    processed_data['centroid'] = clusterid
    processed_data['distance'] = clusterdist

    pca = PCA(n_components=3)
    pcdata3d = pd.DataFrame(pca.fit_transform(features_scaled))
    pcdata3d.columns = ['pc1', 'pc2', 'pc3']

    sizes = np.power(10, 6) / np.power(processed_data['distance'], 7)
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(projection='3d')
    ax.azim = 35
    ax.elev = 35
    sc = ax.scatter(pcdata3d['pc1'], pcdata3d['pc2'], pcdata3d['pc3'], cmap=plt.cm.gist_rainbow,
                    c=processed_data['centroid'], s=sizes)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.savefig('results/cluster_3d_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    return clustering, processed_data, pcdata3d


def showcluster(cid, dat, pcdat):
    cdata = dat[dat['centroid'] == cid]['distance']
    pickcluster = pcdat.loc[dat['centroid'] == cid]
    sizes = np.power(10, 6) / np.power(cdata, 7)
    print(sizes.shape)
    print(pickcluster.shape)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.azim = 35
    ax.elev = 35
    sc = ax.scatter(pickcluster['pc1'], pickcluster['pc2'], pickcluster['pc3'], cmap=plt.cm.viridis, s=sizes)
    plt.legend(*sc.legend_elements(), bbox_to_anchor=(1.05, 1), loc=2)
    plt.savefig(f'results/cluster_{cid}_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


def centroidanalysis(dat, ccount):
    dsts = []
    count = []
    for i in range(ccount):
        centroid = dat[dat['centroid'] == i]
        dsts.append(np.sum(centroid['distance']))
        count.append(len(centroid))
    plt.scatter(count, dsts)
    plt.title('Centroid Count vs Total Distance')
    plt.xlabel('Count')
    plt.ylabel('Distance')
    plt.savefig('results/centroid_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def cal_centroidanalysis(dat, ccount):
    dsts = []
    count = []
    for i in range(ccount):
        centroid = dat[dat['centroid'] == i]
        dsts.append(np.sum(centroid['distance']))
        count.append(len(centroid))
    return dsts, count


def ccount(dat, cs):
    count = []
    for i in range(cs):
        centroid = dat[dat['centroid'] == i]
        count.append(len(centroid))
    plt.bar(range(cs), count)
    plt.title('Centroid Count')
    plt.xlabel('Centroid')
    plt.ylabel('Amount')
    plt.savefig('results/centroid_count.png', dpi=300, bbox_inches='tight')
    plt.close()


def cal_ccount(dat, cs):
    count = []
    for i in range(cs):
        centroid = dat[dat['centroid'] == i]
        count.append(len(centroid))
    return count


def perform_analysis(dat, n, og):
    cl, processed, pcadata = cluster(dat, n)
    ccount(processed, n)
    centroidanalysis(processed, n)
    dat_sorted = processed.copy()
    dat_sorted['smiles'] = og['smiles']
    dat_sorted['target'] = og['target']
    dat_sorted = dat_sorted.sort_values(by=['distance'])
    prevd, prevc = cal_centroidanalysis(processed, n)
    return dat_sorted, processed


def summary(a1, eps, userinputs=None, cluster_count=5):
    if userinputs is None:
        print("No userinputs provided; skipping summary.")
        return None, None, None

    compound_distance = []
    compound_clusters = []
    compound_norm = []
    aggregate_compound_clusters = [[0 for _ in range(cluster_count)] for _ in range(len(userinputs))]

    for i, ui in enumerate(userinputs):
        target = ui[0][1] if isinstance(ui, (list, tuple)) and len(ui) > 0 and isinstance(ui[0], (list, tuple)) and len(ui[0]) > 1 else ui
        if not isinstance(target, str):
            print(f"[WARN] Invalid target format for userinput {ui}; skipping.")
            continue
        subset = a1[a1['target'] == target]
        if subset.empty:
            print(f"[WARN] No data for target {target}; skipping.")
            continue
        tempd, tempc = cal_centroidanalysis(subset, cluster_count)
        compound_distance.append(tempd)
        compound_clusters.append(tempc)
        total_count = np.sum(tempc)
        compound_norm.append(tempc / total_count if total_count > 0 else np.zeros_like(tempc))
        for j in range(i, len(userinputs)):
            aggregate_compound_clusters[j] = np.add(aggregate_compound_clusters[j], tempc)

    # Create stacked bar plot with distinct colors
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(userinputs)))  # Distinct colors for each target
    for j in range(len(userinputs)):
        target = userinputs[len(userinputs) - j - 1]
        target = target[0][1] if isinstance(target, (list, tuple)) and len(target) > 0 and isinstance(target[0], (list, tuple)) and len(target[0]) > 1 else target
        plt.bar(range(cluster_count), aggregate_compound_clusters[len(userinputs) - j - 1],
                label=target, alpha=0.7, color=colors[len(userinputs) - j - 1],
                bottom=np.sum(aggregate_compound_clusters[:len(userinputs) - j - 1], axis=0),
                width=0.8/len(userinputs), align='edge', edgecolor='black')

    plt.xlabel('Cluster')
    plt.ylabel('Number of Molecules')
    plt.title('Molecule Distribution Across Clusters by Target')
    plt.xticks(range(cluster_count), [f'Cluster {i}' for i in range(cluster_count)])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Targets')
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/summary_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    return compound_distance, compound_clusters, compound_norm


# --- User Inputs ---
molecules_userinputs = [
    [('target', 'chembl2014'), ('seq', """MPATAPSCPSGSRSPSWGSTWPCVSEGSWGTALSCTSSSGRLGPKVPVWHTKMKTATNIY
IFNLALADTLVLLTLPFQGTDILLGFWPFGNALCKTVIAIDYYNMFTSTFTLTAMSVDRY
VAICHPIRALDVRTSSKAQAVNVAIWALASVVGVPVAIMGSAQVEDEEIECLVEIPTPQD
YWGPVFAICIFLFSFIVPVLVISVCYSLMIRRLRGVRLLSGSREKDRNLRRITRLVLVVV
AVFVGCWTPVQVFVLAQGLGVQPSSETAVAILRFCTALGYVNSCLNPILYAFLDENFKAC
FRKFCCASALRRDVQVSDRVRSIAKDVALACKTSETVPRPA""")],
    [('target', 'deltaopioid'), ('seq', """MEPAPSAGAELQPPLFANASDAYPSACPSAGANASGPPGARSASSLALAIAITALYSAVC
AVGLLGNVLVMFGIVRYTKMKTATNIYIFNLALADALATSTLPFQSAKYLMETWPFGELL
CKAVLSIDYYNMFTSIFTLTMMSVDRYIAVCHPVKALDFRTPAKAKLINICIWVLASGVG
VPIMVMAVTRPRDGAVVCMLQFPSPSWYWDTVTKICVFLFAFVVPILIITVCYGLMLLRL
RSVRLLSGSKEKDRSLRRITRMVLVVVGAFVVCWAPIHIFVIVWTLVDIDRRDPLVVAAL
HLCIALGYANSSLNPVLYAFLDENFKRCFRQLCRKPCGRPDPSSFSRAREATARERVTAC
TPSDGPGGGAAA""")],
    [('target', 'kappaopioid'), ('seq', """MDSPIQIFRGEPGPTCAPSACLPPNSSAWFPGWAEPDSNGSAGSEDAQLEPAHISPAIPV
IITAVYSVVFVVGLVGNSLVMFVIIRYTKMKTATNIYIFNLALADALVTTTMPFQSTVYL
MNSWPFGDVLCKIVISIDYYNMFTSIFTLTMMSVDRYIAVCHPVKALDFRTPLKAKIINI
CIWLLSSSVGISAIVLGGTKVREDVDVIECSLQFPDDDYSWWDLFMKICVFIFAFVIPVL
IIIVCYTLMILRLKSVRLLSGSREKDRNLRRITRLVLVVVAVFVVCWTPIHIFILVEALG
STSHSTAALSSYYFCIALGYTNSSLNPILYAFLDENFKRCFRDFCFPLKMRMERQSTSRV
RNTVQDPAYLRDIDGMNKPV""")],
    [('target', 'muopioid'), ('seq', """MDSSAAPTNASNCTDALAYSSCSPAPSPGSWVNLSHLDGNLSDPCGPNRTDLGGRDSLCP
PTGSPSMITAITIMALYSIVCVVGLFGNFLVMYVIVRYTKMKTATNIYIFNLALADALAT
STLPFQSVNYLMGTWPFGTILCKIVISIDYYNMFTSIFTLCTMSVDRYIAVCHPVKALDF
RTPRNAKIINVCNWILSSAIGLPVMFMATTKYRQGSIDCTLTFSHPTWYWENLLKICVFI
FAFIMPVLIITVCYGLMILRLKSVRMLSGSKEKDRNLRRITRMVLVVVAVFIVCWTPIHI
YVIIKALVTIPETTFQTVSWHFCIALGYTNSCLNPVLYAFLDENFKRCFREFCIPTSSNI
EQQNSTRIRQNTRDHPSTANTVDRTNHQLENLEAETAPLP""")],
    [('target', 'sigmaopioid'), ('seq', """MQWAVGRRWAWAALLLAVAAVLTQVVWLWLGTQSFVFQREEIAQLARQYAGLDHELAFSR
LIVELRRLHPGHVLPDEELQWVFVNAGGWMGAMCLLHASLSEYVLLFGTALGSRGHSGRY
WAEISDTIISGTFHQWREGTTKSEVFYPGETVVHGPGEATAVEWGPNTWMVEYGRGVIPS
TLAFALADTVFSTQDFLTLFYTLRSYARGLRLELTTYLFGQDP""")]
]

# --- Example Usage ---
if __name__ == "__main__":
    path = "./data/proper_sdf/receptors_tree/full_3ds.csv"
    cluster_count = 10
    max_l = 5
    try:
        dataset = cluster_3d_string_csv(path, cluster_count=cluster_count, max_l=max_l)
        og_data = pd.read_csv(path, low_memory=False)[['target', 'smiles']][:100]
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
