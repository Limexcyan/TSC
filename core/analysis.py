from scipy.special import rel_entr, kl_div
from Bio import pairwise2
from clusters_features import *
from config import *
from main import userinputs


def kl_divergence(p, q, eps=1e-10):
    return np.sum(kl_div(p + eps, q + eps))

def bhattacharyya_distance(p, q):
    return -np.log(np.sum(np.sqrt(p * q)))

def summary(a1, eps=1e-10):
    compound_distance = []
    compound_clusters = []
    aggregate_compound_clusters = [[0 for j in range(cluster_count)] for i in range(len(userinputs))]
    compound_norm = []
    for i in range(len(userinputs)):
        tempd, tempc = cal_centroidanalysis(a1[a1['target'] == userinputs[i][0][1]],cluster_count)
        compound_distance.append(tempd)
        compound_norm.append(tempc / np.sum(tempc))
        compound_clusters.append(tempc)
        for j in range(i,len(userinputs)):
            aggregate_compound_clusters[j] = np.add(aggregate_compound_clusters[j], tempc)

    for j in range(len(userinputs)):
        plt.bar(range(cluster_count), aggregate_compound_clusters[len(userinputs) - j - 1], label=userinputs[len(userinputs) - j - 1][0][1])

    plt.xlabel("centroid")
    plt.ylabel("amount")
    plt.legend()
    plt.show()


    for i in range(len(userinputs)-1):
        for j in range(i+1,len(userinputs)):
            kl_i_j = kl_divergence(compound_norm[i], (compound_norm[i] + compound_norm[j])/2, eps=eps)
            kl_j_i = kl_divergence(compound_norm[j], (compound_norm[j] + compound_norm[i])/2, eps=eps)
            bhat = bhattacharyya_distance(compound_norm[i],compound_norm[j])
            print(f"JS({userinputs[i][0][1]} || {userinputs[j][0][1]}): {(kl_i_j + kl_j_i) / 2}")
            print(f"Bhattacharyya({userinputs[i][0][1]}, {userinputs[j][0][1]}): {bhat}")

    for i in range(len(userinputs) - 1):
        for j in range(i+1,len(userinputs)):
            alignments_i_j = pairwise2.align.globalxx(userinputs[i][1][1],userinputs[j][1][1])
            score_i_j = alignments_i_j[0][2]
            print(f"Alignment Score ({userinputs[i][0][1]} vs {userinputs[j][0][1]}): {score_i_j}")