from scipy.special import rel_entr, kl_div
from scipy.stats import linregress
from Bio import pairwise2
from clusters_features import *
from config import *

def kl_divergence(p, q, eps=1e-10):
    return np.sum(kl_div(p + eps, q + eps))

def bhattacharyya_distance(p, q):
    return -np.log(np.sum(np.sqrt(p * q)))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def scatter_js_vs_alignment(compound_norm, userinputs, eps=1e-10):
    js_values = []
    align_scores = []
    labels = []

    for i in range(len(userinputs) - 1):
        for j in range(i + 1, len(userinputs)):
            # JS divergence
            kl_i_j = kl_divergence(compound_norm[i], (compound_norm[i] + compound_norm[j]) / 2, eps=eps)
            kl_j_i = kl_divergence(compound_norm[j], (compound_norm[j] + compound_norm[i]) / 2, eps=eps)
            js = (kl_i_j + kl_j_i) / 2

            # Alignment score
            alignments = pairwise2.align.globalxx(userinputs[i][1][1], userinputs[j][1][1])
            score = alignments[0][2]

            js_values.append(js)
            align_scores.append(score)
            labels.append((userinputs[i][0][1], userinputs[j][0][1]))

    # Scatter plot
    plt.figure(figsize=(7, 5))
    plt.scatter(align_scores, js_values, alpha=0.7)

    # Regression line
    slope, intercept, r_value, p_value, std_err = linregress(align_scores, js_values)
    x_line = np.linspace(min(align_scores), max(align_scores), 100)
    plt.plot(x_line, intercept + slope * x_line, "r--", label=f"fit (R²={r_value**2:.3f})")

    plt.xlabel("Alignment Score")
    plt.ylabel("Jensen–Shannon Divergence")
    plt.title("JS Divergence vs Alignment Score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return js_values, align_scores, (slope, intercept, r_value, p_value, std_err)


def regression_js_vs_alignment(compound_norm, userinputs, eps=1e-10):
    js_values = []
    align_scores = []
    labels = []

    for i in range(len(userinputs) - 1):
        for j in range(i + 1, len(userinputs)):
            # JS divergence
            kl_i_j = kl_divergence(compound_norm[i], (compound_norm[i] + compound_norm[j]) / 2, eps=eps)
            kl_j_i = kl_divergence(compound_norm[j], (compound_norm[j] + compound_norm[i]) / 2, eps=eps)
            js = (kl_i_j + kl_j_i) / 2

            # Alignment score
            alignments = pairwise2.align.globalxx(userinputs[i][1][1], userinputs[j][1][1])
            score = alignments[0][2]

            js_values.append(js)
            align_scores.append(score)
            labels.append((userinputs[i][0][1], userinputs[j][0][1]))

    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(align_scores, js_values)

    # Scatter + regression line
    plt.figure(figsize=(7, 5))
    plt.scatter(align_scores, js_values, alpha=0.7, label="Compound pairs")
    x_line = np.linspace(min(align_scores), max(align_scores), 100)
    plt.plot(x_line, intercept + slope * x_line, "r--", label=f"fit (R²={r_value**2:.3f})")

    plt.xlabel("Alignment Score")
    plt.ylabel("Jensen–Shannon Divergence")
    plt.title("Regression: JS divergence vs Alignment score")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        "slope": slope,
        "intercept": intercept,
        "r_value": r_value,
        "p_value": p_value,
        "std_err": std_err,
        "pairs": labels
    }


def correlation_plot(compound_norm, labels, metric="js", eps=1e-10):
    n = len(compound_norm)
    corr_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if metric == "js":
                kl_i_j = kl_divergence(compound_norm[i], (compound_norm[i] + compound_norm[j])/2, eps=eps)
                kl_j_i = kl_divergence(compound_norm[j], (compound_norm[j] + compound_norm[i])/2, eps=eps)
                corr_matrix[i, j] = (kl_i_j + kl_j_i) / 2
            elif metric == "bhat":
                corr_matrix[i, j] = bhattacharyya_distance(compound_norm[i], compound_norm[j])
            else:
                raise ValueError("Unknown metric. Use 'js' or 'bhat'.")

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, xticklabels=labels, yticklabels=labels, annot=True, cmap="viridis", fmt=".3f")
    plt.title(f"Compound {metric.upper()} Distance Matrix")
    plt.tight_layout()
    plt.show()
    return corr_matrix

def summary(a1, userinputs, eps=1e-10):

    compound_distance = []
    compound_clusters = []
    aggregate_compound_clusters = [[0 for _ in range(cluster_count)] for _ in range(len(userinputs))]
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

    labels = [u[0][1] for u in userinputs]
    correlation_plot(compound_norm, labels, metric="js")  # Jensen–Shannon
    correlation_plot(compound_norm, labels, metric="bhat")  # Bhattacharyya


    scatter_js_vs_alignment(compound_norm,userinputs, eps)
