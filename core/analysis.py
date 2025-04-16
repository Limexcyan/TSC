import pandas as pd

import numpy as np

from matplotlib import pyplot as plt


from scipy.special import rel_entr, kl_div

from Bio import pairwise2

from clusters_features import *


def targettocolor(dat, n):
    pim1d, pim1c = cal_centroidanalysis(dat[dat['target'] == 'threonine-protein kinase PIM1'],n)
    sgk1d, sgk1c = cal_centroidanalysis(dat[dat['target'] == 'threonine-protein kinase Sgk1'],n)
    sgk2d, sgk2c = cal_centroidanalysis(dat[dat['target'] == 'threonine-protein kinase Sgk2'],n)
    plt.scatter(pim1d, pim1c, marker='^', c='red')
    plt.scatter(sgk1d, sgk1c, marker='o', c='blue')
    plt.scatter(sgk2d, sgk2c, marker='x', c='green')
    plt.title("centroid count")
    plt.xlabel("centroid")
    plt.ylabel("amount")
    plt.show()
    plt.bar(range(n),pim1c,color='red')
    plt.title("centroid count for pim1")
    plt.xlabel("centroid")
    plt.ylabel("amount")
    plt.show()
    plt.bar(range(n),sgk1c,color='blue')
    plt.title("centroid count for sgk1")
    plt.xlabel("centroid")
    plt.ylabel("amount")
    plt.show()
    plt.bar(range(n),sgk2c,color='green')
    plt.title("centroid count for sgk2")
    plt.xlabel("centroid")
    plt.ylabel("amount")
    plt.show()


def summary(a1, eps):
    pim1d, pim1c = cal_centroidanalysis(a1[a1['target'] == 'threonine-protein kinase PIM1'], 50)
    sgk1d, sgk1c = cal_centroidanalysis(a1[a1['target'] == 'threonine-protein kinase Sgk1'], 50)
    sgk2d, sgk2c = cal_centroidanalysis(a1[a1['target'] == 'threonine-protein kinase Sgk2'], 50)
    plt.bar(range(50), pim1c, color='red')
    plt.bar(range(50), sgk1c, color='blue')
    plt.bar(range(50), sgk2c, color='green')
    plt.show()

    pim1c_norm = pim1c / np.sum(pim1c)
    sgk1c_norm = sgk1c / np.sum(sgk1c)
    sgk2c_norm = sgk2c / np.sum(sgk2c)

    def kl_divergence(p, q, eps=1e-20):
        return np.sum(kl_div(p + eps, q + eps))

    def bhattacharyya_distance(p, q):
        return -np.log(np.sum(np.sqrt(p * q)))

    kl_pim1_sgk1 = kl_divergence(pim1c_norm, sgk1c_norm)
    kl_pim1_sgk2 = kl_divergence(pim1c_norm, sgk2c_norm)
    kl_sgk1_sgk2 = kl_divergence(sgk1c_norm, sgk2c_norm)

    kl_sgk1_pim1 = kl_divergence(sgk1c_norm, pim1c_norm)
    kl_sgk2_pim1 = kl_divergence(sgk2c_norm, pim1c_norm)
    kl_sgk2_sgk1 = kl_divergence(sgk2c_norm, sgk1c_norm)

    bhat_pim1_sgk1 = bhattacharyya_distance(pim1c_norm, sgk1c_norm)
    bhat_pim1_sgk2 = bhattacharyya_distance(pim1c_norm, sgk2c_norm)
    bhat_sgk1_sgk2 = bhattacharyya_distance(sgk1c_norm, sgk2c_norm)

    print(f"KL(PIM1 || SGK1): {kl_pim1_sgk1}")
    print(f"KL(PIM1 || SGK2): {kl_pim1_sgk2}")
    print(f"KL(SGK1 || SGK2): {kl_sgk1_sgk2}")

    print("-----------------------------------------")

    print(f"KL(SGK1 || PIM1): {kl_sgk1_pim1}")
    print(f"KL(SGK2 || PIM1): {kl_sgk2_pim1}")
    print(f"KL(SGK2 || SGK1): {kl_sgk2_sgk1}")

    print("-----------------------------------------")

    print(f"JS(SGK1 || PIM1): {(kl_sgk1_pim1 + kl_pim1_sgk1) / 2}")
    print(f"JS(SGK2 || PIM1): {(kl_sgk2_pim1 + kl_pim1_sgk2) / 2}")
    print(f"JS(SGK2 || SGK1): {(kl_sgk2_sgk1 + kl_sgk1_sgk2) / 2}")

    print("-----------------------------------------")

    print(f"Bhattacharyya(PIM1, SGK1): {bhat_pim1_sgk1}")
    print(f"Bhattacharyya(PIM1, SGK2): {bhat_pim1_sgk2}")
    print(f"Bhattacharyya(SGK1, SGK2): {bhat_sgk1_sgk2}")

    sgk1_seq = """MVNKDMNGFPVKKCSAFQFFKKRVRRWIKSPMVSVDKHQSPSLKYTGSSMVHIPPGEPDF
    ESSLCQTCLGEHAFQRGVLPQENESCSWETQSGCEVREPCNHANILTKPDPRTFWTNDDP
    AFMKQRRMGLNDFIQKIANNSYACKHPEVQSILKISQPQEPELMNANPSPPPSPSQQINL
    GPSSNPHAKPSDFHFLKVIGKGSFGKVLLARHKAEEVFYAVKVLQKKAILKKKEEKHIMS
    ERNVLLKNVKHPFLVGLHFSFQTADKLYFVLDYINGGELFYHLQRERCFLEPRARFYAAE
    IASALGYLHSLNIVYRDLKPENILLDSQGHIVLTDFGLCKENIEHNSTTSTFCGTPEYLA
    PEVLHKQPYDRTVDWWCLGAVLYEMLYGLPPFYSRNTAEMYDNILNKPLQLKPNITNSAR
    HLLEGLLQKDRTKRLGAKDDFMEIKSHVFFSLINWDDLINKKITPPFNPNVSGPNDLRHF
    DPEFTEEPVPNSIGKSPDSVLVTASVKEAAEAFLGFSYAPPTDSFL"""

    sgk2_seq = """MNSSPAGTPSPQPSRANGNINLGPSANPNAQPTDFDFLKVIGKGNYGKVLLAKRKSDGAF
    YAVKVLQKKSILKKKEQSHIMAERSVLLKNVRHPFLVGLRYSFQTPEKLYFVLDYVNGGE
    LFFHLQRERRFLEPRARFYAAEVASAIGYLHSLNIIYRDLKPENILLDCQGHVVLTDFGL
    CKEGVEPEDTTSTFCGTPEYLAPEVLRKEPYDRAVDWWCLGAVLYEMLHGLPPFYSQDVS
    QMYENILHQPLQIPGGRTVAACDLLQSLLHKDQRQRLGSKADFLEIKNHVFFSPINWDDL
    YHKRLTPPFNPNVTGPADLKHFDPEFTQEAVSKSIGCTPDTVASSSGASSAFLGFSYAPE
    DDDILDC"""

    pim1_seq = """MLLSKINSLAHLRAAPCNDLHATKLAPGKEKEPLESQYQVGPLLGSGGFGSVYSGIRVSD
    NLPVAIKHVEKDRISDWGELPNGTRVPMEVVLLKKVSSGFSGVIRLLDWFERPDSFVLIL
    ERPEPVQDLFDFITERGALQEELARSFFWQVLEAVRHCHNCGVLHRDIKDENILIDLNRG
    ELKLIDFGSGALLKDTVYTDFDGTRVYSPPEWIRYHRYHGRSAAVWSLGILLYDMVCGDI
    PFEHDEEIIRGQVFFRQRVSSECQHLIRWCLALRPSDRPTFEEIQNHPWMQDVLLPQETA
    EIHLHSLSPGPSK"""

    # Global alignment with simple match/mismatch scoring
    alignments_sgk1_sgk2 = pairwise2.align.globalxx(sgk1_seq, sgk2_seq)
    alignments_sgk1_pim1 = pairwise2.align.globalxx(sgk1_seq, pim1_seq)
    alignments_sgk2_pim1 = pairwise2.align.globalxx(sgk2_seq, pim1_seq)

    score_sgk1_sgk2 = alignments_sgk1_sgk2[0][2]  # Alignment score
    score_sgk1_pim1 = alignments_sgk1_pim1[0][2]
    score_sgk2_pim1 = alignments_sgk2_pim1[0][2]

    distance_sgk1_sgk2 = len(sgk1_seq) - score_sgk1_sgk2
    distance_sgk1_pim1 = len(sgk2_seq) - score_sgk1_pim1
    distance_sgk2_pim1 = len(pim1_seq) - score_sgk2_pim1

    print(f"Alignment Score (SGK1 and SGK2): {score_sgk1_sgk2}")
    print(f"Alignment Score (SGK1 and PIM1): {score_sgk1_pim1}")
    print(f"Alignment Score (SGK2 and PIM1): {score_sgk2_pim1}")

    print(f"Estimated Distance: {distance_sgk1_pim1}")
    print(f"Estimated Distance: {distance_sgk1_sgk2}")
    print(f"Estimated Distance: {distance_sgk2_pim1}")