import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import rdkit
import ipywidgets
import sklearn

from featurizer import *
from datasets import *
from preprocess import *
from clustering import *
from visualization import *

if __name__ == "__main__":
    # Avaliable datasets:
    # dataset1, dataset2, dataset3
    chemblXYZ = 256
    data = chemblXYZ

    print(data)
    a1, a2, a3, a4 = perform_analysis(datas[3], 30, data21, datas[6], data31)

    @interact(cid=(0,30))
    def f1(cid=6):
        return findmcs(cid,a3,data21)
    @interact(cid=(0,30))
    def f2(cid=6):
        return findmcs(cid,a4,data31)
    @interact(cid=(0,30),eid=(0,100))
    def f3(cid=6,eid=0):
        return examine_cluster(a3,a1,cid,eid)
    @interact(cid=(0,30),eid=(0,100))
    def f3(cid=6,eid=0):
        return examine_cluster(a4,a2,cid,eid)


    for i in range(20, 40):
        examine_cluster(a4, a2, 18, i)

    b1, b2, b3, b4 = perform_analysis(datas[0], 10, data11, datas[6], data31)

    @interact(cid=(0,30))
    def f1(cid=6):
        return findmcs(cid,b1,data11)
    @interact(cid=(0,30))
    def f2(cid=6):
        return findmcs(cid,b2,data31)
    @interact(cid=(0,30),eid=(0,100))
    def f3(cid=6,eid=0):
        return examine_cluster(b3,b1,cid,eid)
    @interact(cid=(0,30),eid=(0,100))
    def f3(cid=6,eid=0):
        return examine_cluster(b4,b2,cid,eid)

    for i in range(21):
        examine_cluster(b4, b2, 9, i)

    c1, c2, c3, c4 = perform_analysis(datas[7], 30, dataoo, datas[8], dataoj)
    show_labelc(c4, 30, dataoj)


    @interact(cid=(0, 30))
    def f1(cid=6):
        return findmcs(cid, c1, dataoo)


    @interact(cid=(0, 30))
    def f2(cid=6):
        return findmcs(cid, c2, dataoj)


    @interact(cid=(0, 30), eid=(0, 100))
    def f3(cid=6, eid=0):
        return examine_cluster(c3, c1, cid, eid)


    @interact(cid=(0, 30), eid=(0, 1000))
    def f3(cid=6, eid=0):
        return examine_cluster(c4, c2, cid, eid)


    for i in range(0, 30):
        examine_cluster(c4, c2, 28, i)

    d1, d2, d3, d4 = perform_analysis(datas[3], 30, data13, datas[2], datajo)
    show_labelc(d4, 30, datajo)


    @interact(cid=(0, 30))
    def f1(cid=6):
        return findmcs(cid, d1, data13)


    @interact(cid=(0, 30))
    def f2(cid=6):
        return findmcs(cid, d2, datajo)


    @interact(cid=(0, 30), eid=(0, 100))
    def f3(cid=6, eid=0):
        return examine_cluster(d3, d1, cid, eid)


    @interact(cid=(0, 30), eid=(0, 100))
    def f3(cid=6, eid=0):
        return examine_cluster(d4, d2, cid, eid)


    for i in range(50, 70):
        examine_cluster(d4, d2, 9, i)

    e1, e2, e3, e4 = perform_analysis(datas[0], 30, dataoo, datas[1], dataoB)
    show_labelc(e4, 30, dataoB)

    @interact(cid=(0,30))
    def f1(cid=6):
        return findmcs(cid,e1,dataoo)
    @interact(cid=(0,30))
    def f2(cid=6):
        return findmcs(cid,e2,dataoB)
    @interact(cid=(0,30),eid=(0,100))
    def f3(cid=6,eid=0):
        return examine_cluster(e3,e1,cid,eid)
    @interact(cid=(0,30),eid=(0,100))
    def f3(cid=6,eid=0):
        return examine_cluster(e4,e2,cid,eid)


    e1, _, e3, _ = perform_analysis(datas[1], 250, dataoB, datas[1], dataoB)
    show_labelc(e3, 50, dataoB)


    @interact(cid=(0, 50))
    def f1(cid=6):
        return findmcs(cid, e1, dataoB)


    @interact(cid=(0, 50), eid=(0, 100))
    def f3(cid=6, eid=0):
        return examine_cluster(e3, e1, cid, eid)


    for i in range(70, 87):
        examine_cluster(e3, e1, 29, i)

    dclserrm = []
    dclserrd = []
    dclserrmb = []
    dclserrdb = []
    for i in range(100, 400, 4):
        res = []
        resb = []
        print(i, end=",")
        for _ in range(5):
            cl = BisectingKMeans(n_clusters=i)
            clds = cl.fit_transform(datas[0])
            clid = cl.labels_m
            res.append(silhouette_score(datas[0], cl.labels_))
            resb.append(np.sum(clds[np.arange(clds.shape[0]), clid]))
        dclserrm.append(np.mean(res))
        dclserrd.append(np.std(res))
        dclserrmb.append(np.mean(resb))
        dclserrdb.append(np.std(resb))

    plt.figure(figsize=(8,8))
    plt.errorbar(range(100,400,4), dclserrm,dclserrd,linestyle='None',fmt='o')
    plt.title('Score vs Clusters Used')
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')

    plt.figure(figsize=(8, 8))
    plt.errorbar(range(100, 400, 4), dclserrmb, dclserrdb, linestyle='None', fmt='^')
    plt.title('Total Distance normalized vs Clusters Used')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distance normalized')

