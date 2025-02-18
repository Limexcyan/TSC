import pandas as pd
import numpy as np
from rdkit.Chem import PandasTools as pdt
from featurizer import ClusteringFeaturizer
from sklearn.cluster import BisectingKMeans
from sklearn.metrics import silhouette_score

def prepare_chembl_data(file_name, factor_name=None):
    # join path ("data/", filename)
    chembl_dataset = pdt.LoadSDF(file_name, embedProps=True, molColName=None, smilesName='smiles')
    chembl_dataset = chembl_dataset.drop(dropcols, axis=1)
    if factor_name:
        chembl_dataset['target'] = factor_name
    return chembl_dataset


chembl2016 = prepare_chembl_data('CHEMBL2016_actives.sdf')

# coagulation factor xiii
chembl2016 = pdt.LoadSDF('CHEMBL2016_actives.sdf', embedProps=True, molColName=None, smilesName='smiles')

# coagulation factor vii
chembl244 = pdt.LoadSDF('CHEMBL244_actives.sdf', embedProps=True, molColName=None, smilesName='smiles')

# coagulation factor x
chembl204 = pdt.LoadSDF('CHEMBL204_actives.sdf', embedProps=True, molColName=None, smilesName='smiles')

# thrombin
chembl234 = pdt.LoadSDF('CHEMBL234_actives.sdf', embedProps=True, molColName=None, smilesName='smiles')


dropcols = ['CdId','Mol Weight','Formula','target_id','target_name','organism','target_type','ID','std_unit','std_type']
chembl2016 = chembl2016.drop(dropcols,axis=1)
chembl244 = chembl244.drop(dropcols,axis=1)
chembl204 = chembl204.drop(dropcols,axis=1)
chembl234 = chembl234.drop(dropcols,axis=1)
chembl2016['target'] = 'coagulation factor ix'
chembl244['target'] = 'coagulation factor x'
chembl204['target'] = 'thrombin'
chembl234['target'] = 'dopamine d3 receptor'
data11 = chembl2016
data12 = chembl244
data13 = chembl204

data21 = pd.concat([chembl2016,chembl244]).reset_index(drop=True)
data22 = pd.concat([chembl204,chembl2016]).reset_index(drop=True)
data23 = pd.concat([chembl244,chembl204]).reset_index(drop=True)

data31 = pd.concat([chembl244,chembl204,chembl2016]).reset_index(drop=True)

dataoo = chembl234
dataoj = pd.concat([chembl234,chembl204]).reset_index(drop=True)
datajo = pd.concat([chembl204,chembl234]).reset_index(drop=True)
dataoB = pd.concat([chembl234,chembl244,chembl204]).reset_index(drop=True)


datas = [data11,data12,data13,data21,data22,data23,data31,dataoo,dataoj,dataoB,datajo]
datasq = [dataoo,dataoB,datajo,data13]
datasuber = [dataoB]

feature_count = 1024
quantile_count = 100
featurizer = ClusteringFeaturizer(activity_dist='quantiles',qcount=quantile_count,length=feature_count)
feature_count = (feature_count + 1) * 1.0 # + activity ; konwersja na float
datas = list(map(lambda x: featurizer(x),datasuber))

# clcard = []
clserrm = []
clserrd = []
clserrmb = []
clserrdb = []
rr = 50
print("running cluster cardinality: ")
for dddd in datas:
    print("running analysis for next data.")
    dclserrm = []
    dclserrd = []
    dclserrmb = []
    dclserrdb = []
    for i in range(4, rr, 2):
        res = []
        resb = []
        print(i, end=",")
        for _ in range(5):
            cl = BisectingKMeans(n_clusters=i)
            clds = cl.fit_transform(dddd)
            clid = cl.labels_
            res.append(silhouette_score(dddd, cl.labels_))
            resb.append(np.sum(clds[np.arange(clds.shape[0]), clid]))
        dclserrm.append(np.mean(res))
        dclserrd.append(np.std(res))

        dclserrmb.append(np.mean(resb))
        dclserrdb.append(np.std(resb))
    clserrm.append(dclserrm)
    clserrd.append(dclserrd)
    clserrmb.append(dclserrmb)
    clserrdb.append(dclserrdb)
print("done")

data11.to_csv("data11.csv")
data12.to_csv("data12.csv")
data13.to_csv("data13.csv")
data21.to_csv("data21.csv")
data22.to_csv("data22.csv")
data23.to_csv("data23.csv")
data31.to_csv("data31.csv")
dataoo.to_csv("dataoo.csv")
dataoj.to_csv("dataoj.csv")
dataoB.to_csv("dataoB.csv")
pd.DataFrame(clserrmb).to_csv("clserrmb.csv")
pd.DataFrame(clserrdb).to_csv("clserrdb.csv")
pd.DataFrame(clserrm).to_csv("clserrm.csv")
pd.DataFrame(clserrd).to_csv("clserrd.csv")