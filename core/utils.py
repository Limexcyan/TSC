import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools as pdt
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import rdFMCS as fmcs


def consolidate_data(sdfs, userinputs=[], dropcols=[]):
    """

    :param sdfs: d array of filenames for SDF files
    :param userinputs: 2d array of key value pairs for additional user parameters for each SDF file
    :param dropcols: 1d array of columns to drop from the loaded SDF files
    :return: dataframe which concatenates all loaded SDF files
    """

    sdf_df_aggregate = []

    for i in range(len(sdfs)):
        temp = pdt.LoadSDF(sdfs[i], embedProps=True, molColName=None, smilesName='smiles').drop(dropcols, axis=1)
        for additional_labels in userinputs[i]:
            temp[additional_labels[0]] = additional_labels[1]
        sdf_df_aggregate.append(temp)
        print("\t aggregating " + str(i+1) + "/" + str(len(sdfs)) + " sdfs")

    return pd.concat(sdf_df_aggregate).reset_index(drop=True)


def export_data(data, filename):
    """

    :param data:
    :param filename:
    :return:
    """
    data.to_csv(filename,index=False)
    print("done")


def gaussianScaler(numArr):
    # standard gaussian scaler
    # X' = (X-mu)/sigma
    mean = numArr.mean()
    std = numArr.std()
    numArr = (numArr - mean) / std
    return numArr

# if data follows the 'power law' (~ e^x)
def logScaler(numArr):
    # standard logarythmic scaler
    numArr = np.log(numArr)
    return numArr

# in case of uncertain distribution
# divide the data into quantiles
def createQuantiles(dfColumn, numQuantiles):
    # create as many quantiles as wanted
    return pd.qcut(dfColumn, numQuantiles, labels=False, duplicates='drop')

# minmax normalizer
def minMaxScaler(numArr):
    # standard minmax scaler
    minx = np.min(numArr)
    maxx = np.max(numArr)

    numArr = (numArr - minx) / (maxx - minx)
    return numArr
