import numpy as np
import pandas as pd


def gaussianScaler(numArr):
    mean = numArr.mean()
    std = numArr.std()
    numArr = (numArr - mean) / std
    return numArr


def logScaler(numArr):
    numArr = np.log(numArr)
    return numArr


def createQuantiles(dfColumn, numQuantiles):
    return pd.qcut(dfColumn, numQuantiles, labels=False, duplicates='drop')


def minMaxScaler(numArr):
    min_x = np.min(numArr)
    max_x = np.max(numArr)
    numArr = (numArr - min_x) / (max_x - min_x)
    return numArr
