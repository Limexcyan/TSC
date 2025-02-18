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