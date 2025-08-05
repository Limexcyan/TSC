import pandas as pd

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools as pdt
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import rdFMCS as fmcs


class Featurizer:
    """

    :param smiles_col: name of the column containing the smiles inside the dataframe which will be featurized
    :param activity_col: name of the column containing activity values inside the dataframe which will be featurized

    """
    def __init__(self, smiles_col='smiles', activity_col='std_value', **kwargs):
        self.smiles_col = smiles_col
        self.activity_col = activity_col
        self.__dict__.update(kwargs)

    def __call__(self, df):
        raise NotImplementedError()


class ClusteringPreprocessFeaturizer(Featurizer):
    """

    :param preprocess_type: additional preprocessing steps to take for the activity values, can be quantiles, powerlaw or gaussian
    :param morganfp_radius: radius argument passed into the morgan fingerprint generator for datapoints
    :param morganfp_size: length of the bit vector passed into the morgan fingerprint generator for datapoints
    :param quant_count: in case of preprocess type being quantiles, the amount of quantiles to generate

    """
    def __init__(self, preprocess_type=None, morganfp_radius=2, morganfp_size=1024, quant_count=0, **kwargs):
        self.preprocess_type = preprocess_type
        self.morganfp_radius = morganfp_radius
        self.morganfp_size   = morganfp_size

        self.quant_count     = quant_count

        super().__init__(**kwargs)

    """
    
    :param df: dataframe of compounds which needs to be featurized
    :param quick: switch merging of the dataframe to instead append to an array when generating features
    :return: dataframe or array containing morgan fingerprints of compounds
    """
    def __call__(self, df):
        if(self.preprocess_type != None and self.preprocess_type != 'gaussian' and self.preprocess_type != 'powerlaw' and self.preprocess_type != 'quantiles'):
            print('The preprocess_type argument was wrongly set and is assumed to be None. Please use one of the following:\ngaussian,\npowerlaw,\nquantiles')
            self.preprocess_type = None
        
        # if quantiles are selected, there must be at least two of them
        if(self.preprocess_type == 'quantiles' and self.quant_count < 2):
            print('The preprocess_type of quantiles requires the quant_count argument to be at least 2. Defaulting to 100 quantiles')
            self.quant_count = 100

        labels = ['mfp_' + str(n) for n in np.arange(self.morganfp_size)]

        features = pd.DataFrame(columns=labels)
        features_sub = pd.DataFrame(columns=labels)

        for i,row in df.iterrows():
            smiles = row[self.smiles_col]
            mol = Chem.MolFromSmiles(smiles)
            mfp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morganfp_radius, nBits=self.morganfp_size)
            mfpvec = np.array(mfp, dtype=int)
            fullvec = np.array([np.hstack(mfpvec)],dtype=float)
            if (i % 500 != 0):
                fta = pd.DataFrame(fullvec, columns=labels)
                features_sub = pd.concat([features_sub, fta], ignore_index=True)
            else :
                print("\tcompleted " + str(round(100 * (i/len(df)),1)) + "%")
                features = pd.concat([features, features_sub], ignore_index=True)
                features_sub = pd.DataFrame(columns=labels)

        print("\tdone.")
        return features


