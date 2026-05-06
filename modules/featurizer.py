import numpy as np

import pandas as pd

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

class Featurizer():
    def __init__(self, smiles_col='smiles', activity_col='std_value', **kwargs):
        self.smiles_col   = smiles_col
        self.activity_col = activity_col
        self.__dict__.update(kwargs)

    def __call__(self):
        raise NotImplementedError()

class ClusteringFingerprintFeaturizer(Featurizer):
    def __init__(self, morganfp_radius=2, morganfp_size=1024, **kwargs):
        self.morganfp_radius = morganfp_radius
        self.morganfp_size   = morganfp_size
        super().__init__(**kwargs)

    def __call__(self, in_file, in_df=None):
        print("[featurizer] - called")
        if in_df is None:
            in_df = pd.read_csv(in_file)
        in_df.reset_index(drop=True)
        _labels = ['mfp_' + str(n+1) for n in range(self.morganfp_size)]
        features = pd.DataFrame(columns=_labels)
        features_accumulator = pd.DataFrame(columns=_labels)
        count = 0
        generator = GetMorganGenerator(radius=self.morganfp_radius, fpSize=self.morganfp_size)
        total = len(in_df)
        running_total = 0
        for i, row in in_df.iterrows():
            count += 1
            running_total += 1
            smiles = row[self.smiles_col]
            mol = Chem.MolFromSmiles(smiles)
            mfp = generator.GetFingerprint(mol)
            mfpvec = np.array(mfp, dtype=int) # TODO - was boolean, could stay boolean? not if k-means
            # accumulate in batches of 500-1000
            features_accumulator.loc[len(features_accumulator)] = mfpvec
            if (count > 1000):
                print(f"[featurizer] - accumulated {running_total}/{total} items, swapping buffers")
                features = pd.concat([features,features_accumulator])
                features_accumulator[:0] # clear accumulator
                count = 0
        # finally, append to actual features and free accumulator
        print("[featurizer] - consolidating buffers")
        features = pd.concat([features,features_accumulator])
        in_df.reset_index(drop=True)
        features = pd.concat([features,in_df], axis=1)
        features_accumulator[:0]
        try:
            features = features.drop(['Unnamed: 0'])
        except Exception:
            print("[featurizer] - no 'Unnamed: 0' column, not dropping")
        print("[featurizer] - finished")
        return features
