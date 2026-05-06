import pandas as pd

from rdkit.Chem import PandasTools as pdt

class Consolidator():
    def __init__(self, smiles_col='smiles', keep_columns=["chembl_id","smiles"], **kwargs):
        self.keep_columns = keep_columns
        self.smiles_col = smiles_col
        self.__dict__.update(kwargs)

    def __call__(self, sdfs, user_append=None):
        print("[consolidator] - called")
        sdfs_df_accumulator = []
        if user_append is None:
            user_append = [[] for _ in range(len(sdfs))]
        for i in range(len(sdfs)):
            loaded = pdt.LoadSDF(sdfs[i], embedProps=True, molColName=None, smilesName=self.smiles_col)#.drop(self.dropcols, axis=1)
            loaded = loaded[loaded.columns.intersection(self.keep_columns)]
            for user_label in user_append[i]:
                loaded[user_label['key']] = user_label['value']
            sdfs_df_accumulator.append(loaded)
            print(f"[consolidator] - completed {i+1}/{len(sdfs)} jobs")
        result = pd.concat(sdfs_df_accumulator)
        result.reset_index(drop=True,inplace=True)
        print("[consolidator] - finished")
        return result

