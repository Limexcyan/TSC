from rdkit.Chem import PandasTools as pdt

# coagulation factor xiii
chembl2016 = pdt.LoadSDF('CHEMBL2016_actives.sdf', embedProps=True, molColName=None, smilesName='smiles')

# coagulation factor ix
chembl2820 = None

# coagulation factor xi
chembl2821 = None

# coagulation factor xii
chembl3991 = None

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

# data21 = pd.concat([chembl2016,chembl244]).reset_index(drop=True)
# data22 = pd.concat([chembl204,chembl2016]).reset_index(drop=True)
# data23 = pd.concat([chembl244,chembl204]).reset_index(drop=True)
#
# data31 = pd.concat([chembl244,chembl204,chembl2016]).reset_index(drop=True)
#
# dataoo = chembl234
# dataoj = pd.concat([chembl234,chembl204]).reset_index(drop=True)
# datajo = pd.concat([chembl204,chembl234]).reset_index(drop=True)
# dataoB = pd.concat([chembl234,chembl244,chembl204]).reset_index(drop=True)