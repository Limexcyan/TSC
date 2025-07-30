from utils import *
from featurizer import *
from analysis import *
from config import *
import os

if __name__ == "__main__":
    sdfs = molecules_sdfs
    userinputs = molecules_userinputs
    print("trying to find existing data")
    if os.path.isfile("data/full_sdfs.csv"):
        print("check success for full sdfs")
        full_df = pd.read_csv("data/full_sdfs.csv",index_col=False)
    else:
        print("consolidating data")
        full_df = consolidate_data(sdfs, userinputs=userinputs, dropcols=dropped_columns)
        print("exporting consolidated data")
        export_data(full_df, "data/full_sdfs.csv")
    if os.path.isfile("data/processed_sdfs.csv"):
        print("check success for processed data")
        datas = pd.read_csv("data/processed_sdfs.csv",index_col=False)
        feature_count = feature_count * 1.0
    else:
        print("featurizing data - populating with morgan fingerprints")
        print("\tfeature count - " + str(feature_count))
        featurizer = ClusteringPreprocessFeaturizer(preprocess_type='quantiles', quant_count=quantile_count, morganfp_size=feature_count)
        feature_count = feature_count * 1.0
        datas = featurizer(full_df)
        print("exporting featurized data")
        export_data(datas, "data/processed_sdfs.csv")
    
    print("performing analysis...")
    out_data_full, out_data_processed = perform_analysis(datas,cluster_count,full_df)

    print("SUMMARY:")
    summary(out_data_full, userinputs, 1e-20)


