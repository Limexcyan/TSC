from sklearn.cluster import BisectingKMeans
import pandas as pd

class Clustering():
    def __init__(self,k,**kwargs):
        self.k = k
        self.clustering_object = None
        self.__dict__.update(kwargs)

    def __call__(self):
        raise NotImplementedError()

class BisectingKMeansClustering(Clustering):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def __call__(self,in_file,dataset=None):
        print("[clustering] - called")
        if dataset is None:
            dataset = pd.read_csv(in_file)
        dataset.reset_index(drop=True)
        clustering_dataset = dataset.copy(deep=True)
        clustering_dataset = clustering_dataset.filter(like="mfp",axis=1)
        mfp_cols = clustering_dataset.columns.tolist()
        self.clustering_object = BisectingKMeans(n_clusters=self.k)
        clustering_distances = self.clustering_object.fit_transform(clustering_dataset)
        clustering_clusters_id = self.clustering_object.labels_
        clustering_distances = clustering_distances[range(len(clustering_dataset)),clustering_clusters_id]
        processed = dataset.copy()#.drop(mfp_cols,axis=1)
        processed.reset_index(drop=True, inplace=True)
        processed['centroid'] = clustering_clusters_id
        processed['distance'] = clustering_distances
        # for some reason ID columns persist, get rid of them
        # but only when dealing with items separately
        #processed.drop(processed.columns[0:2],axis=1, inplace=True)
        print("[clustering] - finished")
        return self.clustering_object, processed
    
    def transform(self,in_file,dataset=None,appendto=None):
        print("[clustering] - called")
        if dataset is None:
            dataset = pd.read_csv(in_file)
        dataset.reset_index(drop=True)
        clustering_dataset = dataset.copy(deep=True)
        clustering_dataset = clustering_dataset.filter(like="mfp",axis=1)
        mfp_cols = clustering_dataset.columns.tolist()
        clustering_distances = self.clustering_object.transform(clustering_dataset)
        clustering_clusters_id = self.clustering_object.predict(clustering_dataset)
        clustering_distances = clustering_distances[range(len(clustering_dataset)),clustering_clusters_id]
        processed = dataset.copy()
        processed.reset_index(drop=True, inplace=True)
        processed['centroid'] = clustering_clusters_id
        processed['distance'] = clustering_distances
        if not appendto is None:
            appendto = pd.read_csv(appendto)
            processed = pd.concat([appendto,processed],axis=0,ignore_index=True)
        print("[clustering] - finished")
        return self.clustering_object, processed
