from sklearn.cluster import KMeans
from k_means import k_means

import numpy as np
from keras.datasets import mnist
import time

from eval import ClusterEval

# load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# data dimension
raw_dim = 28 * 28                  # raw dimension

# choose data
train_num = 20000
data = train_images[0 : train_num].reshape((train_num,raw_dim)) / 255.              # X
labels = train_labels[0 : train_num]            # y

def cluster_on_raw_data(data):
    ### using kmeans on raw data
    ### @return cluster labels
    print("Begin clustering on raw data...")
    print("Data shape = ", data.shape)
    start = time.time()
    kmeans = KMeans(n_clusters = 10)
    kmeans.fit(data)
    end = time.time()
    print("Clustering on raw data, using time = ", end - start)
    return kmeans.labels_

def my_cluster_on_raw_data(data):
        ### using kmeans on raw data
    ### @return cluster labels
    print("Begin my clustering on raw data...")
    print("Data shape = ", data.shape)
    start = time.time()
    labels, _, _, _ = k_means(data, n_clusters=10, max_iter=300)
    end = time.time()
    print("Clustering on raw data, using time = ", end - start)
    return labels

def analysis_and_plot(data, clu_labels, labels = None):
    ### analyse the cluster result, CP, SP, RI, ARI, FusionMatrix
    ### @params data : numpy.array
    ### @params clu_labels : clustered labels
    ### @params labels : real labels

    evaler = ClusterEval(data, clu_labels, labels)
    print("CP = ", evaler.CP)
    print("SP = ", evaler.SP)
    if isinstance(labels, np.ndarray):
        print("RI = ", evaler.RI)
        print("ARI = ", evaler.ARI)
        '''
        print("Confusion matrix:")
        for row in evaler.norm_labels_grid:
            print(list(row))
        plt.figure()
        plt.imshow(evaler.norm_labels_grid)
        plt.show()
        '''

print("###################################")
print("Cluster on raw data and evaluate...")
clu_labels = cluster_on_raw_data(data)
analysis_and_plot(data, clu_labels, labels)
print("###################################")

print("###################################")
print("my Cluster on raw data and evaluate...")
clu_labels = my_cluster_on_raw_data(data)
analysis_and_plot(data, clu_labels, labels)
print("###################################")