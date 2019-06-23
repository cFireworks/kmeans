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
low_dim = 200                      # random projection to low-dimension

# random_projection matrix
rj_matrix = 1.0 - 2.0 * (np.random.rand(raw_dim, low_dim) > 0.5)
rj_matrix = rj_matrix / np.sqrt(low_dim)
print(np.sum(rj_matrix), np.max(rj_matrix), np.min(rj_matrix))

# choose data
train_num = 20000
data = train_images[0: train_num].reshape(
    (train_num, raw_dim)) / 255.              # X
labels = train_labels[0: train_num]            # y


def cluster_sklearn_kmeans(data, n_cluster=10):
    # using kmeans on raw data
    # @return cluster labels
    print("Begin sklearn clustering on raw data...")
    print("Data shape = ", data.shape)
    start = time.time()
    kmeans = KMeans(n_clusters=n_cluster)
    kmeans.fit(data)
    end = time.time()
    print("Clustering on raw data, using time = ", end - start)
    return kmeans.labels_


def my_cluster_my_kmeans(data, n_cluster=10):
    # using kmeans on raw data
    # @return cluster labels
    print("Begin my clustering on raw data...")
    print("Data shape = ", data.shape)
    start = time.time()
    labels, _, _, _ = k_means(data, n_clusters=n_cluster, max_iter=300)
    end = time.time()
    print("Clustering on raw data, using time = ", end - start)
    return labels


cluster_fn = [cluster_sklearn_kmeans, my_cluster_my_kmeans]


def cluster_on_rj_data(data, dim=100, function_name=my_cluster_my_kmeans):
    # using random projection to reduce the dimension of raw data, then cluster
    # @return cluster labels
    print("Begin clustering on low-dimension data...")
    print("Data shape = ", data.shape)

    print("First random projection...")
    start = time.time()
    rj_data = np.dot(data, rj_matrix)
    end = time.time()
    print("Random projection time = ", end - start)

    print("Second kmeans...")
    labels = function_name(rj_data, n_cluster=10)
    return labels


def cluster_on_rs_data(data, p=0.01, function_name=my_cluster_my_kmeans):
    # using random sparsification to sparse raw data, then cluster
    # @return cluster labels
    print("Begin clustering on sparsed data...")
    print("Data shape = ", data.shape)

    print("First random projection...")
    start = time.time()
    rj_data = np.dot(data, rj_matrix)
    end = time.time()
    print("Random projection time = ", end - start)

    print("Second random sparsification...")
    start = time.time()
    # construct random sparsification matrix
    n = rj_data.shape[0]                      # the number of data points
    max_v = np.max(np.abs(rj_data))           # max value
    tau = p * ((rj_data / max_v) ** 2)        # tau_ij

    # sparsification probability
    prob = np.zeros_like(tau, dtype=np.float32)
    sqrt_tau = 64. * np.sqrt(tau / n) * np.log(n) * np.log(n)

    prob[tau > sqrt_tau] = tau[tau > sqrt_tau]
    prob[tau <= sqrt_tau] = sqrt_tau[tau <= sqrt_tau]

    sparse_map = np.random.rand(rj_data.shape[0], rj_data.shape[1]) <= prob

    # sparsification
    rs_data = rj_data.copy()
    index = (prob != 0.0) & (sparse_map == 1.0)
    rs_data[index] = rs_data[index] / \
        prob[index]         # data[i][j]/prob[i][j]
    rs_data[sparse_map == 0.0] = 0.0                      # data[i][j] = 0.0

    end = time.time()
    print("Random projection time = ", end - start)

    print("Before sparsification, the number of zero-elements is:",
          np.sum(rj_data == 0.0)/(rj_data.shape[0] * rj_data.shape[1]))
    print("After sparsification, the number of zero-elements is:",
          np.sum(rs_data == 0.0)/(rs_data.shape[0] * rs_data.shape[1]))

    print("Second kmeans...")
    labels = function_name(rs_data, n_cluster=10)
    return labels


def analysis_and_plot(data, clu_labels, labels=None):
    # analyse the cluster result, CP, SP, RI, ARI, FusionMatrix
    # @params data : numpy.array
    # @params clu_labels : clustered labels
    # @params labels : real labels

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

# print("###################################")
# print("Cluster on raw data and evaluate...")
# clu_labels = cluster_on_raw_data(data)
# analysis_and_plot(data, clu_labels, labels)
# print("###################################")

# print("###################################")
# print("my Cluster on raw data and evaluate...")
# clu_labels = my_cluster_on_raw_data(data)
# analysis_and_plot(data, clu_labels, labels)
# print("###################################")


print("###################################")
print("Cluster on random sparsification data and evaluate...")
clu_labels = cluster_on_rs_data(data)
analysis_and_plot(data, clu_labels, labels)
print("###################################")
