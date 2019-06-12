import numpy as np


def centroids_init(X, k):
    """
    初始化中心点
    """
    n_samples, dim = X.shape
    random_state = np.random.mtrand._rand
    seeds = random_state.permutation(n_samples)[:k]
    centroids = X[seeds]
    return centroids


def compute_dist(X, Y):
    """
    使用矩阵乘法的方法，计算样本点与中心点距离平方
    """
    XX = np.sum(X*X, axis=1)[:, np.newaxis]
    YY = np.sum(Y*Y, axis=1)
    XY = np.dot(X, Y.T)
    return XX + YY - 2 * XY


def update_centers(X, n_clusters, labels, distances):
    """
    更新中心点，解决中心点偏离的问题
    """
    n_features = X.shape[1]

    num_in_cluster = np.zeros((n_clusters,))
    centers = np.zeros((n_clusters, n_features))

    # 寻找空类
    for i in range(n_clusters):
        num_in_cluster[i] = (labels == i).sum()
    empty_clusters = np.where(num_in_cluster == 0)[0]

    if len(empty_clusters):
        far_from_centers = distances.argsort()[::-1]

        for i, cluster_id in enumerate(empty_clusters):
            far_index = far_from_centers[i]
            centers[cluster_id] = X[far_index]
            num_in_cluster[cluster_id] = 1

    for i in range(n_clusters):
        centers[i] += X[labels == i].sum(axis=0)

    centers /= num_in_cluster[:, np.newaxis]

    return centers


def k_init(X, ):
    return


def k_means(X, n_clusters, max_iter, verbose=False, tol=1e-4):
    best_labels, best_inertia, best_centers = None, None, None
    # init
    n_samples = X.shape[0]
    centers = centroids_init(X, n_clusters)

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(n_samples,), dtype=X.dtype)

    # iterations
    for i in range(max_iter):
        Y = centers.copy()
        # 计算样本点到中心点的欧式距离
        dist = compute_dist(X, Y)
        # 记录样本点距离最近的中心点序号
        labels = dist.argmin(axis=1)

        distances = dist[np.arange(dist.shape[0]), labels]
        inertia = distances.sum()

        # 计算新的中心点
        centers = update_centers(X, n_clusters, labels, distances)

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        d_center = np.ravel(Y - centers, order='K')
        center_shift_total = np.dot(d_center, d_center)
        if center_shift_total <= tol:
            if verbose:
                print("Converged at iteration %d: "
                      "center shift %e within tolerance %e"
                      % (i, center_shift_total, tol))
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        dist = compute_dist(X, best_centers)
        best_labels = dist.argmin(axis=1)
        distances = dist[np.arange(dist.shape[0]), best_labels]
        best_inertia = distances.sum()

    return best_labels, best_inertia, best_centers, i + 1
