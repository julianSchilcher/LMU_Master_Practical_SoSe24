from clustpy.data import load_optdigits, load_pendigits, load_har, z_normalization
from clustpy.metrics import unsupervised_clustering_accuracy as acc
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
import time
import numpy as np
import torch
from scipy.spatial.distance import cdist

from practical.DeepClustering.minibatchkmeans_schmiedel import MiniBatchKMeans as MinibatchSchmiedel
from practical.DeepClustering.deep_clustering_dummy_Xuechun_Li import MiniBatchKmeans as MinibatchXuechun
from practical.DeepClustering.JulianSchilcher.minibatch_kmeans import MinibatchKmeans as MinibatchSchilcher
from practical.DeepClustering.mini_batch_k_means_niklas_engel import MiniBatchKMeans as MinibatchEngel
from practical.DeepClustering.deep_clustering_dummy import MiniBatchKMeansVan as MinibatchVan
from practical.DeepClustering.FlorianKittel.kmeans import MiniBatchKMeans as MinibatchKittel
from practical.DeepClustering.robin_loebbert import MiniBatchKMeans as MinibatchLoebbert

MAX_ITERS = 25
BATCH_SIZE = 256
N_SEEDS = 10


def implementation_loebbert(data, n_clusters, seed):
    mbk = MinibatchLoebbert(n_clusters, iterations=MAX_ITERS, mini_batch_size=BATCH_SIZE)
    mbk.fit(data)
    labels = np.argmin(cdist(data, mbk.c.numpy()), axis=1)
    return labels


def implementation_kittel(data, n_clusters, seed):
    mbk = MinibatchKittel(n_clusters, iterations=MAX_ITERS, batch_size=BATCH_SIZE)
    centers = mbk.fit(torch.from_numpy(data).float())
    labels = np.argmin(cdist(data, centers.numpy()), axis=1)
    return labels


def implementation_van(data, n_clusters, seed):
    mbk = MinibatchVan(n_clusters, iterations=MAX_ITERS, batch_size=BATCH_SIZE, random_state=seed)
    centers = mbk.fit(torch.from_numpy(data).float())
    labels = np.argmin(cdist(data, centers.numpy()), axis=1)
    return labels


def implementation_engel(data, n_clusters, seed):
    mbk = MinibatchEngel(n_clusters, max_iterations=MAX_ITERS, batch_size=BATCH_SIZE)
    mbk.fit(torch.from_numpy(data).float())
    labels = np.argmin(cdist(data, mbk.cluster_centers_.numpy()), axis=1)
    return labels


def implementation_schilcher(data, n_clusters, seed):
    mbk = MinibatchSchilcher(n_clusters, max_iterations=MAX_ITERS, batch_size=BATCH_SIZE)
    labels = mbk.fit(data)
    return labels


def implementation_schmiedel(data, n_clusters, seed):
    mbk = MinibatchSchmiedel(n_clusters, max_iter=MAX_ITERS, batch_size=BATCH_SIZE)
    mbk.fit(data)
    return mbk.labels_.numpy()


def implementation_xuechun(data, n_clusters, seed):
    mbk = MinibatchXuechun(n_clusters, n_iter=MAX_ITERS, batch_size=BATCH_SIZE, tol=1e-4)
    centers = mbk.forward(torch.from_numpy(data).float()).numpy()
    labels = np.argmin(cdist(data, centers), axis=1)
    return labels


if __name__ == "__main__":
    dataset_loaders = [load_optdigits, load_pendigits, load_har]
    implementations = [("Schmiedel", implementation_schmiedel), ("Xuechun", implementation_xuechun),
                       ("Schilcher", implementation_schilcher), ("Engel", implementation_engel),
                       ("Van", implementation_van), ("Kittel", implementation_kittel),
                       ("Loebbert", implementation_loebbert)]
    for dl in dataset_loaders:
        data, labels_true = dl(return_X_y=True)
        print("===== {0} {1} =====".format(dl.__name__, data.shape))
        # data = z_normalization(data, True)
        n_clusters = np.unique(labels_true).shape[0]
        results = {}
        for name, _ in implementations:
            results[name] = {"time": [], "acc": [], "ari": [], "nmi": []}
        for s in range(N_SEEDS):
            # Alexander schmiedel
            for name, impl in implementations:
                np.random.seed(s)
                torch.manual_seed(s)
                begin = time.time()
                labels_pred = impl(data, n_clusters, s)
                end = time.time()
                results[name]["time"].append(end - begin)
                results[name]["acc"].append(acc(labels_true, labels_pred))
                results[name]["ari"].append(ari(labels_true, labels_pred))
                results[name]["nmi"].append(nmi(labels_true, labels_pred))
        for name, _ in implementations:
            print("--- {0} ---".format(name))
            for key in results[name].keys():
                print("{0}: {1} +- {2}".format(key, round(np.mean(results[name][key]), 3),
                                               round(np.std(results[name][key]), 3)))
