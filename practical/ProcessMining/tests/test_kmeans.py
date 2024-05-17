from practical.ProcessMining.k_means import Kmeans

from sklearn.datasets import make_blobs

def test_kmeans():
    centers = 6
    n_samples = 500
    X, y= make_blobs(n_samples=n_samples, 
                     centers=centers, 
                     cluster_std=0.60, 
                     n_features=2)
    kmeans = Kmeans(k=centers, iterations=100)
    kmeans.kmeans(X)
    kmeans.plot_clusters(X)
    assert kmeans.cluster_indices is not None
    assert kmeans.centroids is not None
    assert len(kmeans.cluster_indices) == n_samples
    assert len(set(kmeans.cluster_indices)) == centers
    assert len(kmeans.centroids) == centers

if __name__ == "__main__":
    test_kmeans()