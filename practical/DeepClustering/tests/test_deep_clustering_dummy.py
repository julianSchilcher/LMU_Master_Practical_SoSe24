from practical.DeepClustering.deep_clustering_dummy import hello_world, MiniBatchKMeansVan
from sklearn.cluster import KMeans
import torch



def test_hello_world_clustering():
    assert hello_world("Deep Clustering") == "Hello Deep Clustering"


def test_minibatch_k_means():
    torch.manual_seed(7)
    data = torch.rand(20, 2)
    cluster = 3

    minibatch_k_means = MiniBatchKMeansVan(cluster, 10, 10, 7)
    minibatch_k_means_centers = minibatch_k_means.fit(data)

    sklearn_k_means = KMeans(cluster, random_state=7)
    sklearn_k_means.fit(data)
    sklearn_k_means_centers = torch.tensor(sklearn_k_means.cluster_centers_, dtype=torch.float)

    assert torch.allclose(minibatch_k_means_centers, sklearn_k_means_centers, atol=1)


def test_minibatch_k_means_distance():
    centroids = torch.tensor([[0.8559, 0.6721],
                              [0.5209, 0.5932],
                              [0.6592, 0.6569]])

    batch = torch.tensor([[0.2868, 0.2063],
                          [0.2071, 0.6297],
                          [0.4451, 0.3593],
                          [0.4451, 0.3593],
                          [0.3742, 0.1953],
                          [0.7204, 0.0731],
                          [0.6592, 0.6569],
                          [0.3887, 0.2214],
                          [0.7572, 0.6948],
                          [0.7204, 0.0731]])

    distance = torch.tensor([1, 1, 1, 1, 1, 1, 2, 1, 0, 1])

    kmeans = MiniBatchKMeansVan(3, 10, 10, 7)
    kmeans_distance = kmeans.distance(batch, centroids)

    assert torch.equal(distance, kmeans_distance)


test_minibatch_k_means_distance()
test_minibatch_k_means()
