import numpy as np
import pytest
from practical.ProcessMining.kMeans import KMeans


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    # Sample data with two clusters
    np.random.seed(0)
    cluster1 = np.random.randn(100, 2) + np.array([5, 5])
    cluster2 = np.random.randn(100, 2) + np.array([10, 10])
    data = np.vstack((cluster1, cluster2))
    return data


def test_kmeans(sample_data):
    """Test the KMeans class."""
    kmeans = KMeans(k=2, max_iterations=100)
    clusters = kmeans.run(sample_data)

    # Check if clusters contain expected number of elements
    assert len(clusters) == len(sample_data)

    # Check if each element in clusters is assigned to a cluster index
    assert all(cluster_idx >= 0 for cluster_idx in clusters)

    # Check if centroids are updated correctly
    updated_centroids = kmeans._compute_means(clusters, sample_data)
    assert updated_centroids.shape == (2, 2)  # Expecting 2 centroids with 2 dimensions

    # Check if the closest centroid function returns valid indices
    x = np.array([1, 1])
    centroids = np.array([[0, 0], [2, 2]])
    closest_centroid_idx = kmeans._closest_centroid(x, centroids)
    assert closest_centroid_idx in [0, 1]

    # Check if the Euclidean distance function returns a valid value
    x1 = np.array([1, 1])
    x2 = np.array([4, 5])
    distance = kmeans._euclidean_distance(x1, x2)
    assert isinstance(distance, float)
    assert distance >= 0

    # Run KMeans algorithm with max_iterations set to 0
    kmeans = KMeans(k=2, max_iterations=0)
    clusters = kmeans.run(sample_data)
    # Check if clusters contain expected number of elements
    assert len(clusters) == len(sample_data)
    # Check if all elements are assigned to the first cluster (since no iterations were performed)
    assert np.all(clusters == 0)