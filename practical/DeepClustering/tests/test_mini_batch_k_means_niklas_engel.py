import numpy as np
import pytest
import torch
from practical.DeepClustering.mini_batch_k_means_niklas_engel import \
    MiniBatchKMeans  # Adjust the import path as necessary


@pytest.fixture
def sample_data():
    # Generate some test data
    data = torch.tensor(np.random.rand(100, 5), dtype=torch.float32)  # 100 samples, 5 features
    return data

@pytest.fixture
def kmeans():
    # Initialize MiniBatchKMeans
    num_clusters = 3
    batch_size = 10
    max_iterations = 10
    kmeans = MiniBatchKMeans(num_clusters=num_clusters, batch_size=batch_size, max_iterations=max_iterations, device='cpu')
    return kmeans

def test_initialization(kmeans):
    # Test initialization of cluster centers
    assert kmeans.cluster_centers_ is None, "Cluster centers should initially be None."

def test_no_data(kmeans):
    # Empty dataset
    empty_data = torch.tensor([], dtype=torch.float32).reshape(0, 5)
    with pytest.raises(ValueError):  # Assuming you raise ValueError on empty data
        kmeans.fit(empty_data)

def test_data_types(kmeans, sample_data):
    # Convert sample_data to integers
    float_data = sample_data.float()
    kmeans.fit(float_data)
    assert kmeans.cluster_centers_ is not None, "Cluster centers should be initialized even with float data."
    labels = kmeans.predict(float_data)
    assert labels.shape[0] == float_data.shape[0], "Every data point should be assigned to a cluster with float data."

def test_fit(kmeans, sample_data):
    # Test fitting the model
    kmeans.fit(sample_data)
    assert kmeans.cluster_centers_ is not None, "Cluster centers should not be None after fitting."
    assert kmeans.cluster_centers_.shape[0] == kmeans.num_clusters, "There should be as many cluster centers as num_clusters."
    assert kmeans.cluster_centers_.shape[1] == sample_data.shape[1], "Cluster centers and data points should have the same dimension."

def test_consistency(kmeans, sample_data):
    torch.manual_seed(42)  # Set seed for reproducibility
    kmeans.fit(sample_data)
    centers_first_run = kmeans.cluster_centers_.clone()
    torch.manual_seed(42)  # Reset seed to ensure same initialization
    kmeans.fit(sample_data)
    centers_second_run = kmeans.cluster_centers_
    assert torch.equal(centers_first_run, centers_second_run), "Results should be consistent across runs with the same seed."


def test_predict(kmeans, sample_data):
    # Ensure predict assigns every data point a cluster
    kmeans.fit(sample_data)
    labels = kmeans.predict(sample_data)
    assert labels.shape[0] == sample_data.shape[0], "Every data point should be assigned to a cluster."

