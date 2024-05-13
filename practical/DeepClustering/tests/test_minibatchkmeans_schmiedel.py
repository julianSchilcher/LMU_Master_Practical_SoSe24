import torch.utils
from practical.DeepClustering import minibatchkmeans_schmiedel
import torch
import numpy as np
import pytest

# Fixed values
N_CLUSTERS = 2
MAX_ITER = 20
BATCH_SIZE = 5
RANDOM_STATE = 0
INIT_METHODS = ['random', 'kmeans++']
TESTDATA_ROWS = 100
TESTDATA_COLS = 3



np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Test dataset
data = np.random.rand(TESTDATA_ROWS, TESTDATA_COLS)
torch_data = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)

large_data = np.random.rand(TESTDATA_ROWS*TESTDATA_ROWS, TESTDATA_COLS*TESTDATA_COLS)
large_torch_data = torch.utils.data.DataLoader(dataset=large_data, batch_size=BATCH_SIZE, shuffle=True)

def test_instantiate_miniBatchKMeans_object():
    # Initialize MiniBatch k-Means
    for INIT_METHOD in INIT_METHODS:
        mbkm = minibatchkmeans_schmiedel.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
        
        assert isinstance(mbkm, minibatchkmeans_schmiedel.MiniBatchKMeans)
        assert mbkm.n_clusters == N_CLUSTERS
        assert mbkm.max_iter == MAX_ITER
        assert mbkm.batch_size == BATCH_SIZE
        assert mbkm.random_state == RANDOM_STATE
        assert mbkm.init_method == INIT_METHOD
        assert mbkm.labels_ == None
        assert mbkm.cluster_centers_ == None

def test_random_initial_centroids():
    # Test data to test the random selection result on.
    test_data = data[np.random.choice(data.shape[0], N_CLUSTERS, replace=False)]
    
    # Assert for numpy array
    mbkm_class = minibatchkmeans_schmiedel.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)
    assert np.all(test_data == np.array(mbkm_class._random_centroids(data).tolist()))

    # Assert for torch object
    mbkm_class = minibatchkmeans_schmiedel.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)
    assert torch.all(torch.tensor(test_data) == mbkm_class._random_centroids(torch_data))
        
def test_kmeanspp_initial_centroids():
    # Test if the output is of correct size
    mbkm_class = minibatchkmeans_schmiedel.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)
    cluster_centers = mbkm_class._kmeanspp_centroids(torch_data)
    assert torch.Size([N_CLUSTERS, TESTDATA_COLS]) == cluster_centers.size()
    
def test_minibatchkmeans_algorithm_result_size():
    # Test if the output is of correct size
    
    # Init object
    for INIT_METHOD in INIT_METHODS:
        mbkm = minibatchkmeans_schmiedel.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
        cluster_centers = mbkm._random_centroids(torch_data)
        
        # Assert if the result is of the expected size
        assert torch.Size([N_CLUSTERS, TESTDATA_COLS]) == cluster_centers.size()
    
def test_assigning_labels():
    for INIT_METHOD in INIT_METHODS:
        # Initialize a new MiniBatch k-Means Object
        mbkm = minibatchkmeans_schmiedel.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
        
        # Retrieve the initial random cluster centers
        cluster_centers_initial = mbkm._random_centroids(torch_data)
        
        # Store cluster centers in the object data
        mbkm.cluster_centers_ = cluster_centers_initial
        
        # Retrieve the initial dataset labels
        labels_initial = mbkm._assign_labels(torch_data)
        
        # Assert that they are of correct length and have the correct range of 0..N_CLUSTERS
        assert labels_initial.shape[0] == TESTDATA_ROWS # Length/Amount
        assert torch.all(labels_initial.unique() == torch.tensor([i for i in range(0, N_CLUSTERS)])) # 0..N_CLUSTERS
    

def test_minibatchkmeans_full_algorithm():
    for INIT_METHOD in INIT_METHODS:
        # Init object
        mbkm = minibatchkmeans_schmiedel.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
        mbkm.fit(torch_data)
        
        # Assert if final label result is again of correct size and correct amount of clusters (0..N_CLUSTERS)
        assert mbkm.labels_.shape[0] == TESTDATA_ROWS # Length/Amount
        assert torch.all(mbkm.labels_.unique() == torch.tensor([i for i in range(0, N_CLUSTERS)])) # 0..N_CLUSTERS
        
def test_minibatchkmeans_full_algorithm_large_dataset():
    for INIT_METHOD in INIT_METHODS:
        # Init object
        mbkm = minibatchkmeans_schmiedel.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
        mbkm.fit(large_torch_data)
        
        # Assert if final label result is again of correct size and correct amount of clusters (0..N_CLUSTERS)
        assert mbkm.labels_.shape[0] == TESTDATA_ROWS*TESTDATA_ROWS # Length/Amount
        assert torch.all(mbkm.labels_.unique() == torch.tensor([i for i in range(0, N_CLUSTERS)])) # 0..N_CLUSTERS
        
def test_minibatchkmeans_full_algorithm_large_dataset_small_iteration():
    for INIT_METHOD in INIT_METHODS:
        mbkm = minibatchkmeans_schmiedel.MiniBatchKMeans(n_clusters=N_CLUSTERS, max_iter=2, batch_size=BATCH_SIZE, random_state=RANDOM_STATE, init=INIT_METHOD)
        mbkm.fit(large_data)
        
        # Assert if final label result is again of correct size and correct amount of clusters (0..N_CLUSTERS)
        assert mbkm.labels_.shape[0] == TESTDATA_ROWS*TESTDATA_ROWS # Length/Amount
        assert torch.all(mbkm.labels_.unique() == torch.tensor([i for i in range(0, N_CLUSTERS)])) # 0..N_CLUSTERS
    
    
# Failing tests, as expected
def test_wrong_dataset_type():
    with pytest.raises(NotImplementedError):
        mbkm = minibatchkmeans_schmiedel.MiniBatchKMeans()
        mbkm.fit([])
        
def test_wrong_batch_size():
    with pytest.raises(ValueError):
        mbkm = minibatchkmeans_schmiedel.MiniBatchKMeans()
        wrong_torch_data = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE*2, shuffle=True)
        mbkm.fit(wrong_torch_data)

def test_wrong_sampler():
    with pytest.raises(ValueError):
        mbkm = minibatchkmeans_schmiedel.MiniBatchKMeans()
        wrong_torch_data = torch.utils.data.DataLoader(dataset=data, batch_size=BATCH_SIZE, sampler=torch.utils.data.sampler.SequentialSampler)
        mbkm.fit(wrong_torch_data)
        
def test_false_centroid_init_method():
    with pytest.raises(NotImplementedError):
        mbkm = minibatchkmeans_schmiedel.MiniBatchKMeans(init='')
        mbkm.fit(data)

def test_false_data_type_initial_random_centroids():
    with pytest.raises(NotImplementedError):
        mbkm = minibatchkmeans_schmiedel.MiniBatchKMeans()
        mbkm._random_centroids([])