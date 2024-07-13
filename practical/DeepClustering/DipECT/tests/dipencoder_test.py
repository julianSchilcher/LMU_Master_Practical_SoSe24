from sklearn.datasets import make_blobs
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._utils import encode_batchwise
from clustpy.metrics import unsupervised_clustering_accuracy
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from sklearn.preprocessing import minmax_scale
from practical.DeepClustering.DipECT.dipect import Cluster_Tree, DipECT
import torch
import numpy as np

def _gen_artificial_dataset(number_clusters, n_features = 2, return_dataloader=True, n_samples=600):
    X, y = make_blobs(n_samples=n_samples, centers=number_clusters, random_state=10, n_features=n_features)
    if return_dataloader:
        return get_dataloader(X, 50), y
    else:
        return X, y

def _get_mock_autoencoder(fkt_object=None):
    if fkt_object is None:
        fkt_object = lambda x: x
    # create mock paramters
    parameters = lambda: iter(torch.nn.Parameter(torch.tensor([1.0])))
    return type("Autoencoder", (), {"encode": fkt_object, "parameters": parameters})

def test_Tree():
    dataloader, labels = _gen_artificial_dataset(4)
    autoencoder = _get_mock_autoencoder()
    tree = Cluster_Tree(dataloader, autoencoder, None, "cpu",  np.random.RandomState(0), "kmeans", 10)
    assert tree.root.projection_axis is not None and tree.root.projection_axis.numel() == 2
    assert tree.root.higher_projection_child is not None and tree.root.lower_projection_child is not None
    assert not tree.root.is_leaf_node() and tree.root.higher_projection_child.is_leaf_node() and tree.root.lower_projection_child.is_leaf_node()

def test_grow_assign():
    dataloader, labels = _gen_artificial_dataset(3)
    autoencoder = _get_mock_autoencoder()

    X = dataloader.dataset.tensors[0].numpy()

    tree = Cluster_Tree(dataloader, autoencoder, None, "cpu", np.random.RandomState(0), "kmeans", 10)
    tree.assign_to_tree(torch.from_numpy(X))
    tree.grow_tree(dataloader, autoencoder, None, 20, 1.0, 1, 0)
    # reasssign data so that the new leaf nodes contain its data
    tree.assign_to_tree(torch.from_numpy(X))

    pred_labels = np.ones(len(X))*-1
    if tree.root.lower_projection_child.is_leaf_node():
        # consistency check if merging of child samples result in parent samples
        X_combined = torch.cat((tree.root.higher_projection_child.lower_projection_child.assignments, tree.root.higher_projection_child.higher_projection_child.assignments), dim=0)
        X_combined_sorted, _ = torch.sort(X_combined, dim=0)
        X_sorted, _ = torch.sort(tree.root.higher_projection_child.assignments, dim=0)
        is_identical = torch.equal(X_sorted, X_combined_sorted)

        # label prediction
        pred_labels[tree.root.higher_projection_child.lower_projection_child.assignment_indices] = 0
        pred_labels[tree.root.higher_projection_child.higher_projection_child.assignment_indices] = 1
        pred_labels[tree.root.lower_projection_child.assignment_indices] = 2
    else:
        # consistency check if merging of child samples result in parent samples
        X_combined = torch.cat((tree.root.lower_projection_child.lower_projection_child.assignments, tree.root.lower_projection_child.higher_projection_child.assignments), dim=0)
        X_combined_sorted, _ = torch.sort(X_combined, dim=0)
        X_sorted, _ = torch.sort(tree.root.lower_projection_child.assignments, dim=0)
        is_identical = torch.equal(X_sorted, X_combined_sorted)

        # label prediction
        pred_labels[tree.root.lower_projection_child.lower_projection_child.assignment_indices] = 0
        pred_labels[tree.root.lower_projection_child.higher_projection_child.assignment_indices] = 1
        pred_labels[tree.root.higher_projection_child.assignment_indices] = 2
    
    # for the simple make_bloobs data set, we should get a good accuracy with a simple tree split
    assert is_identical
    assert sum(pred_labels == -1) == 0 # each point got a label
    assert unsupervised_clustering_accuracy(labels, pred_labels) == 1.0


def test_whole_algorithm():
    # test "whole" dipect algorithm on simple higher dimensinal data 
    X, labels = _gen_artificial_dataset(3, 50, return_dataloader=False, n_samples=1000)
    X = minmax_scale(X, feature_range=(0, 1), axis=1)
    autoencoder = FeedforwardAutoencoder([50, 5])
    dipect = DipECT(autoencoder=autoencoder, autoencoder_pretrain_n_epochs=5, random_state=np.random.RandomState(15), clustering_n_epochs=3, tree_growth_frequency=1, tree_growth_unimodality_treshold=1.0, tree_growth_min_cluster_size=50, pruning_threshold=20, embedding_size=5)
    dipect = dipect.fit_predict(X)
    assert dipect.tree_.flat_accuracy(labels, 3) > 0.95
