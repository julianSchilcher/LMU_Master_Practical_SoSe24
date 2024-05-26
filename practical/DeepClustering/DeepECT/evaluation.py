import os
import sys

sys.path.append(os.getcwd())

from enum import Enum

import numpy as np
import torch
from clustpy.data import load_mnist
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.deep.autoencoders._abstract_autoencoder import \
    _AbstractAutoencoder
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.utils import Bunch


class FlatClusteringMethod(Enum):
    DEEPECT = 'DeepECT'
    DEEPECT_AUGMENTED = 'DeepECT + Augmentation'
    IDEC = 'IDEC'
    KMEANS = 'KMeans'


class HierarchicalClusteringMethod(Enum):
    DEEPECT = 'DeepECT'
    DEEPECT_AUGMENTED = 'DeepECT + Augmentation'
    IDEC_SINGLE = 'IDEC + Single'
    IDEC_COMPLETE = 'IDEC + Complete'
    AE_BISECTING = 'Autoencoder + Bisection'
    AE_SINGLE = 'Autoencoder + Single'
    AE_COMPLETE = 'Autoencoder + Complete'

def calculate_nmi(true_labels, predicted_labels):
    """
    Calculate the Normalized Mutual Information (NMI) between true and predicted labels.

    Parameters
    ----------
    true_labels : array-like
        The true labels of the data points.
    predicted_labels : array-like
        The predicted labels of the data points.

    Returns
    -------
    nmi : float
        The NMI score.
    """
    return normalized_mutual_info_score(true_labels, predicted_labels)

def calculate_acc(true_labels, predicted_labels):
    """
    Calculate the Clustering Accuracy (ACC) between true and predicted labels.

    Parameters
    ----------
    true_labels : array-like
        The true labels of the data points.
    predicted_labels : array-like
        The predicted labels of the data points.

    Returns
    -------
    acc : float
        The accuracy score.
    """

    # Ensure true and predicted labels are numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Create the cost matrix
    max_label = max(predicted_labels.max(), true_labels.max()) + 1
    cost_matrix = np.zeros((max_label, max_label), dtype=int)
    for i in range(predicted_labels.size):
        cost_matrix[predicted_labels[i], true_labels[i]] += 1

    # Solve the linear assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    
    # Calculate accuracy
    total_correct = cost_matrix[row_ind, col_ind].sum()
    acc = total_correct / predicted_labels.size
    
    return acc


def pretraining(init_autoencoder: _AbstractAutoencoder, autoencoder_params_path: str, dataset: Bunch, seed: int):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    # Load and preprocess data
    data = dataset['data']

    # TODO: Check for augmentation here

    # Initialize the autoencoder
    autoencoder: _AbstractAutoencoder = init_autoencoder(layers=[data.shape[1], 500, 500, 2000, 10], reusable=True)
    
    if not os.path.exists(autoencoder_params_path):
        # Train the autoencoder if parameters file does not exist
        autoencoder.fit(n_epochs=50, optimizer_params={'lr': 0.0001}, data=data, batch_size=256)
        autoencoder.save_parameters(autoencoder_params_path)
        print('Autoencoder pretraining complete and saved.')
    else:
        # Load the existing parameters
        autoencoder.load_parameters(autoencoder_params_path)
        print('Autoencoder parameters loaded from file.')
    
    return autoencoder


def flat(autoencoder: _AbstractAutoencoder, autoencoder_params_path: str, dataset: Bunch, seed: int):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    # Load and preprocess data
    data = dataset['data']
    labels = dataset['target']


    for method in FlatClusteringMethod:
        if method == FlatClusteringMethod.KMEANS:
            # Load the autoencoder parameters
            autoencoder.load_parameters(autoencoder_params_path)
            # Encode the data
            embeddings = autoencoder.encode(torch.tensor(data, dtype=torch.float32)).detach().numpy()
            # Perform flat clustering with KMeans
            kmeans = KMeans(n_clusters=10, random_state=seed)
            predicted_labels = kmeans.fit_predict(embeddings)
            # Calculate evaluation metrics
            nmi = calculate_nmi(labels, predicted_labels)
            acc = calculate_acc(labels, predicted_labels)

        elif method == FlatClusteringMethod.DEEPECT_AUGMENTED:
            # Perform flat clustering with DeepECT and augmentation
            pass
        elif method == FlatClusteringMethod.IDEC:
            # Perform flat clustering with IDEC
            pass
    

    
    


def hierarchical(autoencoder: _AbstractAutoencoder, autoencoder_params_path: str, dataset, seed: int):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    # Load and preprocess data
    data = dataset['data']
    labels = dataset['target']

    for method in HierarchicalClusteringMethod:
        if method == HierarchicalClusteringMethod.DEEPECT:
            # Perform hierarchical clustering with DeepECT
            pass
        elif method == HierarchicalClusteringMethod.DEEPECT_AUGMENTED:
            # Perform hierarchical clustering with DeepECT and augmentation
            pass
        elif method == HierarchicalClusteringMethod.IDEC_SINGLE:
            # Perform hierarchical clustering with IDEC and single
            pass
        elif method == HierarchicalClusteringMethod.IDEC_COMPLETE:
            # Perform hierarchical clustering with IDEC and complete
            pass
        elif method == HierarchicalClusteringMethod.AE_BISECTING:
            # Perform hierarchical clustering with Autoencoder and bisection
            pass
        elif method == HierarchicalClusteringMethod.AE_SINGLE:
            # Perform hierarchical clustering with Autoencoder and single
            pass
        elif method == HierarchicalClusteringMethod.AE_COMPLETE:
            # Perform hierarchical clustering with Autoencoder and complete
            pass

# Example usage
def evaluation(init_autoencoder: _AbstractAutoencoder, autoencoder_params_path: str, data_loading_fn, seed: int):
    dataset = data_loading_fn()
    autoencoder = pretraining(init_autoencoder=init_autoencoder, autoencoder_params_path=autoencoder_params_path, dataset=dataset, seed=seed)

    flat(autoencoder=autoencoder, autoencoder_params_path=autoencoder_params_path, dataset=dataset, seed=seed)
    hierarchical(autoencoder=autoencoder, autoencoder_params_path=autoencoder_params_path, dataset=dataset, seed=seed)

# Load the MNIST dataset and evaluate flat and hierarchical clustering
dataset = load_mnist()
autoencoder_params_path = 'practical/DeepClustering/DeepECT/pretrained_autoencoders/mnist_autoencoder_pretrained.pth'
evaluation(init_autoencoder=FeedforwardAutoencoder, autoencoder_params_path=autoencoder_params_path, data_loading_fn=load_mnist, seed=42)