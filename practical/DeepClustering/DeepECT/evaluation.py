import os
import sys

sys.path.append(os.getcwd())

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

from practical.DeepClustering import DeepECT


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

    # TODO: Check for augmentation here
    
    # Encode the data
    embeddings = autoencoder.encode(torch.tensor(data, dtype=torch.float32)).detach().numpy()
    
    # Perform flat clustering with KMeans
    kmeans = KMeans(n_clusters=10, random_state=seed)
    predicted_labels = kmeans.fit_predict(embeddings)
    
    # Calculate evaluation metrics
    nmi = calculate_nmi(labels, predicted_labels)
    acc = calculate_acc(labels, predicted_labels)
    
    return nmi, acc


def hierarchical(autoencoder: _AbstractAutoencoder, autoencoder_params_path: str, dataset, seed: int):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    
    # Load and preprocess data
    data = dataset['data']
    labels = dataset['target']

    # TODO: Check for augmentation here

    
    deepect = DeepECT(number_classes=10, autoencoder=autoencoder, max_leaf_nodes=20)
    deepect.fit(data)
    
    # Evaluate the hierarchical clustering
    embeddings = autoencoder.encode(torch.tensor(data, dtype=torch.float32)).detach().numpy()
    embeddings = torch.tensor(embeddings)  # Convert to torch tensor if needed by the metrics functions
    labels = torch.tensor(labels)
    
    dp = 0 #calculate_dendrogram_purity(deepect_model.tree, labels)
    lp = 0 #calculate_leaf_purity(deepect_model.tree, labels)
    
    return dp, lp

# Example usage
def evaluation(init_autoencoder: _AbstractAutoencoder, autoencoder_params_path: str, data_loading_fn, seed: int):
    dataset = data_loading_fn()
    autoencoder = pretraining(init_autoencoder=init_autoencoder, autoencoder_params_path=autoencoder_params_path, dataset=dataset, seed=seed)
    nmi, acc = flat(autoencoder=autoencoder, autoencoder_params_path=autoencoder_params_path, dataset=dataset, seed=seed)
    print(f"Flat Clustering - NMI: {nmi}, ACC: {acc}")
    dp, lp = hierarchical(autoencoder=autoencoder, autoencoder_params_path=autoencoder_params_path, dataset=dataset, seed=seed)
    print(f"Hierarchical Clustering - DP: {dp}, LP: {lp}")

# Load the MNIST dataset and evaluate flat and hierarchical clustering
dataset = load_mnist()
autoencoder_params_path = 'practical/DeepClustering/DeepECT/pretrained_autoencoders/mnist_autoencoder_pretrained.pth'
evaluation(init_autoencoder=FeedforwardAutoencoder, autoencoder_params_path=autoencoder_params_path, data_loading_fn=load_mnist, seed=42)