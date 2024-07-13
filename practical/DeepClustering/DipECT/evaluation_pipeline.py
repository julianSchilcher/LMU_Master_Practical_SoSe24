import datetime
import logging
import math
import multiprocessing as mp
import os
import pathlib
import sys

# Reproducability - restricting Kmeans to not parallelize
for lib in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[lib] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

sys.path.append(os.getcwd())
os.chdir(os.getcwd())

from enum import Enum
from itertools import product
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from clustpy.data import load_fmnist, load_mnist, load_reuters, load_usps
from clustpy.deep._utils import set_torch_seed
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder
from clustpy.deep.dec import IDEC
from clustpy.metrics import unsupervised_clustering_accuracy
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import minmax_scale
from sklearn.utils import Bunch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
import PIL

from practical.DeepClustering.DeepECT.deepect_ours import DeepECT as DeepECTOurs
from practical.DeepClustering.DipECT.baseline_hierachical.ae_plus import (
    ae_bisecting,
)

# please keep this format to prevent circular imports
import practical.DeepClustering.DipECT.dipect as dipect_module

# from practical.DeepClustering.DipECT.dipect import DipECT


class DatasetType(Enum):
    """
    Enumeration for dataset types.
    """

    MNIST = "MNIST"
    FASHION_MNIST = "Fashion MNIST"
    USPS = "USPS"
    REUTERS = "Reuters"


class ClusteringMethod(Enum):
    """
    Enumeration for clustering methods.
    """

    DIPECT = "DipECT"
    DIPECT_AUGMENTED = "DipECT + Augmentation"
    DEEPECT_OURS = "DeepECT (Ours)"
    DEEPECT_AUGMENTED_OURS = "DeepECT + Augmentation (Ours)"
    IDEC = "IDEC"
    KMEANS = "KMeans"
    AE_BISECTING = "Autoencoder + Bisection"


class AutoencoderType(Enum):
    """
    Enumeration for autoencoder types.
    """

    CLUSTPY_STANDARD = "ClustPy FeedForward"


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
    return unsupervised_clustering_accuracy(true_labels, predicted_labels)


def calculate_ari(true_labels, predicted_labels):
    """
    Calculate the Adjusted Rand Index (ARI) between true and predicted labels.

    Parameters
    ----------
    true_labels : array-like
        The true labels of the data points.
    predicted_labels : array-like
        The predicted labels of the data points.

    Returns
    -------
    ari : float
        The ARI score.
    """
    return adjusted_rand_score(true_labels, predicted_labels)


def get_max_epoch_size(data, max_iterations, batch_size):
    """
    Calculate the maximum epoch size.

    Parameters
    ----------
    data : np.ndarray
        The dataset.
    max_iterations : int
        The maximum number of iterations.
    batch_size : int
        The batch size.

    Returns
    -------
    max_epoch_size : int
        The maximum epoch size.
    """
    return math.ceil(max_iterations / (len(data) / batch_size))


def get_max_iterations(data, max_epochs, batch_size):
    """
    Calculate the maximum number of iterations.

    Parameters
    ----------
    data : np.ndarray or torch.utils.data.Dataset
        The dataset.
    max_epochs : int
        The maximum number of epochs.
    batch_size : int
        The batch size.

    Returns
    -------
    max_iterations : int
        The maximum number of iterations.
    """
    dataset_size = len(data)
    iterations_per_epoch = math.ceil(dataset_size / batch_size)
    max_iterations = max_epochs * iterations_per_epoch
    return max_iterations


def pretraining(
    autoencoder_type: AutoencoderType,
    autoencoder_params_path: pathlib.Path,
    dataset: Bunch,
    seed: int,
    embedding_dim: int,
):
    """
    Pretrain an autoencoder.

    Parameters
    ----------
    autoencoder_type : AutoencoderType
        The type of autoencoder.
    autoencoder_params_path : str
        The path to save the autoencoder parameters.
    dataset : Bunch
        The dataset.
    seed : int
        The random seed for reproducibility.
    embedding_dim : int
        The dimension of the embedding.

    Returns
    -------
    autoencoder : _AbstractAutoencoder
        The pretrained autoencoder.
    """
    set_torch_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["SKLEARN_SEED"] = str(seed)
    torch.use_deterministic_algorithms(mode=True)
    # Reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id):
        set_torch_seed(seed)

    data = torch.tensor(dataset["data"], dtype=torch.float32)
    dataloader = DataLoader(
        TensorDataset(data),
        batch_size=256,
        shuffle=True,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    if not autoencoder_params_path.exists():
        # logging config
        logging.root.handlers = []
        log_path = pathlib.Path(
            f"practical/DeepClustering/DipECT/pretrained_autoencoders_log/{dataset['dataset_name']}_{autoencoder_type.name}_{embedding_dim}_{seed}.txt"
        )
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout),
            ],
        )
        if autoencoder_type == AutoencoderType.CLUSTPY_STANDARD:
            autoencoder = FeedforwardAutoencoder(
                layers=[data.shape[1], 500, 500, 2000, embedding_dim]
            )
            autoencoder.to(device)
            autoencoder.fit(
                n_epochs=get_max_epoch_size(
                    data, 27400, 256
                ),  # 27400 is the max iterations that we do Mnist for
                optimizer_params={"lr": 1e-3},
                dataloader=dataloader,
                batch_size=256,
                device=device,
                print_step=1,
            )
            autoencoder.fitted = True
            autoencoder.cpu().save_parameters(autoencoder_params_path)
        print("Autoencoder pretraining complete and saved.")
    else:
        if autoencoder_type == AutoencoderType.CLUSTPY_STANDARD:
            autoencoder = FeedforwardAutoencoder(
                layers=[data.shape[1], 500, 500, 2000, embedding_dim], reusable=True
            )
            autoencoder.load_parameters(autoencoder_params_path)
        print("Autoencoder parameters loaded from file.")

    return autoencoder


class Original_Dataset(Dataset):
    """
    Dataset class for original dataset.

    Parameters
    ----------
    original_dataset : np.ndarray
        The original dataset.

    Methods
    -------
    __len__():
        Returns the length of the dataset.
    __getitem__(idx):
        Returns the item at the given index.
    """

    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_image = self.original_dataset[idx]
        return idx, original_image


def fit(
    autoencoder: _AbstractAutoencoder,
    autoencoder_params_path: str,
    dataset_type: DatasetType,
    dataset: Bunch,
    seed: int,
    autoencoder_type: AutoencoderType,
    embedding_dim: int,
    can_use_workers: bool = False,
):
    """
    Fit the autoencoder and perform clustering.

    Parameters
    ----------
    autoencoder : _AbstractAutoencoder
        The autoencoder to be fitted.
    autoencoder_params_path : str
        The path to the autoencoder parameters.
    dataset_type : DatasetType
        The type of dataset.
    dataset : Bunch
        The dataset.
    seed : int
        The random seed for reproducibility.
    autoencoder_type : AutoencoderType
        The type of autoencoder.
    embedding_dim : int
        The dimension of the embedding.
    can_use_workers : bool, optional
        Whether to use multiprocessing, by default False.

    Returns
    -------
    pd.DataFrame
        The results of the clustering.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load and preprocess data
    data = dataset["data"]
    labels = dataset["target"]
    batch_size = 256
    max_leaf_nodes = 12 if dataset_type == DatasetType.REUTERS else 20
    n_clusters = 4 if dataset_type == DatasetType.REUTERS else 10

    max_iterations = 16440  # using 60 epochs on MNIST
    max_clustering_epochs = get_max_epoch_size(data, max_iterations, batch_size)

    results = []
    for method in ClusteringMethod:
        # Save path
        result_path = pathlib.Path(
            f"practical/DeepClustering/DipECT/results/{dataset['dataset_name']}_{autoencoder_type.name}_{embedding_dim}_{method.name}_{seed}.pq"
        )
        if os.path.exists(result_path):
            results.append(pd.read_parquet(result_path))
            continue

        # Reproducibility
        generator = torch.Generator()
        generator.manual_seed(seed)
        set_torch_seed(seed)
        for lib in [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ]:
            os.environ[lib] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["KMP_INIT_AT_FORK"] = "FALSE"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

        def seed_worker(worker_id):
            set_torch_seed(seed)

        # Load the autoencoder parameters
        autoencoder = autoencoder.load_parameters(autoencoder_params_path).to(device)
        # Load dataloaders
        dataloaders = (
            DataLoader(
                Original_Dataset(data),
                batch_size=batch_size,
                shuffle=True,
                num_workers=5 if can_use_workers else 0,
                prefetch_factor=200 if can_use_workers else None,
                generator=generator,
                worker_init_fn=seed_worker,
            ),
            DataLoader(
                Original_Dataset(data),
                batch_size=batch_size,
                shuffle=False,
                num_workers=5 if can_use_workers else 0,
                prefetch_factor=200 if can_use_workers else None,
                generator=generator,
                worker_init_fn=seed_worker,
            ),
        )

        # autoencoder save path
        autoencoder_save_path = pathlib.Path(
            f"practical/DeepClustering/DipECT/results_autoencoder/{dataset['dataset_name']}_{autoencoder_type.name}_{embedding_dim}_{method.name}_{seed}.pth"
        )

        # logging config
        logging.root.handlers = []
        log_path = pathlib.Path(
            f"practical/DeepClustering/DipECT/results_log/{dataset['dataset_name']}_{autoencoder_type.name}_{embedding_dim}_{method.name}_{seed}.txt"
        )
        logging.basicConfig(
            level=logging.INFO,
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout),
            ],
        )
        # if not yet evaluated, fit + get results
        if method == ClusteringMethod.KMEANS:
            autoencoder.to(device)
            # Encode the data
            embeddings = []
            for batch in dataloaders[1]:
                embeddings.append(
                    autoencoder.encode(batch[1].to(device)).detach().cpu().numpy()
                )
            # Perform flat clustering with KMeans
            kmeans = KMeans(
                n_clusters=n_clusters,
                n_init=20,
                random_state=np.random.RandomState(seed),
            )
            print("fitting KMeans...")
            predicted_labels = kmeans.fit_predict(
                np.concatenate(embeddings, dtype=np.float32)
            )
            print("finished fitting Kmeans")

            # Calculate evaluation metrics
            result_df = pd.DataFrame(
                [
                    {
                        "autoencoder": autoencoder_type.value,
                        "embedding_dim": embedding_dim,
                        "dataset": dataset_type.value,
                        "method": method.value,
                        "nmi": calculate_nmi(labels, predicted_labels),
                        "acc": calculate_acc(labels, predicted_labels),
                        "ari": calculate_ari(labels, predicted_labels),
                        "seed": seed,
                    }
                ]
            )

        elif method == ClusteringMethod.DEEPECT_OURS:
            autoencoder.to(device)
            deepect = DeepECTOurs(
                max_iterations=max_iterations,
                autoencoder=autoencoder,
                clustering_optimizer_params={"lr": 1e-4, "betas": (0.9, 0.999)},
                max_leaf_nodes=max_leaf_nodes,
                random_state=np.random.RandomState(seed),
                custom_dataloaders=dataloaders,
            )
            print(f"fitting {method.name}...")
            deepect.fit(data)
            autoencoder = deepect.autoencoder
            print(f"finished {method.name}...")
            try:
                # Calculate evaluation metrics
                result_df = pd.DataFrame(
                    [
                        {
                            "autoencoder": autoencoder_type.value,
                            "embedding_dim": embedding_dim,
                            "dataset": dataset_type.value,
                            "method": method.value,
                            "nmi": deepect.tree_.flat_nmi(labels, n_clusters),
                            "acc": deepect.tree_.flat_accuracy(labels, n_clusters),
                            "ari": deepect.tree_.flat_ari(labels, n_clusters),
                            "nmi-kmeans": deepect.tree_.flat_nmi_kmeans(
                                labels, n_clusters
                            ),
                            "acc-kmeans": deepect.tree_.flat_accuracy_kmeans(
                                labels, n_clusters
                            ),
                            "ari-kmeans": deepect.tree_.flat_ari_kmeans(
                                labels, n_clusters
                            ),
                            "dp": deepect.tree_.dendrogram_purity(labels),
                            "lp": deepect.tree_.leaf_purity(labels)[0],
                            "seed": seed,
                        }
                    ]
                )
            except:
                result_df = pd.DataFrame(
                    [
                        {
                            "autoencoder": autoencoder_type.value,
                            "embedding_dim": embedding_dim,
                            "dataset": dataset_type.value,
                            "method": method.value,
                            "nmi": np.nan,
                            "acc": np.nan,
                            "ari": np.nan,
                            "nmi-kmeans": deepect.tree_.flat_nmi_kmeans(
                                labels, n_clusters
                            ),
                            "acc-kmeans": deepect.tree_.flat_accuracy_kmeans(
                                labels, n_clusters
                            ),
                            "ari-kmeans": deepect.tree_.flat_ari_kmeans(
                                labels, n_clusters
                            ),
                            "dp": deepect.tree_.dendrogram_purity(labels),
                            "lp": deepect.tree_.leaf_purity(labels)[0],
                            "seed": seed,
                        }
                    ]
                )
        elif method == ClusteringMethod.DEEPECT_AUGMENTED_OURS:
            # Perform flat clustering with DeepECT and augmentation
            if dataset_type == DatasetType.REUTERS:
                continue
            autoencoder.to(device)

            custom_dataloaders = get_custom_dataloader_augmentations(
                data, dataset_type, seed
            )

            deepect = DeepECTOurs(
                max_iterations=max_iterations,
                autoencoder=autoencoder,
                clustering_optimizer_params={"lr": 1e-4, "betas": (0.9, 0.999)},
                max_leaf_nodes=max_leaf_nodes,
                custom_dataloaders=custom_dataloaders,
                augmentation_invariance=True,
                random_state=np.random.RandomState(seed),
            )
            print(f"fitting {method.name}...")
            deepect.fit(data)
            autoencoder = deepect.autoencoder
            print(f"finished {method.name}...")
            # Calculate evaluation metrics
            try:
                # Calculate evaluation metrics
                result_df = pd.DataFrame(
                    [
                        {
                            "autoencoder": autoencoder_type.value,
                            "embedding_dim": embedding_dim,
                            "dataset": dataset_type.value,
                            "method": method.value,
                            "nmi": deepect.tree_.flat_nmi(labels, n_clusters),
                            "acc": deepect.tree_.flat_accuracy(labels, n_clusters),
                            "ari": deepect.tree_.flat_ari(labels, n_clusters),
                            "nmi-kmeans": deepect.tree_.flat_nmi_kmeans(
                                labels, n_clusters
                            ),
                            "acc-kmeans": deepect.tree_.flat_accuracy_kmeans(
                                labels, n_clusters
                            ),
                            "ari-kmeans": deepect.tree_.flat_ari_kmeans(
                                labels, n_clusters
                            ),
                            "dp": deepect.tree_.dendrogram_purity(labels),
                            "lp": deepect.tree_.leaf_purity(labels)[0],
                            "seed": seed,
                        }
                    ]
                )
            except:
                result_df = pd.DataFrame(
                    [
                        {
                            "autoencoder": autoencoder_type.value,
                            "embedding_dim": embedding_dim,
                            "dataset": dataset_type.value,
                            "method": method.value,
                            "nmi": np.nan,
                            "acc": np.nan,
                            "ari": np.nan,
                            "nmi-kmeans": deepect.tree_.flat_nmi_kmeans(
                                labels, n_clusters
                            ),
                            "acc-kmeans": deepect.tree_.flat_accuracy_kmeans(
                                labels, n_clusters
                            ),
                            "ari-kmeans": deepect.tree_.flat_ari_kmeans(
                                labels, n_clusters
                            ),
                            "dp": deepect.tree_.dendrogram_purity(labels),
                            "lp": deepect.tree_.leaf_purity(labels)[0],
                            "seed": seed,
                        }
                    ]
                )
        elif method == ClusteringMethod.DIPECT:
            autoencoder.to(device)
            dipect = dipect_module.DipECT(
                batch_size=256,
                autoencoder=autoencoder,
                random_state=np.random.RandomState(seed),
                logging_active=True,
                clustering_n_epochs=max_clustering_epochs,
                pruning_threshold=len(data) // 35,  # 2000 for MNIST
                tree_growth_min_cluster_size=len(data) // 35,
                tree_growth_frequency=548 / (len(data) / batch_size),  # 2.0 for MNIST
                custom_dataloaders=dataloaders,
            )
            print(f"fitting {method.name}...")
            dipect.fit_predict(data)
            autoencoder = dipect.autoencoder
            print(f"finished {method.name}...")
            try:
                # Calculate evaluation metrics
                result_df = pd.DataFrame(
                    [
                        {
                            "autoencoder": autoencoder_type.value,
                            "embedding_dim": embedding_dim,
                            "dataset": dataset_type.value,
                            "method": method.value,
                            "nmi": dipect.tree_.flat_nmi(labels, n_clusters),
                            "acc": dipect.tree_.flat_accuracy(labels, n_clusters),
                            "ari": dipect.tree_.flat_ari(labels, n_clusters),
                            "nmi-kmeans": dipect.tree_.flat_nmi_kmeans(
                                labels, n_clusters
                            ),
                            "acc-kmeans": dipect.tree_.flat_accuracy_kmeans(
                                labels, n_clusters
                            ),
                            "ari-kmeans": dipect.tree_.flat_ari_kmeans(
                                labels, n_clusters
                            ),
                            "dp": dipect.tree_.dendrogram_purity(labels),
                            "lp": dipect.tree_.leaf_purity(labels)[0],
                            "seed": seed,
                        }
                    ]
                )
            except:
                result_df = pd.DataFrame(
                    [
                        {
                            "autoencoder": autoencoder_type.value,
                            "embedding_dim": embedding_dim,
                            "dataset": dataset_type.value,
                            "method": method.value,
                            "nmi": np.nan,
                            "acc": np.nan,
                            "ari": np.nan,
                            "nmi-kmeans": dipect.tree_.flat_nmi_kmeans(
                                labels, n_clusters
                            ),
                            "acc-kmeans": dipect.tree_.flat_accuracy_kmeans(
                                labels, n_clusters
                            ),
                            "ari-kmeans": dipect.tree_.flat_ari_kmeans(
                                labels, n_clusters
                            ),
                            "dp": dipect.tree_.dendrogram_purity(labels),
                            "lp": dipect.tree_.leaf_purity(labels)[0],
                            "seed": seed,
                        }
                    ]
                )

        elif method == ClusteringMethod.DIPECT_AUGMENTED:
            if dataset_type == DatasetType.REUTERS:
                continue
            autoencoder.to(device)

            custom_dataloaders = get_custom_dataloader_augmentations(
                data, dataset_type, seed
            )

            dipect = dipect_module.DipECT(
                batch_size=256,
                autoencoder=autoencoder,
                random_state=np.random.RandomState(seed),
                logging_active=True,
                clustering_n_epochs=max_clustering_epochs,
                pruning_threshold=len(data) // 35,  # 2000 for MNIST
                tree_growth_min_cluster_size=len(data) // 35,
                tree_growth_frequency=548 / (len(data) / batch_size),  # 2.0 for MNIST
                augmentation_invariance=True,
                custom_dataloaders=custom_dataloaders,
            )
            print(f"fitting {method.name}...")
            dipect.fit_predict(data)
            autoencoder = dipect.autoencoder
            print(f"finished {method.name}...")
            try:
                # Calculate evaluation metrics
                result_df = pd.DataFrame(
                    [
                        {
                            "autoencoder": autoencoder_type.value,
                            "embedding_dim": embedding_dim,
                            "dataset": dataset_type.value,
                            "method": method.value,
                            "nmi": dipect.tree_.flat_nmi(labels, n_clusters),
                            "acc": dipect.tree_.flat_accuracy(labels, n_clusters),
                            "ari": dipect.tree_.flat_ari(labels, n_clusters),
                            "nmi-kmeans": dipect.tree_.flat_nmi_kmeans(
                                labels, n_clusters
                            ),
                            "acc-kmeans": dipect.tree_.flat_accuracy_kmeans(
                                labels, n_clusters
                            ),
                            "ari-kmeans": dipect.tree_.flat_ari_kmeans(
                                labels, n_clusters
                            ),
                            "dp": dipect.tree_.dendrogram_purity(labels),
                            "lp": dipect.tree_.leaf_purity(labels)[0],
                            "seed": seed,
                        }
                    ]
                )
            except:
                result_df = pd.DataFrame(
                    [
                        {
                            "autoencoder": autoencoder_type.value,
                            "embedding_dim": embedding_dim,
                            "dataset": dataset_type.value,
                            "method": method.value,
                            "nmi": np.nan,
                            "acc": np.nan,
                            "ari": np.nan,
                            "nmi-kmeans": dipect.tree_.flat_nmi_kmeans(
                                labels, n_clusters
                            ),
                            "acc-kmeans": dipect.tree_.flat_accuracy_kmeans(
                                labels, n_clusters
                            ),
                            "ari-kmeans": dipect.tree_.flat_ari_kmeans(
                                labels, n_clusters
                            ),
                            "dp": dipect.tree_.dendrogram_purity(labels),
                            "lp": dipect.tree_.leaf_purity(labels)[0],
                            "seed": seed,
                        }
                    ]
                )

        elif method == ClusteringMethod.IDEC:
            # Perform flat clustering with IDEC
            idec = IDEC(
                n_clusters=n_clusters,
                batch_size=batch_size,
                neural_network=autoencoder.to(device),
                clustering_optimizer_params={"lr": 1e-4},
                clustering_loss_weight=10.0,  # needs to be 10 to weight cluster loss 10x higher than autoencoder loss like in the paper
                clustering_epochs=max_clustering_epochs,
                random_state=seed,
                device=device,
                initial_clustering_class=KMeans,
                initial_clustering_params={
                    "n_init": 20,
                    "random_state": np.random.RandomState(seed),
                },
                custom_dataloaders=dataloaders,
            )
            print("fitting IDEC...")
            idec.fit(data)
            autoencoder = idec.neural_network
            print("finished fitting IDEC")

            # Calculate evaluation metrics
            result_df = pd.DataFrame(
                [
                    {
                        "autoencoder": autoencoder_type.value,
                        "embedding_dim": embedding_dim,
                        "dataset": dataset_type.value,
                        "method": method.value,
                        "nmi": calculate_nmi(labels, idec.dec_labels_),
                        "acc": calculate_acc(labels, idec.dec_labels_),
                        "ari": calculate_ari(labels, idec.dec_labels_),
                        "seed": seed,
                    }
                ]
            )
        elif method == ClusteringMethod.AE_BISECTING:
            # Perform hierarchical clustering with Autoencoder and bisection
            print("fitting ae_bisecting...")

            dendrogram, leaf = ae_bisecting(
                dataloader=dataloaders[1],
                labels=labels,
                ae_module=autoencoder,
                max_leaf_nodes=max_leaf_nodes,
                device=device,
            )
            print("finished ae_bisecting...")
            result_df = pd.DataFrame(
                [
                    {
                        "autoencoder": autoencoder_type.value,
                        "embedding_dim": embedding_dim,
                        "dataset": dataset_type.value,
                        "method": method.value,
                        "dp": dendrogram,
                        "lp": leaf[0],
                        "seed": seed,
                    }
                ]
            )

        result_df.to_parquet(result_path)
        results.append(result_df)
        autoencoder.cpu().save_parameters(autoencoder_save_path)

    return pd.concat(results, axis=0, ignore_index=True)


def shuffle_dataset(data, labels):
    """
    Shuffle the dataset.

    Parameters
    ----------
    data : np.ndarray
        The data to be shuffled.
    labels : np.ndarray
        The labels to be shuffled.

    Returns
    -------
    tuple
        The shuffled data and labels.
    """
    shuffled_indices = np.random.permutation(len(data))
    shuffled_x = data[shuffled_indices, :]
    if labels is not None:
        shuffled_y = labels[shuffled_indices]
        return shuffled_x, shuffled_y
    else:
        return shuffled_x


def get_custom_dataloader_augmentations(
    data: np.ndarray,
    dataset_type: DatasetType,
    seed: int,
    can_use_workers: bool = False,
):
    """
    Get custom dataloaders with augmentations.

    Parameters
    ----------
    data : np.ndarray
        The dataset.
    dataset_type : DatasetType
        The type of dataset.
    can_use_workers : bool, optional
        Whether to use multiprocessing, by default False.

    Returns
    -------
    tuple
        The train and test dataloaders.
    """
    # raise NotImplementedError("Not implemented for dipect evaluation.")
    degrees = (-15, 15)
    translation = (
        0.14 if dataset_type == DatasetType.USPS else 0.08,
        0.14 if dataset_type == DatasetType.USPS else 0.08,
    )

    image_min_value = np.min(data)
    image_max_value = np.max(data)
    image_size = 16 if dataset_type == DatasetType.USPS else 28

    augmentation_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x - image_min_value),
            transforms.Lambda(lambda x: x / image_max_value),  # [0,1]
            transforms.Lambda(lambda x: x.reshape(image_size, image_size)),
            transforms.ToPILImage(),  # [0,255]
            transforms.RandomAffine(
                degrees=degrees,
                shear=degrees,
                translate=translation,
                interpolation=PIL.Image.BILINEAR,
            ),
            transforms.ToTensor(),  # back to [0,1] again
            transforms.Lambda(lambda x: x.reshape(image_size**2)),
            transforms.Lambda(
                lambda x: x * image_max_value
            ),  # back to original data range
            transforms.Lambda(lambda x: x + image_min_value),
        ]
    )

    class Augmented_Dataset(Dataset):
        """
        Dataset class for augmented dataset.

        Parameters
        ----------
        original_dataset : np.ndarray
            The original dataset.
        augmentation_transform : transforms.Compose
            The augmentation transform.

        Methods
        -------
        __len__():
            Returns the length of the dataset.
        __getitem__(idx):
            Returns the item at the given index.
        """

        def __init__(self, original_dataset, augmentation_transform):
            self.original_dataset = original_dataset
            self.augmentation_transform = augmentation_transform

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            original_image = self.original_dataset[idx]
            augmented_image = self.augmentation_transform(original_image)
            return idx, original_image, augmented_image

    # Create an instance of the datasets
    original_dataset = Original_Dataset(data)
    augmented_dataset = Augmented_Dataset(data, augmentation_transform)

    generator = torch.Generator()
    generator.manual_seed(seed)

    def workers_init(worker_id):
        set_torch_seed(seed)

    # Create the dataloaders
    trainloader = DataLoader(
        augmented_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=5 if can_use_workers else 0,
        prefetch_factor=200 if can_use_workers else None,
        worker_init_fn=workers_init,
        generator=generator,
    )
    testloader = DataLoader(
        original_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=5 if can_use_workers else 0,
        prefetch_factor=200 if can_use_workers else None,
        worker_init_fn=workers_init,
        generator=generator,
    )

    return trainloader, testloader


def get_dataset(dataset_type: DatasetType):
    """
    Get the dataset based on the dataset type.

    Parameters
    ----------
    dataset_type : DatasetType
        The type of dataset.

    Returns
    -------
    Bunch
        The dataset.
    """
    if dataset_type == DatasetType.MNIST:
        # data from paper, normalized to [0,1]
        dataset = load_mnist()
        dataset["data"] = np.asarray(
            minmax_scale(dataset["data"], feature_range=(0, 1)), dtype=np.float32
        )
    elif dataset_type == DatasetType.FASHION_MNIST:
        # paper also used fashionmnist from pytorch and scales to [0,1]
        dataset = load_fmnist()
        dataset["data"] = np.asarray(
            minmax_scale(dataset["data"], feature_range=(0, 1)), dtype=np.float32
        )
    elif dataset_type == DatasetType.USPS:
        # usps data from paper normalized to [0,1]
        dataset = load_usps()
        dataset["data"] = np.asarray(
            minmax_scale(dataset["data"], feature_range=(0, 1)),
            dtype=np.float32,
        )
    else:
        # reuters is loaded the same way in clustpy as in the paper and scaled to [0,1]
        dataset = load_reuters()
        dataset["data"] = np.asarray(
            minmax_scale(dataset["data"], feature_range=(0, 1)), dtype=np.float32
        )
    return dataset


def get_autoencoder_path(
    autoencoder_type: AutoencoderType,
    autoencoder_params_path: Union[str, None],
    dataset: Bunch,
    embedding_dim: int,
    seed: int,
) -> pathlib.Path:
    """
    Get the path to the autoencoder parameters.

    Parameters
    ----------
    autoencoder_type : AutoencoderType
        The type of autoencoder.
    autoencoder_params_path : str, optional
        The path to the autoencoder parameters, by default None.
    dataset : Bunch
        The dataset.
    embedding_dim : int
        The dimension of the embedding.
    seed : int
        The random seed for reproducibility.

    Returns
    -------
    str
        The path to the autoencoder parameters.
    """
    if (
        autoencoder_params_path is None
        and autoencoder_type == AutoencoderType.CLUSTPY_STANDARD
    ):
        params_path = pathlib.Path(
            f"practical/DeepClustering/DipECT/pretrained_autoencoders/{dataset['dataset_name']}_autoencoder_{embedding_dim}_pretrained_{seed}.pth"
        )
    else:
        params_path = pathlib.Path(autoencoder_params_path)
    return params_path.resolve()


# Example usage
def evaluate(
    autoencoder_type: AutoencoderType,
    dataset_type: DatasetType,
    seed: int,
    autoencoder_params_path: str = None,
    embedding_dim: int = 10,
    can_use_workers: bool = True,
) -> pd.DataFrame:
    """
    Evaluate the clustering performance.

    Parameters
    ----------
    autoencoder_type : AutoencoderType
        The type of autoencoder.
    dataset_type : DatasetType
        The type of dataset.
    seed : int
        The random seed for reproducibility.
    autoencoder_params_path : str, optional
        The path to the autoencoder parameters, by default None.
    embedding_dim : int, optional
        The dimension of the embedding, by default 10.
    can_use_workers : bool, optional
        Whether to use multiprocessing, by default True.

    Returns
    -------
    pd.DataFrame
        The results of the clustering.
    """
    start = datetime.datetime.now()

    print(
        f"-------------------------------------------Run: {dataset_type.name}_{autoencoder_type.name}_{embedding_dim}_{seed}"
    )
    # Reproducability - restricting Kmeans to not parallelize
    for lib in [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]:
        os.environ[lib] = "1"
    os.environ["KMP_INIT_AT_FORK"] = "FALSE"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["SKLEARN_SEED"] = str(seed)
    torch.use_deterministic_algorithms(mode=True)

    dataset = get_dataset(dataset_type)

    autoencoder_params_path = get_autoencoder_path(
        autoencoder_type, autoencoder_params_path, dataset, embedding_dim, seed
    )

    autoencoder = pretraining(
        autoencoder_type=autoencoder_type,
        autoencoder_params_path=autoencoder_params_path,
        dataset=dataset,
        seed=seed,
        embedding_dim=embedding_dim,
    )

    results = fit(
        autoencoder=autoencoder,
        autoencoder_params_path=autoencoder_params_path,
        dataset_type=dataset_type,
        dataset=dataset,
        seed=seed,
        autoencoder_type=autoencoder_type,
        embedding_dim=embedding_dim,
        can_use_workers=can_use_workers,
    )
    print(results)
    print(
        f"-------------------------------------------Time needed: {(datetime.datetime.now()-start).total_seconds()/60}min"
    )
    return results


def evaluate_multiple_seeds(
    autoencoder_type: AutoencoderType,
    dataset_type: DatasetType,
    seeds: List[int],
    embedding_dims: List[int] = [10],
    autoencoder_params_path: str = None,
):
    """
    Evaluate the clustering performance for multiple seeds.

    Parameters
    ----------
    autoencoder_type : AutoencoderType
        The type of autoencoder.
    dataset_type : DatasetType
        The type of dataset.
    seeds : List[int]
        The list of random seeds for reproducibility.
    embedding_dims : List[int], optional
        The list of embedding dimensions, by default [10].
    autoencoder_params_path : str, optional
        The path to the autoencoder parameters, by default None.

    Returns
    -------
    pd.DataFrame
        The results of the clustering.
    """
    results = []

    for seed, embedding_dim in product(seeds, embedding_dims):
        result = evaluate(
            autoencoder_type=autoencoder_type,
            dataset_type=dataset_type,
            seed=seed,
            embedding_dim=embedding_dim,
            autoencoder_params_path=autoencoder_params_path,
            can_use_workers=True,
        )
        results.append(result)
    return pd.concat(results, ignore_index=True)


def calculate_flat_mean_for_multiple_seeds(results: pd.DataFrame):
    """
    Calculate the mean clustering metrics for flat clustering methods across multiple seeds.

    Parameters
    ----------
    results : pd.DataFrame
        The results of the clustering.

    Returns
    -------
    pd.DataFrame
        The mean clustering metrics.
    """
    results = (
        results.groupby(["autoencoder", "embedding_dim", "dataset", "method"])
        .agg({"nmi": "mean", "acc": "mean", "ari": "mean"})
        .reset_index()
    )
    return results


def calculate_hierarchical_mean_for_multiple_seeds(results: pd.DataFrame):
    """
    Calculate the mean clustering metrics for hierarchical clustering methods across multiple seeds.

    Parameters
    ----------
    results : pd.DataFrame
        The results of the clustering.

    Returns
    -------
    pd.DataFrame
        The mean clustering metrics.
    """
    results = (
        results.groupby(["autoencoder", "embedding_dim", "dataset", "method"])
        .agg({"dp": "mean", "lp": "mean"})
        .reset_index()
    )
    return results


def pretraining_with_data_load(
    autoencoder_type: AutoencoderType,
    autoencoder_params_path: str,
    dataset_type: DatasetType,
    seed: int,
    embedding_dim: int,
):
    """
    Pretrain the autoencoder with data load.

    Parameters
    ----------
    autoencoder_type : AutoencoderType
        The type of autoencoder.
    autoencoder_params_path : str
        The path to save the autoencoder parameters.
    dataset_type : DatasetType
        The type of dataset.
    seed : int
        The random seed for reproducibility.
    embedding_dim : int
        The dimension of the embedding.
    """
    start = datetime.datetime.now()
    dataset = get_dataset(dataset_type)

    autoencoder_params_path = get_autoencoder_path(
        autoencoder_type, autoencoder_params_path, dataset, embedding_dim, seed
    )

    print(
        f"---------------------------------------------{autoencoder_type.name} {dataset_type.name} {seed}"
    )

    autoencoder = pretraining(
        autoencoder_type=autoencoder_type,
        autoencoder_params_path=autoencoder_params_path,
        dataset=dataset,
        seed=seed,
        embedding_dim=embedding_dim,
    )
    del autoencoder
    torch.cuda.memory.empty_cache()
    print(
        f"-------------------------------------------Time needed: {(datetime.datetime.now()-start).total_seconds()/60}min"
    )


def pretrain_for_multiple_seeds(seeds: List[int], embedding_dims=[10], worker_num=1):
    """
    Pretrain the autoencoder for multiple seeds.

    Parameters
    ----------
    seeds : List[int]
        The list of random seeds for reproducibility.
    embedding_dims : list, optional
        The list of embedding dimensions, by default [10].
    worker_num : int, optional
        The number of workers for multiprocessing, by default 1.

    Returns
    -------
    list
        The result of the pretraining.
    """
    all_autoencoders = list(
        product(AutoencoderType, [None], DatasetType, seeds, embedding_dims)
    )
    with mp.Pool(processes=worker_num) as pool:
        result = pool.starmap(pretraining_with_data_load, all_autoencoders)
    return result


def load_precomputed_results(
    dataset_type: DatasetType,
    autoencoder_type: AutoencoderType,
    seeds: List[int],
    embedding_dims: List[int] = [10],
):
    results = []
    dataset = get_dataset(dataset_type)
    for method, seed, embedding_dim in product(ClusteringMethod, seeds, embedding_dims):
        # Save path
        result_path = pathlib.Path(
            f"practical/DeepClustering/DipECT/results/{dataset['dataset_name']}_{autoencoder_type.name}_{embedding_dim}_{method.name}_{seed}.pq"
        )
        if os.path.exists(result_path):
            results.append(pd.read_parquet(result_path))
            continue
    return pd.concat(results, axis=0, ignore_index=True)


if __name__ == "__main__":
    seeds = [21, 42, 63]
    embedding_dims = [10]
    worker_num = 2
    pretrain_for_multiple_seeds(
        seeds, embedding_dims=embedding_dims, worker_num=worker_num
    )
    all_autoencoders = list(
        product(AutoencoderType, DatasetType, seeds, [None], embedding_dims, [None])
    )
    with mp.Pool(processes=worker_num) as pool:
        result = pool.starmap(evaluate, all_autoencoders)
    # compute autoencoder+complete linkage
    for ae_type, dataset_type, seed, ae_path, embedding_dim, _ in all_autoencoders:
        evaluate(ae_type, dataset_type, seed, ae_path, embedding_dim)
