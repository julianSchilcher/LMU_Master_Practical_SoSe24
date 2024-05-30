import math
import os
from enum import Enum

import numpy as np
import pandas as pd
import torch
from clustpy.data import load_fmnist, load_mnist, load_reuters, load_usps
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.deep.autoencoders._abstract_autoencoder import \
    _AbstractAutoencoder
from clustpy.deep.dec import IDEC
from clustpy.metrics import unsupervised_clustering_accuracy
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils import Bunch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from practical.DeepClustering.DeepECT.baseline_hierachical.ae_plus import *
from practical.DeepClustering.DeepECT.baseline_hierachical.methods import \
    idec_hierarchical_clustpy
from practical.DeepClustering.DeepECT.deepect import DeepECT


class DatasetType(Enum):
    MNIST = "MNIST"
    FASHION_MNIST = "Fashion MNIST"
    USPS = "USPS"
    REUTERS = "Reuters"


class FlatClusteringMethod(Enum):
    DEEPECT = "DeepECT"
    DEEPECT_AUGMENTED = "DeepECT + Augmentation"
    IDEC = "IDEC"
    KMEANS = "KMeans"


class HierarchicalClusteringMethod(Enum):
    DEEPECT = "DeepECT"
    DEEPECT_AUGMENTED = "DeepECT + Augmentation"
    IDEC_SINGLE = "IDEC + Single"
    IDEC_COMPLETE = "IDEC + Complete"
    AE_BISECTING = "Autoencoder + Bisection"
    AE_SINGLE = "Autoencoder + Single"
    AE_COMPLETE = "Autoencoder + Complete"


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
    return adjusted_rand_score(true_labels, predicted_labels)


def get_max_epoch_size(data, max_iterations, batch_size):
    return math.ceil(max_iterations / (len(data) / batch_size))


def pretraining(
    init_autoencoder: _AbstractAutoencoder,
    autoencoder_params_path: str,
    dataset: Bunch,
    seed: int,
):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load and preprocess data
    data = dataset["data"]

    # TODO: Check for augmentation here

    # Initialize the autoencoder
    autoencoder: _AbstractAutoencoder = init_autoencoder(
        layers=[data.shape[1], 500, 500, 2000, 10], reusable=True
    )
    if not os.path.exists(autoencoder_params_path):
        # Train the autoencoder if parameters file does not exist
        autoencoder.to(device)
        autoencoder.fit(
            n_epochs=get_max_epoch_size(data, 80000, 256), # 50
            optimizer_params={"lr": 0.0001},# 0.001
            data=data,
            batch_size=256,
            device=device,
        ).cpu()
        autoencoder.save_parameters(autoencoder_params_path)
        print("Autoencoder pretraining complete and saved.")
    else:
        # Load the existing parameters
        autoencoder.load_parameters(autoencoder_params_path)
        print("Autoencoder parameters loaded from file.")

    return autoencoder


def flat(
    autoencoder: _AbstractAutoencoder,
    autoencoder_params_path: str,
    dataset_type: DatasetType,
    dataset: Bunch,
    seed: int,
):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load and preprocess data
    data = dataset["data"]
    labels = dataset["target"]
    results = []
    max_iterations = 50000
    batch_size = 256
    max_leaf_nodes = 12 if dataset_type == DatasetType.REUTERS else 20
    n_clusters = 4 if dataset_type == DatasetType.REUTERS else 10

    max_clustering_epochs = get_max_epoch_size(data, max_iterations, batch_size)

    for method in FlatClusteringMethod:
        # Load the autoencoder parameters
        autoencoder = autoencoder.load_parameters(autoencoder_params_path)
        autoencoder.fitted = True

        if method == FlatClusteringMethod.KMEANS:
            autoencoder.to(device)
            # Encode the data
            embeddings = (
                autoencoder.encode(
                    torch.tensor(data, dtype=torch.float32, device=device)
                )
                .detach()
                .cpu()
                .numpy()
            )
            # Perform flat clustering with KMeans
            kmeans = KMeans(
                n_clusters=n_clusters,
                init="random",
                n_init=20,
                random_state=seed,
            )
            print("fitting KMeans...")
            predicted_labels = kmeans.fit_predict(embeddings)
            print("finished fitting Kmeans")

            # Calculate evaluation metrics
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": method.value,
                    "nmi": calculate_nmi(labels, predicted_labels),
                    "acc": calculate_acc(labels, predicted_labels),
                    "ari": calculate_ari(labels, predicted_labels),
                    "seed": seed,
                }
            )
        elif method == FlatClusteringMethod.DEEPECT:
            autoencoder.to(device)
            deepect = DeepECT(
                autoencoder=autoencoder,
                clustering_optimizer_params={"lr": 1e-4, "betas": (0.9, 0.999)},
                max_leaf_nodes=max_leaf_nodes,
                random_state=np.random.RandomState(seed),
            )
            print("fitting DeepECT...")
            deepect.fit(data)
            print("finished DeepECT...")
            # Calculate evaluation metrics
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": method.value,
                    "nmi": deepect.tree_.flat_nmi(labels, n_clusters),
                    "acc": deepect.tree_.flat_accuracy(labels, n_clusters),
                    "ari": deepect.tree_.flat_ari(labels, n_clusters),
                    "dp": deepect.tree_.dendrogram_purity(labels),
                    "lp": deepect.tree_.leaf_purity(labels)[0],
                    "seed": seed,
                }
            )

        elif method == FlatClusteringMethod.DEEPECT_AUGMENTED:
            # Perform flat clustering with DeepECT and augmentation
            if dataset_type == DatasetType.REUTERS:
                # results.append(
                #     {
                #         "dataset": dataset_type.value,
                #         "method": method.value,
                #         "nmi": "-",
                #         "acc": "-",
                #         "ari": "-",
                #         "seed": "-",
                #     }
                # )
                continue

            custom_dataloaders = get_custom_dataloader_augmentations(data, dataset_type)

            deepect = DeepECT(
                autoencoder=autoencoder,
                clustering_optimizer_params={"lr": 1e-4, "betas": (0.9, 0.999)},
                max_leaf_nodes=max_leaf_nodes,
                custom_dataloaders=custom_dataloaders,
                augmentation_invariance=True,
                random_state=np.random.RandomState(seed),
            )
            print("fitting DeepECT+AUG...")
            deepect.fit(data)
            print("finished DeepECT+AUG...")
            # Calculate evaluation metrics
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": method.value,
                    "nmi": deepect.tree_.flat_nmi(labels, n_clusters),
                    "acc": deepect.tree_.flat_accuracy(labels, n_clusters),
                    "ari": deepect.tree_.flat_ari(labels, n_clusters),
                    "dp": deepect.tree_.dendrogram_purity(labels),
                    "lp": deepect.tree_.leaf_purity(labels)[0],
                    "seed": seed,
                }
            )
        elif method == FlatClusteringMethod.IDEC:
            # Perform flat clustering with IDEC
            idec = IDEC(
                n_clusters=n_clusters,
                batch_size=batch_size,
                autoencoder=autoencoder,
                clustering_optimizer_params={"lr": 1e-4, "betas": (0.9, 0.999)},
                clustering_epochs=max_clustering_epochs,
                random_state=seed,
                initial_clustering_class=KMeans,
                initial_clustering_params={
                    "init": "random",
                    "n_init": 20,
                    "random_state": seed,
                },
            )
            print("fitting IDEC...")
            idec.fit(data)
            print("finished fitting IDEC")

            # Calculate evaluation metrics
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": method.value,
                    "nmi": calculate_nmi(labels, idec.dec_labels_),
                    "acc": calculate_acc(labels, idec.dec_labels_),
                    "ari": calculate_ari(labels, idec.dec_labels_),
                    "seed": seed,
                }
            )

    df_results = pd.DataFrame(results)
    return df_results


def hierarchical(
    autoencoder: _AbstractAutoencoder,
    autoencoder_params_path: str,
    dataset_type: DatasetType,
    dataset: Bunch,
    seed: int,
):
    # Set the seed for reproducibility
    torch.manual_seed(seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load and preprocess data
    data = dataset["data"]
    labels = dataset["target"]
    results = []
    max_leaf_nodes = 12 if dataset_type == DatasetType.REUTERS else 20
    n_clusters = 4 if dataset_type == DatasetType.REUTERS else 10

    for method in HierarchicalClusteringMethod:
        # Load the autoencoder parameters
        autoencoder.load_parameters(autoencoder_params_path)
        autoencoder.to(device)
        autoencoder.fitted = True
        if method == HierarchicalClusteringMethod.AE_BISECTING:
            # Perform hierarchical clustering with Autoencoder and bisection
            print("fitting ae_bisecting...")
            dendrogram, leaf = ae_bisecting(
                data=data,
                labels=labels,
                ae_module=autoencoder,
                max_leaf_nodes=max_leaf_nodes,
                n_cluster=n_clusters,
                seed=seed,
                device=device,
            )
            print("finished ae_bisecting...")
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": method.value,
                    "dp": dendrogram,
                    "lp": leaf[0],
                    "seed": seed,
                }
            )
        elif method == HierarchicalClusteringMethod.IDEC_COMPLETE:
            autoencoder.to(device)
            print("fitting idec hierarchical...")
            # (
            #     dp_value_single,
            #     dp_value_complete,
            #     leaf_purity_value_single,
            #     leaf_purity_value_complete,
            # ) = idec_hierarchical.run_experiment(
            #     data, labels, seed, n_clusters, autoencoder, device=device
            # )
            dp_value_single, dp_value_complete, leaf_purity_value_single, leaf_purity_value_complete = idec_hierarchical_clustpy.run_idec_hierarchical(
                data, labels, seed, n_clusters, autoencoder, device=device
            )
            print("finished idec hierarchical...")
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": HierarchicalClusteringMethod.IDEC_COMPLETE,
                    "dp": dp_value_complete,
                    "lp": leaf_purity_value_complete,
                    "seed": seed,
                }
            )
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": HierarchicalClusteringMethod.IDEC_SINGLE,
                    "dp": dp_value_single,
                    "lp": leaf_purity_value_single,
                    "seed": seed,
                }
            )

        elif method == HierarchicalClusteringMethod.AE_SINGLE:
            continue
            # Perform hierarchical clustering with Autoencoder and single
            print("fitting ae_single...")
            dendrogram, leaf = ae_single(
                data=data,
                labels=labels,
                ae_module=autoencoder,
                max_leaf_nodes=max_leaf_nodes,
                n_clusters=n_clusters,
                seed=seed,
                device=device,
            )
            print("finished ae_single...")
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": method.value,
                    "dp": dendrogram,
                    "lp": leaf[0],
                    "seed": seed,
                }
            )
        elif method == HierarchicalClusteringMethod.AE_COMPLETE:
            continue
            # Perform hierarchical clustering with Autoencoder and complete
            print("fitting ae_complete...")
            dendrogram, leaf = ae_complete(
                data=data,
                labels=labels,
                ae_module=autoencoder,
                max_leaf_nodes=max_leaf_nodes,
                n_clusters=n_clusters,
                seed=seed,
                device=device,
            )
            print("fitting ae_single...")
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": method.value,
                    "dp": dendrogram,
                    "lp": leaf[0],
                    "seed": seed,
                }
            )

    df_results = pd.DataFrame(results)
    return df_results


def get_custom_dataloader_augmentations(data: np.ndarray, dataset_type: DatasetType):

    augmentation_transform = transforms.Compose(
        [
            transforms.RandomAffine(degrees=15),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255),
        ]
    )

    class Augmented_Dataset(Dataset):
        def __init__(self, original_dataset, augmentation_transform, dataset_name):
            if dataset_name == DatasetType.USPS:
                self.image_size = 16
            else:
                self.image_size = 28

            self.original_dataset = original_dataset
            self.augmentation_transform = augmentation_transform

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            original_image = self.original_dataset[idx]
            augmented_image = self.augmentation_transform(
                transforms.ToPILImage()(
                    original_image.reshape(self.image_size, self.image_size) / 255
                )
            )
            return idx, original_image, augmented_image.reshape(self.image_size**2)

    class Original_Dataset(Dataset):
        def __init__(self, original_dataset, dataset_name):
            if dataset_name == DatasetType.USPS:
                self.image_size = 16
            else:
                self.image_size = 28
            self.original_dataset = original_dataset

        def __len__(self):
            return len(self.original_dataset)

        def __getitem__(self, idx):
            original_image = self.original_dataset[idx]
            return idx, original_image.reshape(self.image_size**2)

    # Create an instance of the datasets
    augmented_dataset = Augmented_Dataset(data, augmentation_transform, dataset_type)
    original_dataset = Original_Dataset(data, dataset_type)

    # Create the dataloaders
    trainloader = DataLoader(augmented_dataset, batch_size=256, shuffle=True)
    testloader = DataLoader(original_dataset, batch_size=256, shuffle=False)

    return trainloader, testloader


def evaluate_one_seed(
    init_autoencoder: _AbstractAutoencoder,
    dataset_type: DatasetType,
    seed: int,
    autoencoder_params_path: str = None,
):
    if dataset_type == DatasetType.MNIST:
        dataset = load_mnist()
    elif dataset_type == DatasetType.FASHION_MNIST:
        dataset = load_fmnist()
    elif dataset_type == DatasetType.USPS:
        dataset = load_usps()
    elif dataset_type == DatasetType.REUTERS:
        dataset = load_reuters()

    if autoencoder_params_path is None:
        autoencoder_params_path = f"practical/DeepClustering/DeepECT/pretrained_autoencoders/{dataset['dataset_name']}_autoencoder_pretrained.pth"

    autoencoder = pretraining(
        init_autoencoder=init_autoencoder,
        autoencoder_params_path=autoencoder_params_path,
        dataset=dataset,
        seed=seed,
    )

    hierarchical_results = hierarchical(
        autoencoder=autoencoder,
        autoencoder_params_path=autoencoder_params_path,
        dataset_type=dataset_type,
        dataset=dataset,
        seed=seed,
    )

    flat_results = flat(
        autoencoder=autoencoder,
        autoencoder_params_path=autoencoder_params_path,
        dataset_type=dataset_type,
        dataset=dataset,
        seed=seed,
    )

    return flat_results, hierarchical_results

def evaluate_multiple_seeds(
    init_autoencoder: _AbstractAutoencoder,
    dataset_type: DatasetType,
    seeds: list,
    autoencoder_params_path: str = None,
):
    all_flat_results = []
    all_hierarchical_results = []

    for seed in seeds:
        flat_results, hierarchical_results = evaluate_one_seed(
            init_autoencoder=init_autoencoder,
            dataset_type=dataset_type,
            seed=seed,
            autoencoder_params_path=autoencoder_params_path,
        )
        all_flat_results.append(flat_results)
        all_hierarchical_results.append(hierarchical_results)
    
    combined_flat_results = pd.concat(all_flat_results, ignore_index=True)
    combined_hierarchical_results = pd.concat(all_hierarchical_results, ignore_index=True)

    return combined_flat_results, combined_hierarchical_results

def calculate_flat_mean_for_multiple_seeds(results: pd.DataFrame):
    results = results.groupby(["dataset", "method"]).agg({'nmi': 'mean', 'acc': 'mean', 'ari': 'mean'}).reset_index()
    return results

def calculate_hierarchical_mean_for_multiple_seeds(results: pd.DataFrame):
    results = results.groupby(["dataset", "method"]).agg({'dp': 'mean', 'lp': 'mean'}).reset_index()
    return results

if __name__ == "__main__":

    # Load the MNIST dataset and evaluate flat and hierarchical clustering
    flat_results, _ = evaluate_one_seed(
        init_autoencoder=FeedforwardAutoencoder, dataset_type=DatasetType.MNIST, seed=42
    )
    print(_)
    # evaluation(init_autoencoder=FeedforwardAutoencoder, dataset_type=DatasetType.USPS, seed=42)
    # evaluation(init_autoencoder=FeedforwardAutoencoder, dataset_type=DatasetType.REUTERS, seed=42)
    # evaluation(init_autoencoder=FeedforwardAutoencoder, dataset_type=DatasetType.FASHION_MNIST, seed=42)

    # combine all results and per experiment, do pivot to aggregate the metrics over the seeds
