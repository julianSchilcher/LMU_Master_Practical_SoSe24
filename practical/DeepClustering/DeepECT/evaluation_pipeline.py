import datetime
import math
import os
from enum import Enum
import sys
from pathlib import Path
sys.path.append(os.getcwd())
import time
from typing import List, Tuple, Union
import PIL
import numpy as np
import pandas as pd
from scipy import io
from sklearn.utils import Bunch
from sklearn.datasets import fetch_rcv1
import torch
from torch.utils.data import TensorDataset, DataLoader
from clustpy.data import load_fmnist, load_mnist, load_reuters, load_usps
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from clustpy.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder
from clustpy.deep.dec import IDEC
from clustpy.deep._utils import set_torch_seed
from clustpy.metrics import unsupervised_clustering_accuracy
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.utils import Bunch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import multiprocessing as mp
from itertools import product
from practical.DeepClustering.DeepECT.evaluation.experiments.pre_training.vae.stacked_ae import (
    stacked_ae,
)

from practical.DeepClustering.DeepECT.deepect_adjusted import DeepECT
from practical.DeepClustering.DeepECT.baseline_hierachical.ae_plus import *
from practical.DeepClustering.DeepECT.baseline_hierachical.methods import (
    idec_hierarchical_clustpy,
)


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


class AutoencoderType(Enum):
    CLUSTPY_STANDARD = "ClustPy FeedForward"
    DEEPECT_STACKED_AE = "Stacked AE from DeepECT"


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
    autoencoder_type: AutoencoderType,
    autoencoder_params_path,
    dataset,
    seed,
    embedding_dim: int,
):
    set_torch_seed(np.random.RandomState(seed))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    data = dataset["data"]
    data = torch.tensor(data, dtype=torch.float32)

    if not os.path.exists(autoencoder_params_path):
        if autoencoder_type == AutoencoderType.CLUSTPY_STANDARD:
            autoencoder = FeedforwardAutoencoder(
                layers=[data.shape[1], 500, 500, 2000, embedding_dim]
            )
            autoencoder.to(device)
            autoencoder.fit(
                n_epochs=get_max_epoch_size(data, 50000, 256),
                optimizer_params={},
                data=data,
                batch_size=256,
                device=device,
                print_step=1,
            )
            autoencoder.fitted = True
            autoencoder.cpu().save_parameters(autoencoder_params_path)
        elif autoencoder_type == AutoencoderType.DEEPECT_STACKED_AE:
            weight_initalizer = torch.nn.init.xavier_normal_
            steps_per_layer = 20000
            refine_training_steps = 50000

            def add_noise(batch):
                mask = torch.empty(batch.shape, device=batch.device).bernoulli_(0.8)
                return batch * mask

            autoencoder = stacked_ae(
                data.shape[1],
                [500, 500, 2000, embedding_dim],
                weight_initalizer,
                activation_fn=lambda x: torch.nn.functional.relu(x),
                optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=0.0001),
            ).to(device)
            autoencoder.pretrain(
                DataLoader(TensorDataset(data), batch_size=256, shuffle=True),
                steps_per_layer,
                corruption_fn=add_noise,
            )
            autoencoder.refine_training(
                DataLoader(TensorDataset(data), batch_size=256, shuffle=True),
                refine_training_steps,
                corruption_fn=add_noise,
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
        elif autoencoder_type == AutoencoderType.DEEPECT_STACKED_AE:
            autoencoder = stacked_ae(
                data.shape[1],
                [500, 500, 2000, embedding_dim],
                torch.nn.init.xavier_normal_,
                loss_fn=None,
                optimizer_fn=None,
                activation_fn=lambda x: torch.nn.functional.relu(x),
            )
            autoencoder.load_parameters(autoencoder_params_path)
        print("Autoencoder parameters loaded from file.")

    return autoencoder


class Original_Dataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        original_image = self.original_dataset[idx]
        return idx, original_image


def flat(
    autoencoder: _AbstractAutoencoder,
    autoencoder_params_path: str,
    dataset_type: DatasetType,
    dataset: Bunch,
    seed: int,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load and preprocess data
    data = dataset["data"]
    labels = dataset["target"]
    results = []
    max_iterations = 40000
    batch_size = 256
    max_leaf_nodes = 12 if dataset_type == DatasetType.REUTERS else 20
    n_clusters = 4 if dataset_type == DatasetType.REUTERS else 10

    max_clustering_epochs = get_max_epoch_size(data, max_iterations, batch_size)

    dataloaders = (
        DataLoader(
            Original_Dataset(data),
            batch_size=batch_size,
            shuffle=True,
            num_workers=5,
            prefetch_factor=200,
        ),
        DataLoader(Original_Dataset(data), batch_size=batch_size, shuffle=False),
    )

    for method in FlatClusteringMethod:
        # Reproducibility
        set_torch_seed(seed)
        # Load the autoencoder parameters
        autoencoder = autoencoder.load_parameters(autoencoder_params_path)

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
                n_init=20,
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
                max_iterations=max_iterations,
                autoencoder=autoencoder,
                clustering_optimizer_params={"lr": 1e-4, "betas": (0.9, 0.999)},
                max_leaf_nodes=max_leaf_nodes,
                random_state=np.random.RandomState(seed),
                custom_dataloaders=dataloaders,
            )
            print("fitting DeepECT...")
            deepect.fit(data)
            print("finished DeepECT...")
            try:
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
            except:
                results.append(
                    {
                        "dataset": dataset_type.value,
                        "method": method.value,
                        "nmi": np.nan,
                        "acc": np.nan,
                        "ari": np.nan,
                        "dp": np.nan,
                        "lp": np.nan,
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
                max_iterations=max_iterations,
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
            try:
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
            except:
                results.append(
                    {
                        "dataset": dataset_type.value,
                        "method": method.value,
                        "nmi": np.nan,
                        "acc": np.nan,
                        "ari": np.nan,
                        "dp": np.nan,
                        "lp": np.nan,
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
                cluster_loss_weight=10.0,  # needs to be 10 to weight cluster loss 10x higher than autoencoder loss like in the paper
                clustering_epochs=max_clustering_epochs,
                random_state=seed,
                initial_clustering_class=KMeans,
                initial_clustering_params={"n_init": 20},
                custom_dataloaders=dataloaders,
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


def shuffle_dataset(data, labels):
    shuffled_indices = np.random.permutation(len(data))
    shuffled_x = data[shuffled_indices, :]
    if labels is not None:
        shuffled_y = labels[shuffled_indices]
        return shuffled_x, shuffled_y
    else:
        return shuffled_x


def hierarchical(
    autoencoder: _AbstractAutoencoder,
    autoencoder_params_path: str,
    dataset_type: DatasetType,
    dataset: Bunch,
    seed: int,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load and preprocess data
    data = dataset["data"]
    labels = dataset["target"]

    results = []
    max_leaf_nodes = 12 if dataset_type == DatasetType.REUTERS else 20
    n_clusters = 4 if dataset_type == DatasetType.REUTERS else 10
    max_iterations = 40000
    batch_size = 256

    max_clustering_epochs = get_max_epoch_size(data, max_iterations, batch_size)

    for method in HierarchicalClusteringMethod:
        # Reproducibility
        set_torch_seed(seed)
        # Load the autoencoder parameters
        autoencoder.load_parameters(autoencoder_params_path)
        autoencoder.to(device)
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
            dataloaders = (
                DataLoader(
                    Original_Dataset(data),
                    batch_size,
                    True,
                    num_workers=5,
                    prefetch_factor=200,
                ),
                DataLoader(Original_Dataset(data), batch_size, False),
            )
            print("fitting idec hierarchical...")
            (
                dp_value_single,
                dp_value_complete,
                leaf_purity_value_single,
                leaf_purity_value_complete,
            ) = idec_hierarchical_clustpy.run_idec_hierarchical(
                data,
                labels,
                seed,
                n_clusters,
                autoencoder,
                epochs=max_clustering_epochs,
                batch_size=batch_size,
                custom_dataloaders=dataloaders,
            )
            print("finished idec hierarchical...")
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": HierarchicalClusteringMethod.IDEC_COMPLETE.value,
                    "dp": dp_value_complete,
                    "lp": leaf_purity_value_complete,
                    "seed": seed,
                }
            )
            results.append(
                {
                    "dataset": dataset_type.value,
                    "method": HierarchicalClusteringMethod.IDEC_SINGLE.value,
                    "dp": dp_value_single,
                    "lp": leaf_purity_value_single,
                    "seed": seed,
                }
            )

        elif method == HierarchicalClusteringMethod.AE_SINGLE:
            if data.shape[0] >= 80000:
                data_shuffled, labels_shuffled = shuffle_dataset(data, labels)
                data_shuffled = data_shuffled[:80000, :]
                labels_shuffled = labels_shuffled[:80000]
            else:
                data_shuffled, labels_shuffled = data, labels
            # Perform hierarchical clustering with Autoencoder and single
            print("fitting ae_single...")
            dendrogram, leaf = ae_single(
                data=data_shuffled,
                labels=labels_shuffled,
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
            if data.shape[0] >= 80000:
                data_shuffled, labels_shuffled = shuffle_dataset(data, labels)
                data_shuffled = data_shuffled[:80000, :]
                labels_shuffled = labels_shuffled[:80000]
            else:
                data_shuffled, labels_shuffled = data, labels
            # Perform hierarchical clustering with Autoencoder and complete
            print("fitting ae_complete...")
            dendrogram, leaf = ae_complete(
                data=data_shuffled,
                labels=labels_shuffled,
                ae_module=autoencoder,
                max_leaf_nodes=max_leaf_nodes,
                n_clusters=n_clusters,
                seed=seed,
                device=device,
            )
            print("finished ae_complete...")
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

    degrees = (-15, 15)
    translation = (
        0.14 if dataset_type == DatasetType.USPS else 0.08,
        0.14 if dataset_type == DatasetType.USPS else 0.08,
    )
    image_min_value = -0.999999 if dataset_type == DatasetType.USPS else 0.0
    image_size = 16 if dataset_type == DatasetType.USPS else 28
    augmentation_transform = transforms.Compose(
        [
            transforms.Lambda(lambda x: x - image_min_value),
            transforms.Lambda(lambda x: x.reshape(image_size, image_size)),
            transforms.ToPILImage(),
            transforms.RandomAffine(
                degrees=degrees,
                shear=degrees,
                translate=translation,
                interpolation=PIL.Image.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.reshape(image_size**2)),
            transforms.Lambda(lambda x: x + image_min_value),
        ]
    )

    class Augmented_Dataset(Dataset):

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
    augmented_dataset = Augmented_Dataset(data, augmentation_transform)
    original_dataset = Original_Dataset(data)

    # Create the dataloaders
    trainloader = DataLoader(
        augmented_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=5,
        prefetch_factor=200,
    )
    testloader = DataLoader(original_dataset, batch_size=256, shuffle=False)

    return trainloader, testloader


def get_dataset(dataset_type: DatasetType):
    if dataset_type == DatasetType.MNIST:

        # data from paper, normalized to [0, 5.1]
        with open("practical/DeepClustering/DeepECT/data/mnist-original.mat", 'rb') as matlab_file:
                matlab_dict = io.loadmat(matlab_file, struct_as_record=True)
        data = matlab_dict['data'].T.astype(np.float32)
        label = np.squeeze(matlab_dict['label'], 0).astype(np.int32)
        data = data * 0.02  
        dataset = Bunch(data=data, target=label, dataset_name='MNIST')
        # dataset = load_mnist()
        # dataset["data"] = np.asarray(dataset["data"] / 255.0, dtype=np.float32)

    elif dataset_type == DatasetType.FASHION_MNIST:
        # paper also used fashionmnist from pytorch and scales to [0,1]
        dataset = load_fmnist()
        dataset["data"] = np.asarray(dataset["data"] / 255.0, dtype=np.float32)
    elif dataset_type == DatasetType.USPS:
        # usps data from paper normalized to [~-1,~1]
        data_mat = io.loadmat("practical/DeepClustering/DeepECT/data/usps_resampled.mat")
        data = np.concatenate([data_mat['train_patterns'].T, data_mat['test_patterns'].T], 0)
        label = np.argmax(np.concatenate([data_mat['train_labels'].T, data_mat['test_labels'].T], 0), 1)
        dataset = Bunch(data=data, target=label, dataset_name='USPS')
        # dataset = load_usps()
        # dataset["data"] = np.asarray(dataset["data"] / 255.0, dtype=np.float32)
    else:
        if not os.path.exists('practical/DeepClustering/DeepECT/data/Reuters/reuters_preprocessed_data.npy'):
            load_reuters_paper()
        # in paper they did preprocessing on reuters (e.g. feature reduction)
        data = np.load('practical/DeepClustering/DeepECT/data/Reuters/reuters_preprocessed_data.npy')
        data = data * 100.0  # Similar to preprocessing done in DEC implementation
        label = np.load('practical/DeepClustering/DeepECT/data/Reuters/reuters_preprocessed_target.npy')
        dataset = Bunch(data=data, target=label, dataset_name='REUTERS')
        # dataset = load_reuters()
        # dataset["data"] = np.asarray(dataset["data"] * 100.0, dtype=np.float32)
        # dataset["target"] = dataset["target"]
    return dataset

def load_reuters_paper():
    data_home = Path("practical/DeepClustering/DeepECT/data/Reuters")
    topics_file = Path(data_home, 'rcv1.topics.hier.orig')

    if not topics_file.exists():
        import requests

        data_home.mkdir(exist_ok=True, parents=True)
        response = requests.get(
            'http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig')
        txt = str(response.text)
        print(topics_file)
        f = open(topics_file, 'w+')
        f.write(txt)
        f.close()

    cwd = os.getcwd()
    data = fetch_rcv1(data_home=data_home, download_if_missing=True)
    names = data.target_names

    ind = np.full(len(names), False, dtype=bool)
    f = open(topics_file, 'r')
    count = 0
    cat_names = []
    for i in range(len(names) + 1):
        s = f.readline()
        if s[8:12] == 'Root':
            ind[i - 1] = True
            cat_names.append(s[50:-1])
            count = count + 1
    f.close()

    labels = data.target[:, ind].copy()
    labels = labels.toarray()
    t = labels.sum(axis=1, keepdims=False)
    single_docs = np.where(t == 1)[0]

    # keep only the documents with single label
    labels = labels[single_docs]
    docs = data.data[single_docs]
    #
    frequency = np.squeeze(np.asarray(docs.sum(axis=0)))

    fre_ind = np.argsort(frequency)
    fre_ind = fre_ind[::-1]
    selected_features = fre_ind[0:2000]
    #
    train_x = docs[:, selected_features].todense().astype(np.float32)
    np.save(f'{data_home}/reuters_preprocessed_data.npy', train_x)

    target_names = cat_names
    target = labels.argmax(axis=1).astype(np.int32)
    np.save(f'{data_home}/reuters_preprocessed_target.npy', target)

    f = open(topics_file, 'r')
    ind2 = np.full(len(names), False, dtype=bool)
    for i in range(len(names) + 1):
        s = f.readline()
        if s[9:12] == 'CAT':
            ind2[i - 1] = True
            cat_names.append(s[50:-1])
    f.close()

    lbls_lvl2 = data.target[:, ind2].copy()
    lbls_lvl2 = lbls_lvl2.toarray()
    lbls_lvl2 = lbls_lvl2[single_docs]
    lbls_lvl2 = np.concatenate([labels, lbls_lvl2], axis=1).astype(np.int32)

    np.save(f'{data_home}/reuters_preprocessed_cat_names_2lvls.npy', np.asarray(cat_names))
    np.save(f'{data_home}/reuters_preprocessed_labels_2lvls.npy', lbls_lvl2)

def get_autoencoder_path(
    autoencoder_type: AutoencoderType,
    autoencoder_params_path: Union[str | None],
    dataset: Bunch,
    embedding_dim: int,
    seed: int,
):
    if (
        autoencoder_params_path is None
        and autoencoder_type == AutoencoderType.CLUSTPY_STANDARD
    ):
        autoencoder_params_path = f"practical/DeepClustering/DeepECT/pretrained_autoencoders/{dataset['dataset_name']}_autoencoder_{embedding_dim}_pretrained_{seed}.pth"
    elif (
        autoencoder_params_path is None
        and autoencoder_type == AutoencoderType.DEEPECT_STACKED_AE
    ):
        autoencoder_params_path = f"practical/DeepClustering/DeepECT/pretrained_autoencoders/{dataset['dataset_name']}_stacked_ae_{embedding_dim}_pretrained_{seed}.pth"
    return autoencoder_params_path


# Example usage
def evaluate(
    autoencoder_type: AutoencoderType,
    dataset_type: DatasetType,
    seed: int,
    autoencoder_params_path: str = None,
    embedding_dim: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start = datetime.datetime.now()
    hierarchical_results_path = f"practical/DeepClustering/DeepECT/results/{dataset_type.name}_{autoencoder_type.name}_{embedding_dim}_hierarchical_{seed}.pq"
    flat_results_path = f"practical/DeepClustering/DeepECT/results/{dataset_type.name}_{autoencoder_type.name}_{embedding_dim}_flat_{seed}.pq"
    print(
        f"-------------------------------------------Run: {dataset_type.name}_{autoencoder_type.name}_{embedding_dim}_{seed}"
    )
    if os.path.exists(hierarchical_results_path) and os.path.exists(flat_results_path):
        hierarchical_results = pd.read_parquet(hierarchical_results_path)
        flat_results = pd.read_parquet(flat_results_path)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
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
        # hierarchical
        if not os.path.exists(hierarchical_results_path):
            hierarchical_results = hierarchical(
                autoencoder=autoencoder,
                autoencoder_params_path=autoencoder_params_path,
                dataset_type=dataset_type,
                dataset=dataset,
                seed=seed,
            )
            hierarchical_results["autoencoder"] = autoencoder_type.value
            hierarchical_results["embedding_dim"] = embedding_dim
            hierarchical_results.to_parquet(hierarchical_results_path)
            print(hierarchical_results)
        else:
            hierarchical_results = pd.read_parquet(hierarchical_results_path)

        # flat
        flat_results = flat(
            autoencoder=autoencoder,
            autoencoder_params_path=autoencoder_params_path,
            dataset_type=dataset_type,
            dataset=dataset,
            seed=seed,
        )
        flat_results["autoencoder"] = autoencoder_type.value
        flat_results["embedding_dim"] = embedding_dim
        flat_results.to_parquet(flat_results_path)
        print(flat_results)

    print(
        f"-------------------------------------------Time needed: {(datetime.datetime.now()-start).total_seconds()/60}min"
    )
    return flat_results, hierarchical_results


def evaluate_multiple_seeds(
    autoencoder_type: AutoencoderType,
    dataset_type: DatasetType,
    seeds: List[int],
    embedding_dims: List[int] = [10],
    autoencoder_params_path: str = None,
):
    all_flat_results = []
    all_hierarchical_results = []

    for seed, embedding_dim in product(seeds, embedding_dims):
        flat_results, hierarchical_results = evaluate(
            autoencoder_type=autoencoder_type,
            dataset_type=dataset_type,
            seed=seed,
            embedding_dim=embedding_dim,
            autoencoder_params_path=autoencoder_params_path,
        )
        all_flat_results.append(flat_results)
        all_hierarchical_results.append(hierarchical_results)

    combined_flat_results = pd.concat(all_flat_results, ignore_index=True)
    combined_hierarchical_results = pd.concat(
        all_hierarchical_results, ignore_index=True
    )

    return combined_flat_results, combined_hierarchical_results


def calculate_flat_mean_for_multiple_seeds(results: pd.DataFrame):
    results = (
        results.groupby(["autoencoder", "embedding_dim", "dataset", "method"])
        .agg({"nmi": "mean", "acc": "mean", "ari": "mean"})
        .reset_index()
    )
    return results


def calculate_hierarchical_mean_for_multiple_seeds(results: pd.DataFrame):
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
    start = datetime.datetime.now()
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
    del autoencoder
    torch.cuda.memory.empty_cache()
    print(
        f"-------------------------------------------Time needed: {(datetime.datetime.now()-start).total_seconds()/60}min"
    )


def pretrain_for_multiple_seeds(seeds: List[int], embedding_dims=[10], worker_num=1):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(mode=True)

    all_autoencoders = list(
        product(AutoencoderType, [None], DatasetType, seeds, embedding_dims)
    )
    with mp.Pool(processes=worker_num) as pool:
        result = pool.starmap(pretraining_with_data_load, all_autoencoders)
    return result


if __name__ == "__main__":
    seeds = [21, 42]
    embedding_dims = [10]
    worker_num = 2
    pretrain_for_multiple_seeds(
        seeds, embedding_dims=embedding_dims, worker_num=worker_num
    )
    all_autoencoders = list(
        product(AutoencoderType, DatasetType, seeds, [None], embedding_dims)
    )
    for ae_type, dataset_type, seed, ae_path, embedding_dim in all_autoencoders:
        evaluate(ae_type, dataset_type, seed, ae_path, embedding_dim)
        # Load the dataset and evaluate flat and hierarchical clustering (stacked autoencoder)

    # flat_results, hierarchical_results = evaluate(
    #    autoencoder_type=AutoencoderType.CLUSTPY_STANDARD,
    #    dataset_type=DatasetType.REUTERS,
    #    seed=42,
    # )
    # print(flat_results_stack, hierarchical_results_stack)
    # print(flat_results, hierarchical_results)
    # evaluation(init_autoencoder=FeedforwardAutoencoder, dataset_type=DatasetType.REUTERS, seed=42)
    # evaluation(init_autoencoder=FeedforwardAutoencoder, dataset_type=DatasetType.FASHION_MNIST, seed=42)

    # combine all results and per experiment, do pivot to aggregate the metrics over the seeds
