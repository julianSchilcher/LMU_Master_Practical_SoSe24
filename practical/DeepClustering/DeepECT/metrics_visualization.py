import random
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import Bunch
import torch.utils.data
import umap
from typing import Tuple, List
import pathlib

from matplotlib import pyplot as plt

from practical.DeepClustering.DeepECT.evaluation_pipeline import (
    AutoencoderType,
    DatasetType,
    calculate_flat_mean_for_multiple_seeds,
    calculate_hierarchical_mean_for_multiple_seeds,
    load_precomputed_results,
    pretraining,
    get_dataset
)


def parse_log(file_path):
    """
    Parse the log file to extract training metrics.

    Parameters
    ----------
    file_path : str
        The path to the log file.

    Returns
    -------
    tuple
        A tuple containing lists of iterations, dc_losses, nc_losses, rec_losses,
        rec_losses_aug, total_losses, and accuracies.
    """
    iterations = []
    dc_losses = []
    nc_losses = []
    rec_losses = []
    rec_losses_aug = []
    total_losses = []
    accuracies = []

    with open(file_path, "r") as file:
        for line in file:
            if "moving averages" in line:
                iteration = int(re.search(r"(\d+) - moving averages", line).group(1))
                dc_loss = float(re.search(r"dc_loss: ([\d.]+)", line).group(1))
                nc_loss = float(re.search(r"nc_loss: ([\d.]+)", line).group(1))
                rec_loss = float(re.search(r"rec_loss: ([\d.]+)", line).group(1))
                rec_loss_aug_match = re.search(r"rec_loss_aug: ([\d.]+)", line)
                if rec_loss_aug_match:
                    rec_loss_aug = float(rec_loss_aug_match.group(1))
                else:
                    rec_loss_aug = None
                total_loss = float(re.search(r"total_loss: ([\d.]+)", line).group(1))
                accuracy_match = re.search(r"accuracy: ([\d.]+)", line)
                if accuracy_match:
                    accuracy = float(accuracy_match.group(1))
                else:
                    accuracy = None

                iterations.append(iteration)
                dc_losses.append(dc_loss)
                nc_losses.append(nc_loss)
                rec_losses.append(rec_loss)
                rec_losses_aug.append(rec_loss_aug)
                total_losses.append(total_loss)
                accuracies.append(accuracy)

    return (
        iterations,
        dc_losses,
        nc_losses,
        rec_losses,
        rec_losses_aug,
        total_losses,
        accuracies,
    )


def plot_metrics(log_file_path):
    """
    Plot training metrics from a log file.

    Parameters
    ----------
    log_file_path : str
        The path to the log file.
    """
    (
        iterations,
        dc_losses,
        nc_losses,
        rec_losses,
        rec_losses_aug,
        total_losses,
        accuracies,
    ) = parse_log(log_file_path)

    plt.figure(figsize=(12, 6))
    title = log_file_path.split("/")[-1].replace(".txt", "")
    plt.suptitle(title)

    plt.subplot(2, 1, 1)
    plt.plot(iterations, dc_losses, label="DC Loss")
    plt.plot(iterations, nc_losses, label="NC Loss")
    plt.plot(iterations, rec_losses, label="Reconstruction Loss")
    plt.plot(iterations, rec_losses_aug, label="Reconstruction Loss Augmented")
    plt.plot(iterations, total_losses, label="Total Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(iterations, accuracies, label="Accuracy", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_comparison(log_file_path1, log_file_path2):
    """
    Plot comparison of training metrics from two log files.

    Parameters
    ----------
    log_file_path1 : str
        The path to the first log file.
    log_file_path2 : str
        The path to the second log file.
    """
    (
        iterations1,
        dc_losses1,
        nc_losses1,
        rec_losses1,
        rec_losses_aug1,
        total_losses1,
        accuracies1,
    ) = parse_log(log_file_path1)
    (
        iterations2,
        dc_losses2,
        nc_losses2,
        rec_losses2,
        rec_losses_aug2,
        total_losses2,
        accuracies2,
    ) = parse_log(log_file_path2)

    plt.figure(figsize=(12, 15))

    title1 = log_file_path1.split("/")[-1].replace(".txt", "")
    title2 = log_file_path2.split("/")[-1].replace(".txt", "")

    plt.subplot(5, 2, 1)
    plt.plot(iterations1, dc_losses1, label=f"{title1} DC Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title1} DC Loss")
    plt.legend()

    plt.subplot(5, 2, 2)
    plt.plot(iterations2, dc_losses2, label=f"{title2} DC Loss", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title2} DC Loss")
    plt.legend()

    plt.subplot(5, 2, 3)
    plt.plot(iterations1, nc_losses1, label=f"{title1} NC Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title1} NC Loss")
    plt.legend()

    plt.subplot(5, 2, 4)
    plt.plot(iterations2, nc_losses2, label=f"{title2} NC Loss", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title2} NC Loss")
    plt.legend()

    plt.subplot(5, 2, 5)
    plt.plot(iterations1, rec_losses1, label=f"{title1} Reconstruction Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title1} Reconstruction Loss")
    plt.legend()

    plt.subplot(5, 2, 6)
    plt.plot(
        iterations2, rec_losses2, label=f"{title2} Reconstruction Loss", color="orange"
    )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title2} Reconstruction Loss")
    plt.legend()

    plt.subplot(5, 2, 7)
    plt.plot(
        iterations1, rec_losses_aug1, label=f"{title1} Reconstruction Loss Augmented"
    )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title1} Reconstruction Loss Augmented")
    plt.legend()

    plt.subplot(5, 2, 8)
    plt.plot(
        iterations2,
        rec_losses_aug2,
        label=f"{title2} Reconstruction Loss Augmented",
        color="orange",
    )
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title2} Reconstruction Loss Augmented")
    plt.legend()

    plt.subplot(5, 2, 9)
    plt.plot(iterations1, total_losses1, label=f"{title1} Total Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title1} Total Loss")
    plt.legend()

    plt.subplot(5, 2, 10)
    plt.plot(iterations2, total_losses2, label=f"{title2} Total Loss", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(f"{title2} Total Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_accuracy_comparison(log_file_path1, log_file_path2):
    """
    Plot comparison of training accuracy from two log files.

    Parameters
    ----------
    log_file_path1 : str
        The path to the first log file.
    log_file_path2 : str
        The path to the second log file.
    """
    iterations1, _, _, _, _, _, accuracies1 = parse_log(log_file_path1)
    iterations2, _, _, _, _, _, accuracies2 = parse_log(log_file_path2)

    plt.figure(figsize=(12, 6))

    title1 = log_file_path1.split("/")[-1].replace(".txt", "")
    title2 = log_file_path2.split("/")[-1].replace(".txt", "")

    plt.plot(iterations1, accuracies1, label=f"{title1} Accuracy")
    plt.plot(iterations2, accuracies2, label=f"{title2} Accuracy", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy Comparison")
    plt.legend()

    plt.show()


def visualize_peformance_AE(
    param_path_autoencoder: str,
    autoencoder_class: torch.nn.Module,
    dataset: Bunch,
    image_size: tuple,
    number_samples: int,
    seed: int = None,
    title: str = None
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if seed is not None and type(seed) == int:
        random.seed(seed)

    autoencoder = pretraining(
        autoencoder_class, param_path_autoencoder, dataset, seed, 10
    ).to(device)
    samples = dataset["data"]
    labels = dataset["target"]
    fig, ax = plt.subplots(2, number_samples)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    ax = ax.flatten()
    with torch.no_grad():
        for i, index in enumerate(
            sorted(random.sample(range(samples.shape[0]), number_samples))
        ):
            img = samples[index]
            if img.ndim == 1:
                img = np.expand_dims(img, 0)
            img_rec = (
                autoencoder.decode(autoencoder.encode(torch.from_numpy(img).to(device)))
                .cpu()
                .numpy()
            )
            ax[i].imshow(img.reshape(image_size[0], image_size[1]), cmap="gray")
            ax[i + number_samples].imshow(
                img_rec.reshape(image_size[0], image_size[1]), cmap="gray"
            )
            ax[i].set_title(f"original")
            ax[i + number_samples].set_title(f"reconstructed")
            ax[i].set_axis_off()
            ax[i + number_samples].set_axis_off()

        embeddings = []

        for batch in torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.tensor(samples, dtype=torch.float32)),
            batch_size=256,
        ):
            embeddings.append(autoencoder.encode(batch[0].to(device)).cpu().numpy())

    embeddings = np.concatenate(embeddings)
    # PCA of embedded space
    print("fitting pca")
    plt.figure()
    pca = PCA(n_components=2)
    projected_data = pca.fit_transform(embeddings)
    plt.scatter(projected_data[:, 0], projected_data[:, 1], c=labels, cmap="viridis")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"PCA of embedded space  {title if title is not None else ''}")
    if dataset["dataset_name"] == "FashionMNIST":
        cbar = plt.colorbar()
        cbar.set_ticks(np.arange(10))
        cbar.set_ticklabels(
            [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ]
        )

    else:
        plt.colorbar(label="Digit")
    plt.show()
    print("fitted pca")

    print("fitting umap")
    plt.figure()
    projected_data = umap.UMAP().fit_transform(embeddings)
    plt.scatter(projected_data[:, 0], projected_data[:, 1], c=labels, cmap="viridis")
    plt.xlabel("umap feature 1")
    plt.ylabel("umap feature 2")
    plt.title(f"umap of embedded space  {title if title is not None else ''}")
    if dataset["dataset_name"] == "FashionMNIST":
        cbar = plt.colorbar()
        cbar.set_ticks(np.arange(10))
        cbar.set_ticklabels(
            [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ]
        )

    else:
        plt.colorbar(label="Digit")
    plt.show()
    print("fitted umap")


def visualize_perfomance_multiple_AE(dataset_types: List[DatasetType], autoencoders: List[str], autoencoder_type: AutoencoderType = AutoencoderType.DEEPECT_STACKED_AE):
    
    def get_title(filepath):
        after_dipect = filepath.split('DeepECT/')[1]
        
        title = after_dipect.split('.pth')[0]
    
        return title


    for i, autoencoder in enumerate(autoencoders):
        dataset = get_dataset(dataset_types[i])
        visualize_peformance_AE(pathlib.Path(autoencoders[i]), autoencoder_type[i], dataset, (16, 16) if dataset_types[i] == DatasetType.USPS else (28,28), 5, 0, get_title(autoencoders[i]))
 



def load_results():
    seeds = [21, 42, 63]
    # MNIST
    mnist_multiple_seeds_stacked_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.DEEPECT_STACKED_AE,
        dataset_type=DatasetType.MNIST,
        seeds=seeds,
    )

    mean_flat_mnist = calculate_flat_mean_for_multiple_seeds(
        mnist_multiple_seeds_stacked_ae
    )
    mean_hierarchical_mnist = calculate_hierarchical_mean_for_multiple_seeds(
        mnist_multiple_seeds_stacked_ae
    )

    # USPS

    usps_multiple_seeds_stacked_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.DEEPECT_STACKED_AE,
        dataset_type=DatasetType.USPS,
        seeds=seeds,
    )

    mean_flat_usps = calculate_flat_mean_for_multiple_seeds(
        usps_multiple_seeds_stacked_ae
    )
    mean_hierarchical_usps = calculate_hierarchical_mean_for_multiple_seeds(
        usps_multiple_seeds_stacked_ae
    )

    # FashionMNIST

    fashion_multiple_seeds_stacked_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.DEEPECT_STACKED_AE,
        dataset_type=DatasetType.FASHION_MNIST,
        seeds=seeds,
    )

    mean_flat_fashion = calculate_flat_mean_for_multiple_seeds(
        fashion_multiple_seeds_stacked_ae
    )
    mean_hierarchical_fashion = calculate_hierarchical_mean_for_multiple_seeds(
        fashion_multiple_seeds_stacked_ae
    )

    # Reuters
    reuters_multiple_seeds_stacked_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.DEEPECT_STACKED_AE,
        dataset_type=DatasetType.REUTERS,
        seeds=seeds,
    )

    mean_flat_reuters = calculate_flat_mean_for_multiple_seeds(
        reuters_multiple_seeds_stacked_ae
    )
    mean_hierarchical_reuters = calculate_hierarchical_mean_for_multiple_seeds(
        reuters_multiple_seeds_stacked_ae
    )

    ## Flat metrics
    flat_combined_df = pd.concat(
        [
            mnist_multiple_seeds_stacked_ae,
            usps_multiple_seeds_stacked_ae,
            fashion_multiple_seeds_stacked_ae,
            reuters_multiple_seeds_stacked_ae,
        ],
        ignore_index=True,
    )
    flat_combined_df.fillna(0)
    # Pivot the DataFrame to match the desired format
    flat_pivot_df = (
        flat_combined_df.pivot_table(
            index=[
                "autoencoder",
                "method",
            ],
            columns="dataset",
            values=[
                "nmi",
                "acc",
                "ari",
            ],
            aggfunc=[np.mean, lambda x: np.std(x, ddof=0)],
            fill_value=0,
        )
        .dropna(how="all")
        .round(2)
    )
    flat_pivot_df.rename(columns={"<lambda>": "std"}, level=0, inplace=True)
    # Reorder the columns to match the order in the image
    flat_pivot_df.columns = flat_pivot_df.columns.swaplevel(0, 2)
    flat_pivot_df = flat_pivot_df.reindex(
        columns=[
            (DatasetType.MNIST.value, "nmi", "mean"),
            (DatasetType.MNIST.value, "nmi", "std"),
            (DatasetType.MNIST.value, "acc", "mean"),
            (DatasetType.MNIST.value, "acc", "std"),
            (DatasetType.MNIST.value, "ari", "mean"),
            (DatasetType.MNIST.value, "ari", "std"),
            (DatasetType.USPS.value, "nmi", "mean"),
            (DatasetType.USPS.value, "nmi", "std"),
            (DatasetType.USPS.value, "acc", "mean"),
            (DatasetType.USPS.value, "acc", "std"),
            (DatasetType.USPS.value, "ari", "mean"),
            (DatasetType.USPS.value, "ari", "std"),
            (DatasetType.FASHION_MNIST.value, "nmi", "mean"),
            (DatasetType.FASHION_MNIST.value, "nmi", "std"),
            (DatasetType.FASHION_MNIST.value, "acc", "mean"),
            (DatasetType.FASHION_MNIST.value, "acc", "std"),
            (DatasetType.FASHION_MNIST.value, "ari", "mean"),
            (DatasetType.FASHION_MNIST.value, "ari", "std"),
            (DatasetType.REUTERS.value, "nmi", "mean"),
            (DatasetType.REUTERS.value, "nmi", "std"),
            (DatasetType.REUTERS.value, "acc", "mean"),
            (DatasetType.REUTERS.value, "acc", "std"),
            (DatasetType.REUTERS.value, "ari", "mean"),
            (DatasetType.REUTERS.value, "ari", "std"),
        ]
    )

    # For Jupyter Notebook display with better formatting
    flat_results_html = flat_pivot_df.style.set_table_styles(
        [
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#f7f7f9"),
                    ("color", "#333"),
                    ("border", "1px solid #ddd"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#f9f9f9"), ("color", "#333")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#fff"), ("color", "#333")],
            },
        ]
    ).set_caption("Flat Clustering results")

    ## Hierarchical clustering
    hierarchical_combined = pd.concat(
        [
            mnist_multiple_seeds_stacked_ae,
            usps_multiple_seeds_stacked_ae,
            fashion_multiple_seeds_stacked_ae,
            reuters_multiple_seeds_stacked_ae,
        ],
        ignore_index=True,
    )
    hierarchical_combined.fillna(0)
    # Pivot the DataFrame to match the desired format
    hierarchical_pivot_df = (
        hierarchical_combined.pivot_table(
            index=["autoencoder", "method"],
            columns="dataset",
            values=["dp", "lp"],
            aggfunc=[np.mean, lambda x: np.std(x, ddof=0)],
            fill_value=0,
        )
        .dropna(how="all")
        .round(2)
    )
    hierarchical_pivot_df.rename(columns={"<lambda>": "std"}, level=0, inplace=True)

    # Reorder the columns to match the order in the image
    hierarchical_pivot_df.columns = hierarchical_pivot_df.columns.swaplevel(0, 2)
    hierarchical_pivot_df = hierarchical_pivot_df.reindex(
        columns=[
            (DatasetType.MNIST.value, "dp", "mean"),
            (DatasetType.MNIST.value, "dp", "std"),
            (DatasetType.MNIST.value, "lp", "mean"),
            (DatasetType.MNIST.value, "lp", "std"),
            (DatasetType.USPS.value, "dp", "mean"),
            (DatasetType.USPS.value, "dp", "std"),
            (DatasetType.USPS.value, "lp", "mean"),
            (DatasetType.USPS.value, "lp", "std"),
            (DatasetType.FASHION_MNIST.value, "dp", "mean"),
            (DatasetType.FASHION_MNIST.value, "dp", "std"),
            (DatasetType.FASHION_MNIST.value, "lp", "mean"),
            (DatasetType.FASHION_MNIST.value, "lp", "std"),
            (DatasetType.REUTERS.value, "dp", "mean"),
            (DatasetType.REUTERS.value, "dp", "std"),
            (DatasetType.REUTERS.value, "lp", "mean"),
            (DatasetType.REUTERS.value, "lp", "std"),
        ]
    )

    # For Jupyter Notebook display with better formatting
    hierarchical_results_html = hierarchical_pivot_df.style.set_table_styles(
        [
            {
                "selector": "thead th",
                "props": [
                    ("background-color", "#f7f7f9"),
                    ("color", "#333"),
                    ("border", "1px solid #ddd"),
                ],
            },
            {
                "selector": "tbody tr:nth-child(even)",
                "props": [("background-color", "#f9f9f9"), ("color", "#333")],
            },
            {
                "selector": "tbody tr:nth-child(odd)",
                "props": [("background-color", "#fff"), ("color", "#333")],
            },
        ]
    ).set_caption("Hierarchical Clustering results")
    return (
        flat_pivot_df,
        flat_combined_df,
        flat_results_html,
        hierarchical_pivot_df,
        hierarchical_combined,
        hierarchical_results_html,
    )
