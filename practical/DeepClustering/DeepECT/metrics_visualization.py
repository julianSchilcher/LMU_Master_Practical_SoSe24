import random
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.utils import Bunch
import torch.utils.data
import umap

from matplotlib import pyplot as plt

from practical.DeepClustering.DeepECT.evaluation_pipeline import (
    AutoencoderType,
    DatasetType,
    calculate_flat_mean_for_multiple_seeds,
    calculate_hierarchical_mean_for_multiple_seeds,
    load_precomputed_results,
    pretraining,
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
    plt.title("PCA of embedded space")
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
    plt.title("umap of embedded space")
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


def load_results():
    seeds = [21, 42]
    # MNIST
    mnist_multiple_seeds_clustpy_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.CLUSTPY_STANDARD,
        dataset_type=DatasetType.MNIST,
        seeds=seeds,
    )
    mnist_multiple_seeds_stacked_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.DEEPECT_STACKED_AE,
        dataset_type=DatasetType.MNIST,
        seeds=seeds,
    )

    mean_flat_mnist = calculate_flat_mean_for_multiple_seeds(
        pd.concat(
            [mnist_multiple_seeds_clustpy_ae, mnist_multiple_seeds_stacked_ae],
            ignore_index=True,
        )
    )
    mean_hierarchical_mnist = calculate_hierarchical_mean_for_multiple_seeds(
        pd.concat(
            [mnist_multiple_seeds_clustpy_ae, mnist_multiple_seeds_stacked_ae],
            ignore_index=True,
        )
    )

    # USPS
    usps_multiple_seeds_clustpy_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.CLUSTPY_STANDARD,
        dataset_type=DatasetType.USPS,
        seeds=seeds,
    )

    usps_multiple_seeds_stacked_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.DEEPECT_STACKED_AE,
        dataset_type=DatasetType.USPS,
        seeds=seeds,
    )

    mean_flat_usps = calculate_flat_mean_for_multiple_seeds(
        pd.concat(
            [usps_multiple_seeds_clustpy_ae, usps_multiple_seeds_stacked_ae],
            ignore_index=True,
        )
    )
    mean_hierarchical_usps = calculate_hierarchical_mean_for_multiple_seeds(
        pd.concat(
            [usps_multiple_seeds_clustpy_ae, usps_multiple_seeds_stacked_ae],
            ignore_index=True,
        )
    )

    # FashionMNIST
    fashion_multiple_seeds_clustpy_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.CLUSTPY_STANDARD,
        dataset_type=DatasetType.FASHION_MNIST,
        seeds=seeds,
    )

    fashion_multiple_seeds_stacked_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.DEEPECT_STACKED_AE,
        dataset_type=DatasetType.FASHION_MNIST,
        seeds=seeds,
    )

    mean_flat_fashion = calculate_flat_mean_for_multiple_seeds(
        pd.concat(
            [fashion_multiple_seeds_clustpy_ae, fashion_multiple_seeds_stacked_ae],
            ignore_index=True,
        )
    )
    mean_hierarchical_fashion = calculate_hierarchical_mean_for_multiple_seeds(
        pd.concat(
            [fashion_multiple_seeds_clustpy_ae, fashion_multiple_seeds_stacked_ae],
            ignore_index=True,
        )
    )

    # Reuters
    reuters_multiple_seeds_clustpy_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.CLUSTPY_STANDARD,
        dataset_type=DatasetType.REUTERS,
        seeds=seeds,
    )

    reuters_multiple_seeds_stacked_ae = load_precomputed_results(
        autoencoder_type=AutoencoderType.DEEPECT_STACKED_AE,
        dataset_type=DatasetType.REUTERS,
        seeds=seeds,
    )

    mean_flat_reuters = calculate_flat_mean_for_multiple_seeds(
        pd.concat(
            [reuters_multiple_seeds_clustpy_ae, reuters_multiple_seeds_stacked_ae],
            ignore_index=True,
        )
    )
    mean_hierarchical_reuters = calculate_hierarchical_mean_for_multiple_seeds(
        pd.concat(
            [reuters_multiple_seeds_clustpy_ae, reuters_multiple_seeds_stacked_ae],
            ignore_index=True,
        )
    )

    ## Flat metrics
    flat_combined_df = pd.concat(
        [mean_flat_mnist, mean_flat_usps, mean_flat_fashion, mean_flat_reuters],
        ignore_index=True,
    )
    # Pivot the DataFrame to match the desired format
    flat_pivot_df = flat_combined_df.pivot(
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
    ).dropna(how="all")

    # Reorder the columns to match the order in the image
    flat_pivot_df.columns = flat_pivot_df.columns.swaplevel(0, 1)
    flat_pivot_df = flat_pivot_df.reindex(
        columns=[
            (DatasetType.MNIST.value, "nmi"),
            (DatasetType.MNIST.value, "acc"),
            (DatasetType.MNIST.value, "ari"),
            (DatasetType.USPS.value, "nmi"),
            (DatasetType.USPS.value, "acc"),
            (DatasetType.USPS.value, "ari"),
            (DatasetType.FASHION_MNIST.value, "nmi"),
            (DatasetType.FASHION_MNIST.value, "acc"),
            (DatasetType.FASHION_MNIST.value, "ari"),
            (DatasetType.REUTERS.value, "nmi"),
            (DatasetType.REUTERS.value, "acc"),
            (DatasetType.REUTERS.value, "ari"),
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
            mean_hierarchical_mnist,
            mean_hierarchical_usps,
            mean_hierarchical_fashion,
            mean_hierarchical_reuters,
        ],
        ignore_index=True,
    )
    # Pivot the DataFrame to match the desired format
    hierarchical_pivot_df = hierarchical_combined.pivot(
        index=["autoencoder", "method"],
        columns="dataset",
        values=["dp", "lp"],
    ).dropna(how="all")

    # Reorder the columns to match the order in the image
    hierarchical_pivot_df.columns = hierarchical_pivot_df.columns.swaplevel(0, 1)
    hierarchical_pivot_df = hierarchical_pivot_df.reindex(
        columns=[
            (DatasetType.MNIST.value, "dp"),
            (DatasetType.MNIST.value, "lp"),
            (DatasetType.USPS.value, "dp"),
            (DatasetType.USPS.value, "lp"),
            (DatasetType.FASHION_MNIST.value, "dp"),
            (DatasetType.FASHION_MNIST.value, "lp"),
            (DatasetType.REUTERS.value, "dp"),
            (DatasetType.REUTERS.value, "lp"),
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
