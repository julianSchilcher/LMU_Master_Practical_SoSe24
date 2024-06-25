import glob
import os
import random
import re

import networkx as nx
import numpy as np
import pandas as pd
import torch.utils.data
import umap
from clustpy.deep._data_utils import get_dataloader
from clustpy.deep._utils import encode_batchwise
from clustpy.deep.dipencoder import dip_test
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils import Bunch

from practical.DeepClustering.DipECT.dipect import Cluster_Tree
from practical.DeepClustering.DipECT.evaluation_pipeline import (
    AutoencoderType, DatasetType, calculate_flat_mean_for_multiple_seeds,
    calculate_hierarchical_mean_for_multiple_seeds,
    get_custom_dataloader_augmentations, load_precomputed_results, pretraining)


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
        A tuple containing lists of epochs, mov_rec_losses, mov_losses,
        rec_losses_aug, and total_losses.
    """
    epochs = []
    mov_rec_losses = []
    mov_losses = []
    rec_losses_aug = []
    total_losses = []

    with open(file_path, "r") as file:
        for line in file:
            if "moving averages" in line:
                epoch = int(re.search(r"epoch: (\d+) - moving averages", line).group(1))
                mov_rec_loss = float(re.search(r"mov_rec_loss: ([\d.-]+)", line).group(1))
                mov_loss = float(re.search(r"mov_loss: ([\d.-]+)", line).group(1))
                rec_loss_aug_match = re.search(r"rec_loss_aug: ([\d.-]+)", line)
                if rec_loss_aug_match:
                    rec_loss_aug = float(rec_loss_aug_match.group(1))
                else:
                    rec_loss_aug = None
                total_loss = float(re.search(r"total_loss: ([\d.-]+)", line).group(1))

                epochs.append(epoch)
                mov_rec_losses.append(mov_rec_loss)
                mov_losses.append(mov_loss)
                rec_losses_aug.append(rec_loss_aug)
                total_losses.append(total_loss)

    return (
        epochs,
        mov_rec_losses,
        mov_losses,
        rec_losses_aug,
        total_losses,
    )


import glob
import os

import numpy as np


def parse_multiple_logs(relative_path, file_pattern):
    """
    Parse multiple log files matching the file pattern within the specified relative path
    and calculate the mean of metrics.

    Parameters
    ----------
    relative_path : str
        The relative path containing the log files.
    file_pattern : str
        The file pattern to match log files.

    Returns
    -------
    tuple
        A tuple containing lists of epochs, mean_mov_rec_losses, mean_mov_losses,
        mean_rec_losses_aug, and mean_total_losses.
    """
    file_paths = glob.glob(os.path.join(relative_path, file_pattern))
    
    all_epochs = []
    all_mov_rec_losses = []
    all_mov_losses = []
    all_rec_losses_aug = []
    all_total_losses = []

    for file_path in file_paths:
        epochs, mov_rec_losses, mov_losses, rec_losses_aug, total_losses = parse_log(file_path)
        
        all_epochs.append(epochs)
        all_mov_rec_losses.append(mov_rec_losses)
        all_mov_losses.append(mov_losses)
        all_rec_losses_aug.append(rec_losses_aug)
        all_total_losses.append(total_losses)

    # Convert lists to numpy arrays for mean calculation, handling None values
    all_epochs = np.array(all_epochs)
    all_mov_rec_losses = np.array([[x if x is not None else np.nan for x in sublist] for sublist in all_mov_rec_losses])
    all_mov_losses = np.array([[x if x is not None else np.nan for x in sublist] for sublist in all_mov_losses])
    all_rec_losses_aug = np.array([[x if x is not None else np.nan for x in sublist] for sublist in all_rec_losses_aug])
    all_total_losses = np.array([[x if x is not None else np.nan for x in sublist] for sublist in all_total_losses])

    # Calculate means, ignoring nan values
    mean_epochs = all_epochs[0]  # Assuming epochs are the same across all logs
    mean_mov_rec_losses = np.nanmean(all_mov_rec_losses, axis=0)
    mean_mov_losses = np.nanmean(all_mov_losses, axis=0)
    mean_rec_losses_aug = np.nanmean(all_rec_losses_aug, axis=0)
    mean_total_losses = np.nanmean(all_total_losses, axis=0)

    return (
        mean_epochs,
        mean_mov_rec_losses,
        mean_mov_losses,
        mean_rec_losses_aug,
        mean_total_losses,
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
        epochs,
        mov_rec_losses,
        mov_losses,
        rec_losses_aug,
        total_losses,
    ) = parse_log(log_file_path)

    plt.figure(figsize=(12, 10))
    title = log_file_path.split("/")[-1].replace(".txt", "")
    plt.suptitle(title)

    plt.subplot(3, 1, 1)
    plt.plot(epochs, mov_rec_losses, label="Moving Average Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Moving Average Reconstruction Loss")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, mov_losses, label="Moving Average Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Moving Average Loss")
    plt.legend()

    if any(rec_losses_aug):
        plt.subplot(3, 1, 3)
        plt.plot(epochs, rec_losses_aug, label="Reconstruction Loss Augmented")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Reconstruction Loss Augmented")
        plt.legend()
    else:
        plt.subplot(3, 1, 3)
        plt.plot(epochs, total_losses, label="Total Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Total Loss")
        plt.legend()

    plt.tight_layout()
    plt.show()

def plot_mean_metrics(relative_path, file_pattern):
    """
    Plot mean training metrics from multiple log files within a specified relative path.

    Parameters
    ----------
    relative_path : str
        The relative path containing the log files.
    file_pattern : str
        The file pattern to match log files.
    """
    (
        epochs,
        mean_mov_rec_losses,
        mean_mov_losses,
        mean_rec_losses_aug,
        mean_total_losses,
    ) = parse_multiple_logs(relative_path, file_pattern)

    plt.figure(figsize=(12, 10))
    title = "Mean Metrics for " + os.path.basename(file_pattern).replace("*", "")
    plt.suptitle(title)

    plt.subplot(3, 1, 1)
    plt.plot(epochs, mean_mov_rec_losses, label="Mean Moving Average Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Mean Moving Average Reconstruction Loss")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(epochs, mean_mov_losses, label="Mean Moving Average Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Mean Moving Average Loss")
    plt.legend()

    if any(mean_rec_losses_aug):
        plt.subplot(3, 1, 3)
        plt.plot(epochs, mean_rec_losses_aug, label="Mean Reconstruction Loss Augmented")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Mean Reconstruction Loss Augmented")
        plt.legend()
    else:
        plt.subplot(3, 1, 3)
        plt.plot(epochs, mean_total_losses, label="Mean Total Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Mean Total Loss")
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
        epochs1,
        mov_rec_losses1,
        mov_losses1,
        rec_losses_aug1,
        total_losses1,
    ) = parse_log(log_file_path1)
    (
        epochs2,
        mov_rec_losses2,
        mov_losses2,
        rec_losses_aug2,
        total_losses2,
    ) = parse_log(log_file_path2)

    plt.figure(figsize=(12, 15))

    title1 = log_file_path1.split("/")[-1].replace(".txt", "")
    title2 = log_file_path2.split("/")[-1].replace(".txt", "")

    plt.subplot(4, 2, 1)
    plt.plot(epochs1, mov_rec_losses1, label=f"{title1} Moving Average Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title1} Moving Average Reconstruction Loss")
    plt.legend()

    plt.subplot(4, 2, 2)
    plt.plot(epochs2, mov_rec_losses2, label=f"{title2} Moving Average Reconstruction Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title2} Moving Average Reconstruction Loss")
    plt.legend()

    plt.subplot(4, 2, 3)
    plt.plot(epochs1, mov_losses1, label=f"{title1} Moving Average Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title1} Moving Average Loss")
    plt.legend()

    plt.subplot(4, 2, 4)
    plt.plot(epochs2, mov_losses2, label=f"{title2} Moving Average Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title2} Moving Average Loss")
    plt.legend()

    if any(rec_losses_aug1) or any(rec_losses_aug2):
        plt.subplot(4, 2, 5)
        plt.plot(epochs1, rec_losses_aug1, label=f"{title1} Reconstruction Loss Augmented")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{title1} Reconstruction Loss Augmented")
        plt.legend()

        plt.subplot(4, 2, 6)
        plt.plot(epochs2, rec_losses_aug2, label=f"{title2} Reconstruction Loss Augmented", color="orange")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"{title2} Reconstruction Loss Augmented")
        plt.legend()

    plt.subplot(4, 2, 7)
    plt.plot(epochs1, total_losses1, label=f"{title1} Total Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{title1} Total Loss")
    plt.legend()

    plt.subplot(4, 2, 8)
    plt.plot(epochs2, total_losses2, label=f"{title2} Total Loss", color="orange")
    plt.xlabel("Epochs")
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


def show_augmented_data(data: np.ndarray, dataset_type: DatasetType, image_size: tuple, number_samples: int):

    (trainloader, _) = get_custom_dataloader_augmentations(data, dataset_type)
    idx, M, M_aug = next(iter(trainloader))

    print(torch.max(M_aug))
    print(torch.min(M_aug))

    fig, ax = plt.subplots(2, number_samples)
    fig.tight_layout()
    ax = ax.flatten()
    for i, index in enumerate(sorted(random.sample(range(M.shape[0]),number_samples))):
        img = M[index]
        img_aug = M_aug[index]
        if img.ndim == 1:
            img = np.expand_dims(img,0)
        ax[i].imshow(img.reshape(image_size[0],image_size[1]), cmap='gray')
        ax[i+number_samples].imshow(img_aug.reshape(image_size[0],image_size[1]), cmap='gray')
        ax[i].set_title(f'original')
        ax[i+number_samples].set_title(f'augmented')
        ax[i].set_axis_off()
        ax[i+number_samples].set_axis_off()
        




def graphviz_layout_binary_tree(G, root):
    A = nx.nx_agraph.to_agraph(G)
    A.layout(prog='dot')  # 'dot' is used for hierarchical layouts
    pos = {}
    for node in G.nodes():
        x, y = A.get_node(node).attr['pos'].split(',')
        pos[node] = (float(x), float(y))
    return pos

def build_and_visualize_tree(root, autoencoder, data):
    if root is None:
        return
    testloader = get_dataloader(data, 256, False, False)
    embedded_data = encode_batchwise(testloader, autoencoder, "cpu")
    embedded_data = torch.from_numpy(embedded_data)

    # Create a directed graph
    G = nx.DiGraph()
    
    # Helper function to add nodes and edges to the graph
    def add_edges(G, node):
        if node.left_child:
            G.add_edge(node, node.left_child)
            add_edges(G, node.left_child)
        if node.right_child:
            G.add_edge(node, node.right_child)
            add_edges(G, node.right_child)
    
    # Add nodes and edges starting from the root
    add_edges(G, root)

    # Create a position dictionary for the nodes
    pos = graphviz_layout_binary_tree(G, root)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))
    nx.draw(G, pos, ax=ax, with_labels=False)
    
    # Draw the images at the nodes
    for node in G.nodes:
        image = np.mean(autoencoder.decode(embedded_data[node.assigned_indices]).detach().numpy(), axis=0)
        # image = autoencoder.decode(center).detach().numpy()
        image = image.reshape(28,28)
        
        # if dataset_type == DatasetType.USPS:
        #     image = image.reshape(16,16)
        # else:
        #     image = image.reshape(28,28)
        imagebox = OffsetImage(image, zoom=1)
        ab = AnnotationBbox(imagebox, pos[node], frameon=False)
        ax.add_artist(ab)
    
    plt.show()

from PIL import Image, ImageDraw, ImageFont


def integer_to_image(number, font_size=10, image_size=(28, 28), bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    image = Image.new('RGB', image_size, bg_color)
    
    
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    text = str(number)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
  
    position = ((image_size[0] - text_width) / 2, (image_size[1] - text_height) / 2)
    

    draw.text(position, text, fill=text_color, font=font)
    
    return np.array(image)


def build_and_visualize_splitindex_tree(root):
    if root is None:
        return

    # Create a directed graph
    G = nx.DiGraph()
    
    # Helper function to add nodes and edges to the graph
    def add_edges(G, node):
        if node.left_child:
            G.add_edge(node, node.left_child)
            add_edges(G, node.left_child)
        if node.right_child:
            G.add_edge(node, node.right_child)
            add_edges(G, node.right_child)
    
    # Add nodes and edges starting from the root
    add_edges(G, root)

    # Create a position dictionary for the nodes
    pos = graphviz_layout_binary_tree(G, root)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))
    nx.draw(G, pos, ax=ax, with_labels=False)
    
    # Draw the images at the nodes
    for node in G.nodes:
        image = integer_to_image(node.split_id)
        # image = autoencoder.decode(center).detach().numpy()
        # image = image.reshape(28,28)
        
        # if dataset_type == DatasetType.USPS:
        #     image = image.reshape(16,16)
        # else:
        #     image = image.reshape(28,28)
        imagebox = OffsetImage(image, zoom=1)
        ab = AnnotationBbox(imagebox, pos[node], frameon=False)
        ax.add_artist(ab)
    
    plt.show()


def visualize_axis_for_2_clusters(X1, X2, y1, y2):
    def get_inital_projection_axis(X_embedd):
        """
        Returns the inital projection axis for the data in the given trainloader. Furthermore, the size of the higher projection cluster and the lower projection cluster will be returned (e.g to initialise pruning indicator).
        """
        # init projection axis on full dataset
        kmeans = KMeans(n_clusters=2, n_init=10).fit(X_embedd)
        kmeans_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        axis = kmeans_centers[0] - kmeans_centers[1]        
        return axis, kmeans_centers[0], kmeans_centers[1]
        
    def predict_subclusters(data, axis):
        projections = data @ axis
        sorted_indices = projections.argsort()
        dip_value, modal_interval, modal_triangle = dip_test(projections[sorted_indices], is_data_sorted=True, just_dip=False)
        index_lower, index_upper = modal_interval
        index_tri1, index_tri2, index_tri3 = modal_triangle
        if projections[sorted_indices[index_tri2]] > projections[sorted_indices[index_upper]]:
                treshhold =  (projections[sorted_indices[index_tri2]] + projections[sorted_indices[index_upper]])/2
        else:
                treshhold =  (projections[sorted_indices[index_tri2]] + projections[sorted_indices[index_lower]])/2
        labels = np.zeros(len(data))
        labels[projections >= treshhold] = 1
        labels[sorted_indices[index_tri2]] = 2
        labels[sorted_indices[index_lower]] = 3
        labels[sorted_indices[index_upper]] = 3
        return labels

    # Combine the datasets
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2 + 1))  # Adjust labels for the second cluster

    dataloader = get_dataloader(X, 50 )
    encode = lambda x: x
    autoencoder = type("Autoencoder", (), {"encode": encode})

    axis, c1, c2 = get_inital_projection_axis(X)

    labels = predict_subclusters(X, axis)


    # Create a scatter plot
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', marker='o')

    # origin = [np.mean(X, axis=0)[0]], [np.mean(X, axis=0)[1]]  # Vector origin point
    # plt.quiver(*origin, axis[0], axis[1], scale=5, color='red', label='Direction Vector')
    plt.plot([c1[0], c2[0]], [c1[1], c2[1]], color='red',linewidth=2.5, label='Line between points')

    # Add title and labels
    plt.title('Scatter Plot of 2 Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Show plot
    plt.show()




def visualize_axis_after_tree_growth(X):
        # Generate sample data

    # X1, y1 = make_blobs(n_samples=50, centers=1)
    # X2, y2 = make_blobs(n_samples=200, centers=1)
    # X3, y3 = make_blobs(n_samples=100, centers=1)
    # X4, y4 = make_blobs(n_samples=250, centers=1)

    # Combine the datasets
    # X = np.vstack((X1, X2, X3, X4))
    # y = np.hstack((y1, y2 + 1, y3 + 2, y4 + 3))  

    dataloader = get_dataloader(X, 50 )
    encode = lambda x: x
    autoencoder = type("Autoencoder", (), {"encode": encode})

    tree = Cluster_Tree(dataloader, autoencoder, None, "cpu", max_leaf_nodes=10)

    tree.assign_to_tree(torch.from_numpy(X))

    # Create a scatter plot
    plt.scatter(tree.root.lower_projection_child.assignments[:, 0], tree.root.lower_projection_child.assignments[:, 1], c="blue", marker='o')
    plt.scatter(tree.root.higher_projection_child.assignments[:, 0], tree.root.higher_projection_child.assignments[:, 1], c="red", marker='o')

    origin = [np.mean(X, axis=0)[0]], [np.mean(X, axis=0)[1]]  # Vector origin point
    plt.quiver(*origin, tree.root.projection_axis.data[0], tree.root.projection_axis.data[1], scale=5, color='red', label='Direction Vector')

    # Add title and labels
    plt.title('Scatter Plot of 2 Clusters')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Show plot


    tree.grow_tree(dataloader, autoencoder, None, 0.0)
    tree.assign_to_tree(torch.from_numpy(X))

    pred_labels = np.ones(len(X))*-1

    if tree.root.lower_projection_child.is_leaf_node():
        plt.scatter(tree.root.higher_projection_child.lower_projection_child.assignments[:, 0],tree.root.higher_projection_child.lower_projection_child.assignments[:, 1], c="green", marker='o')
        plt.scatter(tree.root.higher_projection_child.higher_projection_child.assignments[:, 0], tree.root.higher_projection_child.higher_projection_child.assignments[:, 1], c="orange", marker='o')

        origin = [np.mean(tree.root.higher_projection_child.assignments.numpy(), axis=0)[0]], [np.mean(tree.root.higher_projection_child.assignments.numpy(), axis=0)[1]]  # Vector origin point
        plt.quiver(*origin, tree.root.higher_projection_child.projection_axis.data[0], tree.root.higher_projection_child.projection_axis.data[1], scale=5, color='brown', label='Direction Vector')
        X_combined = torch.cat((tree.root.higher_projection_child.lower_projection_child.assignments, tree.root.higher_projection_child.higher_projection_child.assignments), dim=0)
        
        X_combined_sorted, _ = torch.sort(X_combined, dim=0)
        X_sorted, _ = torch.sort(tree.root.higher_projection_child.assignments, dim=0)

        # Check if the sorted tensors are identical
        is_identical = torch.equal(X_sorted, X_combined_sorted)
        print(is_identical)
        pred_labels[tree.root.higher_projection_child.lower_projection_child.assignment_indices] = 0
        pred_labels[tree.root.higher_projection_child.higher_projection_child.assignment_indices] = 1
        pred_labels[tree.root.lower_projection_child.assignment_indices] = 2
    else:
        plt.scatter(tree.root.lower_projection_child.lower_projection_child.assignments[:, 0],tree.root.lower_projection_child.lower_projection_child.assignments[:, 1], c="green", marker='o')
        plt.scatter(tree.root.lower_projection_child.higher_projection_child.assignments[:, 0], tree.root.lower_projection_child.higher_projection_child.assignments[:, 1], c="orange", marker='o')
        origin = [np.mean(tree.root.lower_projection_child.assignments.numpy(), axis=0)[0]], [np.mean(tree.root.lower_projection_child.assignments.numpy(), axis=0)[1]]  # Vector origin point
        plt.quiver(*origin, tree.root.lower_projection_child.projection_axis.data[0], tree.root.lower_projection_child.projection_axis.data[1], scale=5, color='brown', label='Direction Vector')

        X_combined = torch.cat((tree.root.lower_projection_child.lower_projection_child.assignments, tree.root.lower_projection_child.higher_projection_child.assignments), dim=0)

        X_combined_sorted, _ = torch.sort(X_combined, dim=0)
        X_sorted, _ = torch.sort(tree.root.lower_projection_child.assignments, dim=0)

        # Check if the sorted tensors are identical
        is_identical = torch.equal(X_sorted, X_combined_sorted)
        print(is_identical)
        pred_labels[tree.root.lower_projection_child.lower_projection_child.assignment_indices] = 0
        pred_labels[tree.root.lower_projection_child.higher_projection_child.assignment_indices] = 1
        pred_labels[tree.root.higher_projection_child.assignment_indices] = 2

    plt.show()
