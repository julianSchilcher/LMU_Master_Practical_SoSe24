import torch
import torch.nn.functional as F
import torch.utils.data
import random
import numpy as np
import torch
import sys
import os

sys.path.append(os.getcwd())

from practical.DeepClustering.DeepECT.baseline_hierachical.methods.dendrogram_purity import *
from practical.DeepClustering.DeepECT.baseline_hierachical.methods.bisceting_kmeans import *
from sklearn.cluster import AgglomerativeClustering


def ae_bisecting(
    dataloader: torch.utils.data.DataLoader,
    labels,
    ae_module,
    max_leaf_nodes,
    device,
):
    embedded_data = []
    for batch_data in dataloader:
        embedded_data.append(
            ae_module.encode(batch_data[1].to(device)).detach().cpu().numpy()
        )

    tree = bisection(max_leaf_nodes, np.concatenate(embedded_data))
    bisec_tree = predict_id_tree(tree, embedded_data)
    bisec_den = dendrogram_purity(bisec_tree, labels)
    bisec_lp = leaf_purity(bisec_tree, labels)
    return (
        bisec_den,
        bisec_lp,
    )


def ae_single(
    dataloader: torch.utils.data.DataLoader,
    labels,
    ae_module,
    max_leaf_nodes,
    n_clusters,
    device,
):
    embedded_data = []
    for batch_data in dataloader:
        embedded_data.append(
            ae_module.encode(batch_data[1].to(device)).detach().cpu().numpy()
        )
    single_cluster = AgglomerativeClustering(
        compute_full_tree=True, n_clusters=n_clusters, linkage="single"
    ).fit(np.concatenate(embedded_data))

    single_purity_tree = prune_dendrogram_purity_tree(
        to_dendrogram_purity_tree(single_cluster.children_), max_leaf_nodes
    )
    single_purity = dendrogram_purity(single_purity_tree, labels)
    single_lp = leaf_purity(single_purity_tree, labels)

    return single_purity, single_lp


def ae_complete(
    dataloader: torch.utils.data.DataLoader,
    labels,
    ae_module,
    max_leaf_nodes,
    n_clusters,
    device,
):
    embedded_data = []
    for batch_data in dataloader:
        embedded_data.append(
            ae_module.encode(batch_data[1].to(device)).detach().cpu().numpy()
        )
    complete_cluster = AgglomerativeClustering(
        compute_full_tree=True, n_clusters=n_clusters, linkage="complete"
    ).fit(np.concatenate(embedded_data))
    complete_purity_tree = prune_dendrogram_purity_tree(
        to_dendrogram_purity_tree(complete_cluster.children_), max_leaf_nodes
    )
    complete_purity = dendrogram_purity(complete_purity_tree, labels)
    complete_lp = leaf_purity(complete_purity_tree, labels)

    return complete_purity, complete_lp
