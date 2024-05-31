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


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(random.randint(0, 1000))
    torch.manual_seed(random.randint(0, 1000))
    torch.cuda.manual_seed_all(random.randint(0, 1000))
    torch.backends.cudnn.deterministic = True


def ae_bisecting(data, labels, ae_module, max_leaf_nodes, n_cluster, seed, device):
    set_random_seed(seed)
    embedded_data = None
    for batch_data in torch.utils.data.DataLoader(data, batch_size=256, shuffle=False):
        
        embedded_batch_np = (
                ae_module.encode(batch_data.to(device)).detach().cpu().numpy()
            )

        if embedded_data is None:
            embedded_data = embedded_batch_np
        else:
            embedded_data = np.concatenate([embedded_data, embedded_batch_np], 0)
    del ae_module

    tree = bisection(max_leaf_nodes, embedded_data)
    bisec_labels = predict_by_tree(tree, embedded_data, n_cluster)
    bisec_tree = predict_id_tree(tree, embedded_data)
    bisec_den = dendrogram_purity(bisec_tree, labels)
    bisec_lp = leaf_purity(bisec_tree, labels)
    return bisec_den, bisec_lp


def ae_single(data, labels, ae_module, max_leaf_nodes, n_clusters, seed, device):
    set_random_seed(seed)
    embedded_data = None
    for batch_data in torch.utils.data.DataLoader(data, batch_size=256, shuffle=False):
        embedded_batch_np = (
                ae_module.encode(batch_data.to(device)).detach().cpu().numpy()
            )
        if embedded_data is None:
            embedded_data = embedded_batch_np
        else:
            embedded_data = np.concatenate([embedded_data, embedded_batch_np], 0)
    del ae_module
    single_cluster = AgglomerativeClustering(
        compute_full_tree=True, n_clusters=n_clusters, linkage="single"
    ).fit(embedded_data)

    single_labels = single_cluster.labels_
    single_purity_tree = prune_dendrogram_purity_tree(
        to_dendrogram_purity_tree(single_cluster.children_), max_leaf_nodes
    )
    single_purity = dendrogram_purity(single_purity_tree, labels)
    single_lp = leaf_purity(single_purity_tree, labels)

    return single_purity, single_lp


def ae_complete(data, labels, ae_module, max_leaf_nodes, n_clusters, seed, device):
    set_random_seed(seed)
    embedded_data = None
    for batch_data in torch.utils.data.DataLoader(data, batch_size=256, shuffle=False):
        embedded_batch_np = (
                ae_module.encode(batch_data.to(device)).detach().cpu().numpy()
            )
        if embedded_data is None:
            embedded_data = embedded_batch_np
        else:
            embedded_data = np.concatenate([embedded_data, embedded_batch_np], 0)
    del ae_module
    complete_cluster = AgglomerativeClustering(
        compute_full_tree=True, n_clusters=n_clusters, linkage="complete"
    ).fit(embedded_data)
    complete_labels = complete_cluster.labels_
    complete_purity_tree = prune_dendrogram_purity_tree(
        to_dendrogram_purity_tree(complete_cluster.children_), max_leaf_nodes
    )
    complete_purity = dendrogram_purity(complete_purity_tree, labels)
    complete_lp = leaf_purity(complete_purity_tree, labels)

    return complete_purity, complete_lp
