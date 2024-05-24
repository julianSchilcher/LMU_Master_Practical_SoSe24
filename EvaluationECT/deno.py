from queue import Queue
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.utils.data
from clustpy.deep._train_utils import get_standard_initial_deep_clustering_setting
from clustpy.deep._utils import set_torch_seed
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.utils import check_random_state
from clustpy.data.real_torchvision_data import load_mnist
from clustpy.metrics.clustering_metrics import unsupervised_clustering_accuracy
from tqdm import tqdm
import math
import attr
from ect.utils.statistics import weighted_avg_and_std
from ect.utils.functional import count_values_in_sequence
from scipy.special import comb

# Cluster_Node class definition as provided earlier

@attr.s(cmp=False)
class DpNode:
    left_child = attr.ib()
    right_child = attr.ib()
    node_id = attr.ib()

    @property
    def children(self):
        return [self.left_child, self.right_child]

    @property
    def is_leaf(self):
        return False

@attr.s(cmp=False)
class DpLeaf:
    dp_ids = attr.ib()
    node_id = attr.ib()

    @property
    def children(self):
        return []

    @property
    def is_leaf(self):
        return True

def combine_to_trees(tree_a, tree_b):
    def recursive(ta, tb):
        if ta.is_leaf != tb.is_leaf or ta.node_id != tb.node_id:
            print(f"{ta.node_id} != {tb.node_id}")
            raise RuntimeError("Trees are not equivalent!")
        if ta.is_leaf:
            return DpLeaf(ta.dp_ids + tb.dp_ids, ta.node_id)
        else:
            left_child = recursive(ta.left_child, tb.left_child)
            right_child = recursive(ta.right_child, tb.right_child)
            return DpNode(left_child, right_child, ta.node_id)

    return recursive(tree_a, tree_b)

def leaf_purity(tree_root, ground_truth):
    values = []
    weights = []

    def get_leaf_purities(node):
        nonlocal values
        nonlocal weights
        if node.is_leaf:
            node_total_dp_count = len(node.dp_ids)
            node_per_label_counts = count_values_in_sequence([ground_truth[id] for id in node.dp_ids])
            purity_rate = max(node_per_label_counts.values()) / node_total_dp_count if node_total_dp_count > 0 else 1.0
            values.append(purity_rate)
            weights.append(node_total_dp_count)
        else:
            get_leaf_purities(node.left_child)
            get_leaf_purities(node.right_child)

    get_leaf_purities(tree_root)

    return weighted_avg_and_std(values, weights)

def dendrogram_purity(tree_root, ground_truth):
    total_per_label_frequencies = count_values_in_sequence(ground_truth)
    total_per_label_pairs_count = {k: comb(v, 2, exact=True) for k, v in total_per_label_frequencies.items()}
    total_n_of_pairs = sum(total_per_label_pairs_count.values())

    if total_n_of_pairs == 0:
        return 1.0

    one_div_total_n_of_pairs = 1.0 / total_n_of_pairs
    purity = 0.0

    def calculate_purity(node):
        nonlocal purity
        if node.is_leaf:
            node_total_dp_count = len(node.dp_ids)
            node_per_label_frequencies = count_values_in_sequence([ground_truth[id] for id in node.dp_ids])
            node_per_label_pairs_count = {k: comb(v, 2, exact=True) for k, v in node_per_label_frequencies.items()}
        else:
            left_child_per_label_freq, left_child_total_dp_count = calculate_purity(node.left_child)
            right_child_per_label_freq, right_child_total_dp_count = calculate_purity(node.right_child)
            node_total_dp_count = left_child_total_dp_count + right_child_total_dp_count
            node_per_label_frequencies = {k: left_child_per_label_freq.get(k, 0) + right_child_per_label_freq.get(k, 0)
                                          for k in set(left_child_per_label_freq) | set(right_child_per_label_freq)}
            node_per_label_pairs_count = {k: left_child_per_label_freq.get(k, 0) * right_child_per_label_freq.get(k, 0)
                                          for k in set(left_child_per_label_freq) & set(right_child_per_label_freq)}

        for label, pair_count in node_per_label_pairs_count.items():
            label_freq = node_per_label_frequencies[label]
            label_pairs = pair_count
            purity += one_div_total_n_of_pairs * label_freq / node_total_dp_count * label_pairs

        return node_per_label_frequencies, node_total_dp_count

    calculate_purity(tree_root)
    return purity

# Testing the methods with sample data
if __name__ == "__main__":
    l4 = DpLeaf([0], 4)
    l7 = DpLeaf([1], 7)
    l8 = DpLeaf([2], 8)
    l5 = DpLeaf([3], 5)
    l6 = DpLeaf([4], 6)
    n3 = DpNode(l7, l8, 3)
    n1 = DpNode(l4, n3, 1)
    n2 = DpNode(l5, l6, 2)
    n0 = DpNode(n1, n2, 0)
    test_target = np.asarray([0, 1, 1, 2, 3])
    
    # Test leaf_purity
    leaf_purity_value = leaf_purity(n0, test_target)
    print(f"Leaf Purity: {leaf_purity_value}")
    
    # Test dendrogram_purity
    dendrogram_purity_value = dendrogram_purity(n0, test_target)
    print(f"Dendrogram Purity: {dendrogram_purity_value}")
