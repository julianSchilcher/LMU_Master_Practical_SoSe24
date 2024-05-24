from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from scipy.special import comb

from practical.DeepClustering.DeepECT.pre_training.vae.functional import (
    count_values_in_sequence,
)


class PurityNode:
    def __init__(
        self,
        id: int,
        parent: "PurityNode" = None,
        left_child: "PurityNode" = None,
        right_child: "PurityNode" = None,
    ) -> "PurityNode":
        self.id = id
        self.parent: PurityNode = parent
        self.left_child: PurityNode = left_child
        self.right_child: PurityNode = right_child
        self.assigned_indices: List[int] = []

    def assign_batch(
        self, dataset_indices: torch.Tensor, assigned_batch_indices: torch.Tensor
    ):
        self.assigned_indices.extend(dataset_indices[assigned_batch_indices].tolist())

    @property
    def assignments(self):
        return np.asarray(sorted(self.assigned_indices), dtype=np.int16)

    @property
    def is_leaf(self):
        return self.left_child is None and self.right_child is None


class PurityTree:
    def __init__(self) -> None:
        pass


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    # https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


def leaf_purity(tree_root: PurityNode, ground_truth):
    values = []
    weights = []

    def get_leaf_purities(node: PurityNode):
        nonlocal values
        nonlocal weights
        if node.is_leaf:
            node_total_dp_count = len(node.assignments)
            node_per_label_counts = count_values_in_sequence(
                [ground_truth[id] for id in node.assignments]
            )
            if node_total_dp_count > 0:
                purity_rate = max(node_per_label_counts.values()) / node_total_dp_count
            else:
                purity_rate = 1.0
            values.append(purity_rate)
            weights.append(node_total_dp_count)
        else:
            get_leaf_purities(node.left_child)
            get_leaf_purities(node.right_child)

    get_leaf_purities(tree_root)

    return weighted_avg_and_std(values, weights)


def dendrogram_purity(tree_root: PurityNode, ground_truth) -> float:
    total_per_label_frequencies = count_values_in_sequence(ground_truth)
    total_per_label_pairs_count = {
        k: comb(v, 2, True) for k, v in total_per_label_frequencies.items()
    }
    total_n_of_pairs = sum(total_per_label_pairs_count.values())

    one_div_total_n_of_pairs = 1.0 / total_n_of_pairs

    purity = 0.0

    def calculate_purity(node: PurityNode) -> Tuple[Dict[Any, int], int]:
        nonlocal purity
        if node.is_leaf:
            node_total_dp_count = len(node.assignments)
            node_per_label_frequencies = count_values_in_sequence(
                [ground_truth[id] for id in node.assignments]
            )
            node_per_label_pairs_count: Dict[Any, int] = {
                k: comb(v, 2, True) for k, v in node_per_label_frequencies.items()
            }

        else:  # it is an inner node
            left_child_per_label_freq, left_child_total_dp_count = calculate_purity(
                node.left_child
            )
            right_child_per_label_freq, right_child_total_dp_count = calculate_purity(
                node.right_child
            )
            node_total_dp_count = left_child_total_dp_count + right_child_total_dp_count
            node_per_label_frequencies: Dict[Any, int] = {
                k: left_child_per_label_freq.get(k, 0)
                + right_child_per_label_freq.get(k, 0)
                for k in set(left_child_per_label_freq)
                | set(right_child_per_label_freq)
            }

            node_per_label_pairs_count: Dict[Any, int] = {
                k: left_child_per_label_freq.get(k) * right_child_per_label_freq.get(k)
                for k in set(left_child_per_label_freq)
                & set(right_child_per_label_freq)
            }

        for label, pair_count in node_per_label_pairs_count.items():
            label_freq = node_per_label_frequencies[label]
            label_pairs = pair_count
            purity += (
                one_div_total_n_of_pairs
                * label_freq
                / node_total_dp_count
                * label_pairs
            )
        return node_per_label_frequencies, node_total_dp_count

    calculate_purity(tree_root)
    return purity
