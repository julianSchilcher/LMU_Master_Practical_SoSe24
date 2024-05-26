from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from scipy.special import comb
from collections import defaultdict
import clustpy.metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score


class PredictionClusterNode:
    def __init__(
        self,
        id: int,
        split_id: int,
        center: np.ndarray,
        parent: "PredictionClusterNode" = None,
        left_child: "PredictionClusterNode" = None,
        right_child: "PredictionClusterNode" = None,
    ) -> "PredictionClusterNode":
        self.id = id
        self.split_id = split_id
        self.parent: PredictionClusterNode = parent
        self.left_child: PredictionClusterNode = left_child
        self.right_child: PredictionClusterNode = right_child
        self.assigned_indices: List[int] = []
        self.center = center

    def assign_batch(
        self,
        dataset_indices: torch.Tensor,
        assigned_batch_indices: Union[torch.Tensor | None],
    ):
        if assigned_batch_indices is not None:
            self.assigned_indices.extend(
                dataset_indices[assigned_batch_indices].tolist()
            )

    @property
    def assignments(self):
        return sorted(self.assigned_indices)

    @property
    def is_leaf(self):
        return self.left_child is None and self.right_child is None


def count_values_in_sequence(seq: np.ndarray):
    res = defaultdict(int)
    for key in seq:
        res[key] += 1
    return dict(res)


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


def leaf_purity(tree_root: PredictionClusterNode, ground_truth: np.ndarray):
    values = []
    weights = []

    def get_leaf_purities(node: PredictionClusterNode):
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


def dendrogram_purity(
    tree_root: PredictionClusterNode, ground_truth: np.ndarray
) -> float:
    total_per_label_frequencies = count_values_in_sequence(ground_truth)
    total_per_label_pairs_count = {
        k: comb(v, 2, True) for k, v in total_per_label_frequencies.items()
    }
    total_n_of_pairs = sum(total_per_label_pairs_count.values())

    one_div_total_n_of_pairs = 1.0 / total_n_of_pairs

    purity = 0.0

    def calculate_purity(node: PredictionClusterNode) -> Tuple[Dict[Any, int], int]:
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


class PredictionClusterTree:
    def __init__(self, root_node: "PredictionClusterNode") -> None:
        self.root = root_node

    def __getitem__(self, id):
        def find_idx_recursive(node: PredictionClusterNode):
            if node.id == id:
                return node
            if not node.is_leaf:
                left = find_idx_recursive(node.left_child)
                if left is not None:
                    return left
                right = find_idx_recursive(node.right_child)
                if right is not None:
                    return right
            return None

        found_node = find_idx_recursive(self.root)
        if found_node is not None:
            return found_node
        raise IndexError(f"Node with id: {id} not found")

    @property
    def leaf_nodes(self) -> List[PredictionClusterNode]:
        def get_nodes_recursive(node: PredictionClusterNode):
            result = []
            if node.is_leaf:
                result.append(node)
                return result
            result.extend(get_nodes_recursive(node.left_child))
            result.extend(get_nodes_recursive(node.right_child))
            return result

        return get_nodes_recursive(self.root)

    @property
    def nodes(self):
        def get_nodes_recursive(node: PredictionClusterNode):
            result = [node]
            if node.is_leaf:
                return result
            result.extend(get_nodes_recursive(node.left_child))
            result.extend(get_nodes_recursive(node.right_child))
            return result

        return get_nodes_recursive(self.root)

    def __aggregate_assignments(self):
        def aggregate_nodes_recursive(node: PredictionClusterNode):
            if node.is_leaf:
                return node.assigned_indices
            node.assigned_indices.clear()
            node.assigned_indices.extend(aggregate_nodes_recursive(node.left_child))
            node.assigned_indices.extend(aggregate_nodes_recursive(node.right_child))
            return node.assigned_indices

        aggregate_nodes_recursive(self.root)

    def get_k_clusters(self, k: int) -> List[PredictionClusterNode]:
        self.__aggregate_assignments()
        result_nodes = []
        max_split_level = sorted(list(set([node.split_id for node in self.nodes])))[
            k - 1
        ]

        # the leaf nodes after the first <k> - 1 growing steps (splits) are the nodes representing the <k> clusters
        def get_nodes_at_split_level(node: PredictionClusterNode):
            if (
                node.is_leaf or node.left_child.split_id > max_split_level
            ) and node.split_id <= max_split_level:
                result_nodes.append(node)
                return
            get_nodes_at_split_level(node.left_child)
            get_nodes_at_split_level(node.right_child)

        get_nodes_at_split_level(self.root)
        # consistency check
        assert (
            len(result_nodes) == k
        ), "Number of cluster nodes doesn't correspond to number of classes"
        return result_nodes

    def get_k_cluster_predictions(self, ground_truth: np.ndarray, k: int):
        predictions = np.zeros_like(ground_truth, dtype=np.int32)
        for i, cluster in enumerate(self.get_k_clusters(k)):
            predictions[cluster.assignments] = i
        return predictions

    def dendrogram_purity(self, ground_truth: np.ndarray):
        return dendrogram_purity(self.root, ground_truth)

    def leaf_purity(self, ground_truth: np.ndarray):
        return leaf_purity(self.root, ground_truth)

    def flat_accuracy(self, ground_truth: np.ndarray, n_clusters: int):
        return clustpy.metrics.unsupervised_clustering_accuracy(
            ground_truth, self.get_k_cluster_predictions(ground_truth, n_clusters)
        )

    def flat_nmi(self, ground_truth: np.ndarray, n_clusters: int):
        return normalized_mutual_info_score(
            ground_truth, self.get_k_cluster_predictions(ground_truth, n_clusters)
        )

    def flat_ari(self, ground_truth: np.ndarray, n_clusters: int):
        return adjusted_rand_score(
            ground_truth, self.get_k_cluster_predictions(ground_truth, n_clusters)
        )