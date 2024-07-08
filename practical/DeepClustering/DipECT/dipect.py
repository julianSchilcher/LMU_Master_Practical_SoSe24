import datetime
import logging
import math
import os
import sys

sys.path.append(os.getcwd())

import warnings
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.utils.data
from clustpy.deep._data_utils import augmentation_invariance_check, get_dataloader
from clustpy.deep._train_utils import get_trained_network
from clustpy.deep._utils import detect_device, encode_batchwise, set_torch_seed
from clustpy.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder
from clustpy.deep.dipencoder import _Dip_Gradient
from clustpy.utils import dip_pval, dip_test
from clustpy.data import load_fmnist, load_mnist, load_usps, load_reuters, load_cifar10
from clustpy.deep.autoencoders import FeedforwardAutoencoder, ConvolutionalAutoencoder
import sklearn
from ray import train
from sklearn.cluster import KMeans
from tqdm import tqdm

from practical.DeepClustering.DeepECT.metrics import (
    PredictionClusterNode,
    PredictionClusterTree,
)


# replaces the dip module
class Cluster_Node:
    """pruning_indicator
    This class represents a cluster node within a binary cluster tree. Each node in a cluster tree represents a cluster.
    Each inner node in the tree stores a projection axis used for the dip test.
    """

    def __init__(
        self,
        device: torch.device,
        id: int = 0,
        parent: "Cluster_Node" = None,
        split_id: int = 0,
        split_level: int = 0,
        number_assignments: int = 0,  # used to initialise pruning_indicator
    ) -> "Cluster_Node":
        """
        Constructor for class Cluster_Node

        Parameters
        ----------
        device : torch.device
            The device to be trained on.
        id : int, optional
            The ID of the node, by default 0.
        parent : Cluster_Node, optional
            The parent node, by default None.
        split_id : int, optional
            The ID of the split, by default 0.
        split_level : int, optional
            The level of the node in the cluster tree, by default 0.
        number_assignments : int, optional
            Number of assignments for this node to (re-)initialise pruning indicator, by default 0.

        Returns
        -------
        Cluster_Node
            The initialized Cluster_Node object.
        """
        self.device = device
        self.pruning_indicator = float(number_assignments)
        self.higher_projection_child = None
        self.lower_projection_child = None
        self.projection_axis = None
        self.assignments: Union[torch.Tensor, None] = None
        self.assignment_indices: Union[torch.Tensor, None] = None
        self.id = id
        self.split_id = split_id
        self.split_level = split_level
        self.parent = parent
        self.check_invariant()

    def check_invariant(self):
        """
        Class invariant for assuring that the following bidirectional implication holds:
        leaf node <=> (projection_axis==None)
        """
        if not (
            (self.is_leaf_node() and self.projection_axis is None)
            or (not self.is_leaf_node() and self.projection_axis is not None)
        ):
            raise RuntimeError("Bad state: a leaf node stores a projection axis")

    def clear_assignments(self):
        """
        Clears all assignments in the cluster node.
        """
        if self.higher_projection_child is not None:
            self.higher_projection_child.clear_assignments()
        if self.lower_projection_child is not None:
            self.lower_projection_child.clear_assignments()
        self.assignments = None
        self.assignment_indices = None

    def is_leaf_node(self) -> bool:
        """
        Checks if this node is a leaf node.

        Returns
        -------
        bool
            True if the node is a leaf node, else False.
        """
        return (
            self.higher_projection_child is None and self.lower_projection_child is None
        )

    def prune(self):
        """
        Prunes the tree by removing all nodes below this node.
        """
        if self.higher_projection_child is not None:
            self.higher_projection_child.prune()
        if self.lower_projection_child is not None:
            self.lower_projection_child.prune()

        if not self.is_leaf_node():
            self.projection_axis.requires_grad = False
        self.assignments = None
        self.assignment_indices = None

    def expand_tree(
        self,
        projection_axis: np.ndarray,
        projection_axis_optimizer: Union[torch.optim.Optimizer, None],
        num_assignments_higher_projection_child: int,
        num_assignments_lower_projection_child: int,
        max_id: int = 0,
        max_split_id: int = 0,
    ):
        """
        Set new children to this cluster node, thus changing this node to an inner node.

        Parameters
        ----------
        projection_axis: np.ndarray
            The projection axis used for the dip test
            (used for cluster loss and assigning data to its childs (top.down approach))
        projection_axis_optimizer : Union[torch.optim.Optimizer, None]
            The optimizer used for improving the projection axes.
        num_assignments_higher_projection_child : int
            Number of initial assignments for the higher projection child - used to initialise
            the pruning indicator.
        num_assignments_lower_projection_child : int
            Number of initial assignments for the lower projection child - used to initialise
            the pruning indicator.
        max_id : int, optional
            The maximum ID, by default 0.
        max_split_id : int, optional
            The maximum split ID, by default 0.
        """
        # set projection axis
        self.projection_axis = torch.nn.Parameter(
            torch.nn.Parameter(torch.from_numpy(projection_axis).float())
        )

        if projection_axis_optimizer is not None:
            projection_axis_optimizer.add_param_group({"params": self.projection_axis})

        self.higher_projection_child = Cluster_Node(
            self.device,
            max_id + 1,
            self,
            max_split_id + 1,
            self.split_level + 1,
            num_assignments_higher_projection_child,
        )
        self.lower_projection_child = Cluster_Node(
            self.device,
            max_id + 2,
            self,
            max_split_id + 1,
            self.split_level + 1,
            num_assignments_lower_projection_child,
        )
        self.check_invariant()

    def add_projection_axis_to_optimizer(
        self, optimizer: torch.optim.Optimizer, new_axis: torch.nn.Parameter
    ):
        """
        Add new projection axis to the optimizer's param group 'projection_axes'

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer used.
        new_axis: torch.nn.Parameter
            The new projection axis which should be added to the optimizer's param group 'projection_axes'
        """
        for param_group in optimizer.param_groups:
            if param_group.get("name") == "projection_axes":
                param_group["params"].extend(
                    [new_axis]
                )  # optimizer expects a list of parameters
                return

        raise ValueError(
            "Parameter group with with name projection_axes not initialised yet. Please initialise it by calling optimizer.add_param_group({'params': [], 'lr': desired_learning_rate, 'name': 'projection_axes'},)"
        )

    def adapt_pruning_indicator(self, pruning_factor: float, number_assignments: int):
        """
        Adapt pruning treshhold based on the new number of assignments with an exponential moving average.

        Parameters
        ----------
        number_assignments : int
            The number of assignments to this node
        """
        # adapt pruning indicator with EMA
        self.pruning_indicator = pruning_factor * (
            self.pruning_indicator + number_assignments
        )


class Cluster_Tree:
    """
    This class represents a binary cluster tree. It provides multiple
    functionalities used for improving the cluster tree, like calculating
    the cluster loss and assigning samples in a top-down manner. Furthermore,
    it provides methods for growing and pruning the tree.
    """

    def __init__(
        self,
        trainloader: torch.utils.data.DataLoader,
        autoencoder: torch.nn.Module,
        projection_axis_optimizer: torch.optim.Optimizer,
        device: torch.device,
        random_seed: np.random.RandomState,
    ) -> "Cluster_Tree":
        """
        Constructor for the Cluster_Tree class.

        Parameters
        ----------
        trainloader : torch.utils.data.DataLoader
            Dataloader used to initialize the cluster tree (root and its 2 children).
        autoencoder : torch.nn.Module
            The autoencoder used to embed the data of the dataloader.
        projection_axis_optimizer : torch.optim.Optimizer
            Optimizer for improving the projection axes.
        device : torch.device
            The device to be trained on.

        Returns
        -------
        Cluster_Tree
            The initialized Cluster_Tree object.
        """
        # initialize cluster tree
        self.device = device
        self.random_seed = random_seed
        embedded_data = encode_batchwise(trainloader, autoencoder)
        axis, number_left_assignments, number_right_assignments = (
            self.get_inital_projection_axis(embedded_data)
        )
        self.root = Cluster_Node(device, number_assignments=len(embedded_data))
        self.root.expand_tree(
            axis,
            projection_axis_optimizer,
            number_right_assignments,
            number_left_assignments,
        )

    @property
    def number_nodes(self):
        """
        Calculates the total number of nodes in the tree.

        Returns
        -------
        int
            The total number of nodes.
        """

        def count_recursive(node: Cluster_Node):
            if node.is_leaf_node():
                return 1
            return (
                1
                + count_recursive(node.higher_projection_child)
                + count_recursive(node.lower_projection_child)
            )

        return count_recursive(self.root)

    @property
    def nodes(self) -> List[Cluster_Node]:
        """
        Gets the list of all nodes in the tree.

        Returns
        -------
        List[Cluster_Node]
            The list of all nodes.
        """

        def get_nodes_recursive(node: Cluster_Node):
            result = [node]
            if node.is_leaf_node():
                return result
            result.extend(get_nodes_recursive(node.higher_projection_child))
            result.extend(get_nodes_recursive(node.lower_projection_child))
            return result

        return get_nodes_recursive(self.root)

    @property
    def leaf_nodes(self) -> List[Cluster_Node]:
        """
        Gets the list of all leaf nodes in the tree.

        Returns
        -------
        List[Cluster_Node]
            The list of all leaf nodes.
        """

        def get_nodes_recursive(node: Cluster_Node):
            result = []
            if node.is_leaf_node():
                result.append(node)
                return result
            result.extend(get_nodes_recursive(node.higher_projection_child))
            result.extend(get_nodes_recursive(node.lower_projection_child))
            return result

        return get_nodes_recursive(self.root)

    def get_inital_projection_axis(self, embedded_data: np.ndarray):
        """
        Returns the initial projection axis for the given data as well as the
        number of assignments to the 2 clusters. The axis is defined through the resulting
        centers from applying KMeans to the given data.

        Parameters
        ----------
        embedded_data : np.ndarray
            Embedded data samples of shape [#Samples, Dimensionality]

        Returns
        -------
        axis : np.array
            The found projection axis.
        number_of_samples_cluster_0 : int
            Number of assigned samples to cluster 0.
        number_of_samples_cluster_1 : int
            Number of assigned samples to cluster 1.
        """
        # init projection axis on full dataset
        kmeans = KMeans(n_clusters=2, n_init=20, random_state=self.random_seed).fit(
            embedded_data
        )
        kmeans_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        # higher projection by cluster 1 since axis points to cluster 1
        return (
            kmeans_centers[0] - kmeans_centers[1],
            np.sum(labels == 0),
            np.sum(labels == 1),
        )

    def clear_node_assignments(self):
        """
        Clears the assignments for all nodes in the tree.
        """
        self.root.clear_assignments()

    def assign_to_tree(
        self,
        data_embedded: torch.Tensor,
        pruning_strategy: Union[str | None] = None,
        pruning_factor: Union[float | None] = None,
        set_pruning_incidator: bool = False,
    ):
        """
        Assigns the given data recursively top-down to the cluster tree.

        Parameters
        ----------
        data_embedded : torch.Tensor
            Embedded data samples of shape [#Samples, Dimensionality].
        pruning_strategy : Union[str, None], optional
            Strategy to use for pruning, by default None.
        pruning_factor : Union[float, None], optional
            Factor to use for pruning, by default None.
        set_pruning_incidator : bool, optional
            Whether to update the pruning threshold, by default False.
        """
        # clear all assignments
        self.clear_node_assignments()
        # assign top-down
        self.assign_top_down(
            self.root,
            data_embedded,
            torch.tensor([i for i in range(len(data_embedded))]),
            set_pruning_incidator,
            pruning_strategy,
            pruning_factor,
        )

    def assign_top_down(
        self,
        node: Cluster_Node,
        embedded_data: torch.Tensor,
        embedded_data_indices: torch.Tensor,
        set_pruning_incidator: bool,
        pruning_strategy: str,
        pruning_factor: float,
    ):
        """
        Helper function which assigns the given data to the given node and divides the given data to its children if they exist.

        Parameters
        ----------
        node : Cluster_Node
            The node object which should get the given data assigned.
        embedded_data : torch.Tensor
            Embedded data samples of shape [#Samples, Dimensionality].
        embedded_data_indices : torch.Tensor
            Indices of the embedded data samples in the batch [#Samples, ].
        set_pruning_incidator : bool
            Whether to update the pruning threshold.
        pruning_strategy : str
            The pruning strategy to use.
        pruning_factor : float
            The pruning factor to use.
        """

        if set_pruning_incidator:
            node.adapt_pruning_indicator(pruning_factor, len(embedded_data))

        if embedded_data.numel() == 0:
            return

        node.assignments = embedded_data
        node.assignment_indices = embedded_data_indices
        if node.is_leaf_node():
            return

        labels = self.predict_subclusters(node)
        if node.higher_projection_child is not None:
            self.assign_top_down(
                node.higher_projection_child,
                embedded_data[labels == 1],
                embedded_data_indices[labels == 1],
                set_pruning_incidator,
                pruning_strategy,
                pruning_factor,
            )
        if node.lower_projection_child is not None:
            self.assign_top_down(
                node.lower_projection_child,
                embedded_data[labels == 0],
                embedded_data_indices[labels == 0],
                set_pruning_incidator,
                pruning_strategy,
                pruning_factor,
            )

    def predict_subclusters(self, node: Cluster_Node) -> np.array:
        """
        Predicts the 2 subclusters (child assignments) for the given node using properties of the dip test.
        The threshold is set between the middle coordinate of the modal triangle and the upper/lower modal interval.

        Parameters
        ----------
        node : Cluster_Node
            The node object for which we want to predict 2 subclusters.

        Returns
        -------
        labels : np.array
            A label (0/1) for each data point of the given node, where label 1 indicates the higher projection
            cluster.
        """

        if node.assignments.numel() == 1:
            warnings.warn(
                "Node just got 1 sample assigned. Data point will be assigned to higher projection node"
            )
            return np.array([1])

        projections = torch.matmul(
            node.assignments.detach().cpu().float(),
            node.projection_axis.detach().reshape(-1, 1),
        ).numpy()[
            :, 0
        ]  # remove second dimension after projection
        sorted_indices = projections.argsort()
        _, modal_interval, modal_triangle = dip_test(
            projections[sorted_indices], is_data_sorted=True, just_dip=False
        )
        index_lower, index_upper = modal_interval
        _, mid_point_triangle, _ = modal_triangle
        if (
            projections[sorted_indices[mid_point_triangle]]
            > projections[sorted_indices[index_upper]]
        ):
            threshold = (
                projections[sorted_indices[mid_point_triangle]]
                + projections[sorted_indices[index_upper]]
            ) / 2
        else:
            threshold = (
                projections[sorted_indices[mid_point_triangle]]
                + projections[sorted_indices[index_lower]]
            ) / 2
        labels = np.zeros(len(node.assignments))
        labels[projections >= threshold] = 1
        return labels

    def clear_pruning_values(self):
        for node in self.nodes:
            node.pruning_indicator = 0.0

    def prune_tree(self, pruning_threshold: float, metrics: dict = None) -> bool:
        """
        Prunes the tree by removing nodes with pruning indicators below the pruning threshold.

        Parameters
        ----------
        pruning_threshold : float
            The threshold value for pruning. Nodes with pruning indicators below this threshold will be removed.
        metrics : dict, optional
            Dictionary to store metrics, by default None.

        Returns
        -------
        bool
            Returns True if pruning occurred, otherwise False.
        """

        def prune_node(parent: Cluster_Node, child_attr: str):
            """
            Prunes a node from the tree by replacing it with its child or sibling node.

            Parameters
            ----------
            parent : Cluster_Node
                The parent node from which the child or sibling node will be pruned.
            child_attr : str
                The attribute name of the child node to be pruned.

            Returns
            -------
            None
            """
            child_node: Cluster_Node = getattr(parent, child_attr)
            sibling_attr = (
                "higher_projection_child"
                if child_attr == "lower_projection_child"
                else "lower_projection_child"
            )
            sibling_node: Cluster_Node = getattr(parent, sibling_attr)

            if sibling_node is None:
                raise ValueError(sibling_node)
            else:
                if parent == self.root:
                    self.root = sibling_node
                    self.root.parent = None
                else:
                    grandparent = parent.parent
                    if grandparent.lower_projection_child == parent:
                        grandparent.lower_projection_child = sibling_node
                    else:
                        grandparent.higher_projection_child = sibling_node
                    sibling_node.parent = grandparent
                sibling_node.split_id = parent.split_id
                sibling_node.assignments = parent.assignments
                sibling_node.assignment_indices = parent.assignment_indices
                sibling_node.pruning_indicator = parent.pruning_indicator
                sibling_node.split_level = parent.split_level

                child_node.prune()
                del child_node
                del parent

                logging.info(
                    f"Tree size after pruning: {self.number_nodes}, leaf nodes: {len(self.leaf_nodes)}"
                )

        def prune_recursive(node: Cluster_Node) -> bool:
            """
            Recursively prunes the tree starting from the given node.

            Parameters
            ----------
            node : Cluster_Node
                The node from which to start pruning.

            Returns
            -------
            bool
                Returns True if pruning occurred, otherwise False.
            """
            result = False
            if node.higher_projection_child is not None:
                result = prune_recursive(node.higher_projection_child)
            if node.lower_projection_child is not None:
                result = prune_recursive(node.lower_projection_child)

            if node.pruning_indicator < pruning_threshold:
                if node.parent is not None:
                    if node.parent.higher_projection_child == node:
                        prune_node(node.parent, "higher_projection_child")
                        result = True
                    else:
                        prune_node(node.parent, "lower_projection_child")
                        result = True
                else:
                    if (
                        self.root.higher_projection_child
                        and self.root.higher_projection_child.pruning_indicator
                        < pruning_threshold
                    ):
                        prune_node(self.root, "higher_projection_child")
                        result = True
                    elif (
                        self.root.lower_projection_child
                        and self.root.lower_projection_child.pruning_indicator
                        < pruning_threshold
                    ):
                        prune_node(self.root, "higher_projection_child")
                        result = True
            return result

        has_pruned = prune_recursive(self.root)
        if has_pruned and metrics is not None:
            metrics["nodes"] = len(self.nodes)
            metrics["leaf_nodes"] = len(self.leaf_nodes)
        return has_pruned

    def grow_tree(
        self,
        dataloader: torch.utils.data.DataLoader,
        autoencoder: torch.nn.Module,
        projection_axis_optimizer: torch.optim.Optimizer,
        max_leaf_nodes: int,
        unimodality_treshold: float,
        number_of_grow_steps: int = 1,
        tree_growth_min_cluster_size: int = 1000,
        use_pvalue: bool = True,
        metrics: dict = None,
    ) -> bool:
        """
        Grows the tree at the leaf node with the highest multimodality. Since the dip value depends
        on the number of samples, we use the p-value for the split criteria if use_pvalue is true. In this case
        we split the leaf node with the lowest p-value (lowest probability for unimodality). If the lowest p-value
        found is 0, we expand the multimodal leaf node (p-value < unimodal_threshold) with the maximal number of assigned
        samples. If use_pvalue is set to false, we introduce a criteria which includes the dip value and the number of samples
        of the node for the decision.
        The tree growing is stopped if all leaf nodes are unimodal (p-value > unimodal_threshold) or the maximal number of
        leaf nodes is reached.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The data loader for the dataset.
        autoencoder : torch.nn.Module
            The autoencoder model for embedding the data.
        projection_axis_optimizer : torch.optim.Optimizer
            The optimizer for the projection axes (adding the new axis to it.)
        max_leaf_nodes : int
            The maximal number of leaf nodes.
        unimodality_treshhold : float
            Specifies the minimal probability we demand so that we can call a node unimodal.
        number_of_grow_steps : int, optional
            Specifies how many tree grow steps should be applied, default is 1.
        tree_growth_min_cluster_size : int, optional
            Minimum cluster size to allow tree growth, default is 1000.
        use_pvalue : bool, optional
            Specifies which splitting criteria should be used, default is True.
        metrics : dict, optional
            Dictionary to store metrics, by default None.

        Returns
        -------
        bool
            Returns True if the algorithm should be stopped or False otherwise.
        """

        # do not grow further if threshold was already reached
        if len(self.leaf_nodes) >= max_leaf_nodes:
            return True

        X_embedded = torch.from_numpy(encode_batchwise(dataloader, autoencoder))
        multimodal_nodes = []
        for i in range(number_of_grow_steps):
            # do not grow further if treshhold was already reached
            if len(self.leaf_nodes) >= max_leaf_nodes:
                return True
            multimodal_nodes.clear()
            self.assign_to_tree(X_embedded)

            total_assignments = sum(
                [
                    len(node.assignments)
                    for node in self.leaf_nodes
                    if node.assignments is not None
                ]
            )

            for node in self.leaf_nodes:
                if (
                    node.assignments is None
                    or len(node.assignments) < 2
                    or (len(node.assignments) / 2) < tree_growth_min_cluster_size
                ):
                    continue
                node_data = node.assignments.numpy()
                (
                    axis,
                    number_assign_lower_projection_cluster,
                    number_assign_higher_projection_cluster,
                ) = self.get_inital_projection_axis(node_data)
                if (
                    number_assign_higher_projection_cluster
                    < tree_growth_min_cluster_size
                    or number_assign_lower_projection_cluster
                    < tree_growth_min_cluster_size
                ):
                    continue
                projections = np.matmul(node_data, axis)
                dip_value = dip_test(projections, just_dip=True, is_data_sorted=False)
                # p-value gives the probability for unimodality (smaller dip value, higher p-value)
                pvalue = dip_pval(dip_value, len(node.assignments))
                # the more samples, the smaller the dip value, consider this:

                multimodal_nodes.append(
                    (
                        node,
                        pvalue,
                        dip_value,
                        dip_value
                        + 0.5 * len(node.assignments) / (4 * total_assignments),
                        axis,
                        len(node.assignments),
                        number_assign_higher_projection_cluster,
                        number_assign_lower_projection_cluster,
                    )
                )
                # if use_pvalue:
                #     current_value = pvalue
                #     better = False
                #     if (
                #         best_value > unimodality_treshhold
                #         and current_value < best_value
                #     ):
                #         # if best_value is above unimodality_treshhold, it is sufficient to be smaller than
                #         # best_value
                #         max_assignments = len(node_data)
                #         better = True
                #     elif (
                #         current_value <= unimodality_treshhold
                #         and len(node_data) > max_assignments
                #     ):
                #         # if best_value is also already smaller than unimodality_treshhold, the current
                #         # node must have more assignments in order to be accepted
                #         max_assignments = len(node_data)
                #         better = True
                # else:
                #     current_value = dip_value + 0.5 * len(node.assignments) / (
                #         4 * total_assignments
                #     )
                #     better = current_value > best_value

                logging.info(
                    f"checking node {node.id} with #assignments: {len(node.assignments)} - pvalue: {pvalue} - dip value: {dip_value}"
                )

                # if pvalue > unimodality_treshhold:
                #     continue
                # if better:
                #     best_value = current_value
                #     best_node_axis = axis
                #     best_node_to_split = node
                #     best_node_number_assign_lower_projection_cluster = (
                #         number_assign_lower_projection_cluster
                #     )
                #     best_node_number_assign_higher_projection_cluster = (
                #         number_assign_higher_projection_cluster
                #     )
            best_node_to_split = None
            if use_pvalue:
                nodes_to_split = sorted(
                    [x for x in multimodal_nodes if x[1] <= unimodality_treshold],
                    key=lambda x: (x[1], -x[5]),
                )
                if len(nodes_to_split) > 0:
                    best_node_to_split = nodes_to_split[0]
            else:
                nodes_to_split = sorted(
                    multimodal_nodes, key=lambda x: (x[1], x[3]), reverse=True
                )
                if len(nodes_to_split) > 0:
                    best_node_to_split = nodes_to_split[0]

            if (
                best_node_to_split == None
            ):  # unimodality threshold reached - no multimodality found
                return True

            # split best node
            logging.info(
                f"split node {best_node_to_split[0].id} with #assignments: {len(best_node_to_split[0].assignments)} "
            )
            best_node_to_split[0].expand_tree(
                best_node_to_split[4],
                projection_axis_optimizer,
                best_node_to_split[6],
                best_node_to_split[7],
                max([leaf.id for leaf in self.leaf_nodes]),
                max([node.split_id for node in self.nodes]),
            )
            if metrics is not None:
                metrics["nodes"] = len(self.nodes)
                metrics["leaf_nodes"] = len(self.leaf_nodes)
        return False

    def improve_space(
        self,
        embedded_data: torch.Tensor,
        embedded_augmented_data: Union[torch.Tensor | None],
        projection_axis_optimizer: torch.optim.Optimizer,
        unimodal_loss_application,
        unimoal_loss_node_criteria_method,
        unimodal_loss_weight_function,
        unimodal_loss_weight_direction,
        unimodal_loss_weight,
        loss_weight_function_normalization,
        mulitmodal_loss_application,
        mulitmodal_loss_node_criteria_method,
        mulitmodal_loss_weight_function,
        mulitmodal_loss_weight_direction,
        multimodal_loss_weight,
        projection_axis_learning,
        pruning_strategy: str,
        pruning_factor: float,
    ):
        """
        Calculates the cluster loss of the given data samples based on the current the tree structure.
        Based on the mode <projection_axis_learning>, it also adapts the projection axis.

        Parameters
        ----------
        embedded_data : torch.Tensor
            The embedded data samples.
        embedded_augmented_data : Union[torch.Tensor, None]
            The embedded augmented data samples.
        projection_axis_optimizer : torch.optim.Optimizer
            The optimizer for the projection axes.
        unimodal_loss_application : str
            Specifies where the unimodal loss should be applied.
        unimoal_loss_node_criteria_method : str
            Specifies the method to use for node criteria.
        unimodal_loss_weight_function : str
            Specifies the weight function for unimodal loss.
        unimodal_loss_weight_direction : str
            Specifies the weight direction for unimodal loss.
        unimodal_loss_weight : float
            The weight for unimodal loss.
        loss_weight_function_normalization : float
            The normalization factor for the loss weight function.
        mulitmodal_loss_application : str
            Specifies where the multimodal loss should be applied.
        mulitmodal_loss_node_criteria_method : str
            Specifies the method to use for node criteria for multimodal loss.
        mulitmodal_loss_weight_function : str
            Specifies the weight function for multimodal loss.
        mulitmodal_loss_weight_direction : str
            Specifies the weight direction for multimodal loss.
        multimodal_loss_weight : float
            The weight for multimodal loss.
        projection_axis_learning : str
            Specifies the learning strategy for projection axes.
        pruning_strategy : str
            The strategy to use for pruning.
        pruning_factor : float
            The factor to use for pruning.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The unimodal and multimodal losses.
        """
        self.assign_to_tree(
            embedded_data,
            pruning_strategy,
            pruning_factor,
            set_pruning_incidator=True,
        )

        unimodal_losses = []
        multimodal_losses = []
        self._improve_space_recursive(
            self.root,
            projection_axis_optimizer,
            unimodal_losses,
            multimodal_losses,
            embedded_augmented_data,
            unimodal_loss_application,
            unimoal_loss_node_criteria_method,
            unimodal_loss_weight_function,
            unimodal_loss_weight_direction,
            unimodal_loss_weight,
            loss_weight_function_normalization,
            mulitmodal_loss_application,
            mulitmodal_loss_node_criteria_method,
            mulitmodal_loss_weight_function,
            mulitmodal_loss_weight_direction,
            multimodal_loss_weight,
            projection_axis_learning,
        )
        if len(unimodal_losses) > 0:
            unimodal_loss = torch.mean(torch.stack(unimodal_losses, dim=0))
        else:
            torch.tensor(0.0, dtype=torch.float)
        if len(multimodal_losses) > 0:
            multimodal_loss = torch.mean(torch.stack(multimodal_losses, dim=0))
        else:
            torch.tensor(0.0, dtype=torch.float)
        return unimodal_loss, multimodal_loss

    def _improve_space_recursive(
        self,
        node: Cluster_Node,
        projection_axis_optimizer: torch.optim.Optimizer,
        unimodal_loss: List[torch.Tensor],
        multimodal_loss: List[torch.Tensor],
        embedded_augmented_data: Union[torch.Tensor | None],
        unimodal_loss_application,
        unimoal_loss_node_criteria_method,
        unimodal_loss_weight_function,
        unimodal_loss_weight_direction,
        unimodal_loss_weight,
        loss_weight_function_normalization,
        mulitmodal_loss_application,
        mulitmodal_loss_node_criteria_method,
        mulitmodal_loss_weight_function,
        mulitmodal_loss_weight_direction,
        multimodal_loss_weight,
        projection_axis_learning,
    ):
        """
        Helper function for going recursively through the tree and calculating the cluster loss for each node.
        The losses per node are summed up and returned. Based on the mode <projection_axis_learning>,
        the axis of the node is adapted before calculating the loss.

        Parameters
        ----------
        node : Cluster_Node
            The node whose space should be improved.
        projection_axis_optimizer : torch.optim.Optimizer
            The optimizer for the projection axes.
        unimodal_loss : torch.Tensor
            The unimodal loss tensor.
        multimodal_loss : torch.Tensor
            The multimodal loss tensor.
        embedded_augmented_data : Union[torch.Tensor, None]
            The embedded augmented data samples.
        unimodal_loss_application : str
            Specifies where the unimodal loss should be applied.
        unimoal_loss_node_criteria_method : str
            Specifies the method to use for node criteria.
        unimodal_loss_weight_function : str
            Specifies the weight function for unimodal loss.
        unimodal_loss_weight_direction : str
            Specifies the weight direction for unimodal loss.
        unimodal_loss_weight : float
            The weight for unimodal loss.
        loss_weight_function_normalization : float
            The normalization factor for the loss weight function.
        mulitmodal_loss_application : str
            Specifies where the multimodal loss should be applied.
        mulitmodal_loss_node_criteria_method : str
            Specifies the method to use for node criteria for multimodal loss.
        mulitmodal_loss_weight_function : str
            Specifies the weight function for multimodal loss.
        mulitmodal_loss_weight_direction : str
            Specifies the weight direction for multimodal loss.
        multimodal_loss_weight : float
            The weight for multimodal loss.
        projection_axis_learning : str
            Specifies the learning strategy for projection axes.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The unimodal and multimodal losses.
        """
        if projection_axis_learning is not None:
            if (
                (
                    projection_axis_learning == "only_leaf_nodes"
                    and node.higher_projection_child.is_leaf_node()
                    and node.lower_projection_child.is_leaf_node()
                )
                or (
                    projection_axis_learning == "partial_leaf_nodes"
                    and (
                        node.higher_projection_child.is_leaf_node()
                        or node.lower_projection_child.is_leaf_node()
                    )
                )
                or projection_axis_learning == "all"
            ):
                self._adjust_axis(node, projection_axis_optimizer)

        # axis need to be cloned, otherwise pytorch throws an error
        # that the tensor will be inplaced modified while it is needed
        # for gradient computation
        axis = node.projection_axis.detach().clone()

        higher_projection_child_improvement = (
            node.higher_projection_child.assignments is not None
            and node.higher_projection_child.assignments.numel() > 1
        )
        lower_projection_child_improvement = (
            node.lower_projection_child.assignments is not None
            and node.lower_projection_child.assignments.numel() > 1
        )

        if higher_projection_child_improvement and lower_projection_child_improvement:
            calc_uni_loss_weight = self._calc_loss_weight(
                node,
                unimodal_loss_application,
                unimoal_loss_node_criteria_method,
                unimodal_loss_weight_function,
                unimodal_loss_weight_direction,
                unimodal_loss_weight,
                loss_weight_function_normalization,
            )
            calc_multi_loss_weight = self._calc_loss_weight(
                node,
                mulitmodal_loss_application,
                mulitmodal_loss_node_criteria_method,
                mulitmodal_loss_weight_function,
                mulitmodal_loss_weight_direction,
                multimodal_loss_weight,
                loss_weight_function_normalization,
                multimodal=True,
            )[0]

            if embedded_augmented_data is not None:
                higher_projection_cluster = torch.cat(
                    (
                        node.higher_projection_child.assignments,
                        embedded_augmented_data[
                            node.higher_projection_child.assignment_indices
                        ],
                    ),
                    dim=0,
                )
                lower_projection_cluster = torch.cat(
                    (
                        node.lower_projection_child.assignments,
                        embedded_augmented_data[
                            node.lower_projection_child.assignment_indices
                        ],
                    ),
                    dim=0,
                )
            else:
                higher_projection_cluster = node.higher_projection_child.assignments
                lower_projection_cluster = node.lower_projection_child.assignments

            if calc_uni_loss_weight[0] != 0.0:
                unimodal_loss.append(
                    calc_uni_loss_weight[0]
                    * _Dip_Gradient.apply(higher_projection_cluster, axis)
                )
            if calc_uni_loss_weight[1] != 0.0:
                unimodal_loss.append(
                    calc_uni_loss_weight[1]
                    * _Dip_Gradient.apply(lower_projection_cluster, axis)
                )

            if calc_multi_loss_weight != 0.0:
                multimodal_loss.append(
                    calc_multi_loss_weight
                    * _Dip_Gradient.apply(
                        torch.cat(
                            (higher_projection_cluster, lower_projection_cluster), dim=0
                        ),
                        axis,
                    )
                )

        if (
            higher_projection_child_improvement
            and not node.higher_projection_child.is_leaf_node()
        ):
            self._improve_space_recursive(
                node.higher_projection_child,
                projection_axis_optimizer,
                unimodal_loss,
                multimodal_loss,
                embedded_augmented_data,
                unimodal_loss_application,
                unimoal_loss_node_criteria_method,
                unimodal_loss_weight_function,
                unimodal_loss_weight_direction,
                unimodal_loss_weight,
                loss_weight_function_normalization,
                mulitmodal_loss_application,
                mulitmodal_loss_node_criteria_method,
                mulitmodal_loss_weight_function,
                mulitmodal_loss_weight_direction,
                multimodal_loss_weight,
                projection_axis_learning,
            )
        if (
            lower_projection_child_improvement
            and not node.lower_projection_child.is_leaf_node()
        ):
            self._improve_space_recursive(
                node.lower_projection_child,
                projection_axis_optimizer,
                unimodal_loss,
                multimodal_loss,
                embedded_augmented_data,
                unimodal_loss_application,
                unimoal_loss_node_criteria_method,
                unimodal_loss_weight_function,
                unimodal_loss_weight_direction,
                unimodal_loss_weight,
                loss_weight_function_normalization,
                mulitmodal_loss_application,
                mulitmodal_loss_node_criteria_method,
                mulitmodal_loss_weight_function,
                mulitmodal_loss_weight_direction,
                multimodal_loss_weight,
                projection_axis_learning,
            )

    def _adjust_axis(
        self, node: Cluster_Node, projection_axis_optimizer: torch.optim.Optimizer
    ):
        """
        Adjusts the projection axis of the given node.

        Parameters
        ----------
        node : Cluster_Node
            The node whose projection axis should be adjusted.
        projection_axis_optimizer : torch.optim.Optimizer
            The optimizer handling the projection axes.
        """
        # data gradients should not be stored
        projection_axis_optimizer.zero_grad()
        data = (
            node.assignments.detach().cpu()
        )  # cpu() already creates a copy, no cloning necessary
        loss = -_Dip_Gradient.apply(data, node.projection_axis)
        loss.backward()
        projection_axis_optimizer.step()

    def _calc_loss_weight(
        self,
        node: Cluster_Node,
        loss_application: Union[str | None],
        loss_node_criteria_method: str,
        loss_weight_scale_function: str,
        loss_weight_direction: str,
        loss_weight: float,
        weight_normalization: float,
        multimodal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the loss weight for a node based on the given criteria.

        Parameters
        ----------
        node : Cluster_Node
            The node for which to calculate the loss weight.
        loss_application : Union[str, None]
            Specifies where the loss should be applied.
        loss_node_criteria_method : str
            The method to use for node criteria.
        loss_weight_scale_function : str
            The weight function for loss.
        loss_weight_direction : str
            The weight direction for loss.
        loss_weight : float
            The weight for the loss.
        weight_normalization : float
            The normalization factor for the weight.
        multimodal : bool, optional
            Whether to calculate weight for multimodal loss, by default False.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The calculated weights for unimodal and multimodal loss.
        """
        if loss_application is None:
            return (
                torch.tensor(0.0, dtype=torch.float),
                torch.tensor(0.0, dtype=torch.float),
            )

        node_criteria = self._get_node_criteria(
            node, loss_node_criteria_method, loss_weight_direction
        )

        if loss_weight_scale_function == "linear":
            weights = self._linear(
                loss_weight_direction, node_criteria, weight_normalization, loss_weight
            )
        elif loss_weight_scale_function == "exponential":
            weights = self._exponential(
                loss_weight_direction, node_criteria, weight_normalization, loss_weight
            )
        elif loss_weight_scale_function == "log":
            weights = self._log(
                loss_weight_direction, node_criteria, weight_normalization, loss_weight
            )
        elif loss_weight_scale_function == "sqrt":
            weights = self._sqrt(
                loss_weight_direction, node_criteria, weight_normalization, loss_weight
            )
        elif loss_weight_scale_function is None:
            weights = (loss_weight, loss_weight)
        else:
            raise ValueError(
                f"method for calculating the unimodal loss weight not supported. Make sure it is a string indicating the method (e.g. linear) or directly pass the weight as a float"
            )

        if loss_application == "leaf_nodes" and not multimodal:
            if not node.higher_projection_child.is_leaf_node():
                weights = (torch.tensor(0.0, dtype=torch.float), weights[1])
            if not node.lower_projection_child.is_leaf_node():
                weights = (weights[0], torch.tensor(0.0, dtype=torch.float))
        return weights

    def _linear(self, direction, node_criteria, normalization, unimodal_loss_weight):
        if normalization == -1:
            normalization = 1
        weight = (unimodal_loss_weight / normalization) * node_criteria
        return (weight, weight)

    def _exponential(
        self, direction, node_criteria, normalization, unimodal_loss_weight
    ):
        if normalization == -1:
            normalization = 1
        weight = (unimodal_loss_weight / normalization) * np.exp2(node_criteria)
        return (weight, weight)

    def _log(self, direction, node_criteria, normalization, unimodal_loss_weight):
        if normalization == -1:
            normalization = 1
        weight = np.log1p((unimodal_loss_weight / normalization) * node_criteria)
        return (weight, weight)

    def _sqrt(self, direction, node_criteria, normalization, unimodal_loss_weight):
        if normalization == -1:
            normalization = 1
        weight = np.sqrt((unimodal_loss_weight / normalization) * node_criteria)
        return (weight, weight)

    def _get_node_criteria(
        self, node: Cluster_Node, node_criteria_method: str, direction: str
    ):
        if node_criteria_method == "equal":
            return 1
        if node_criteria_method == "time_of_split":
            node_ids = sorted([node.split_id for node in self.nodes])
            index = node_ids.index(node.split_id) + 1
            if direction == "ascending":
                return index / len(node_ids)
            elif direction == "descending":
                return (len(node_ids) - index + 1) / len(node_ids)
            else:
                raise ValueError(f"loss direction {direction} not supported")
        if node_criteria_method == "tree_depth":
            max_level = max([node.split_level for node in self.nodes]) + 1
            if direction == "ascending":
                return (node.split_level + 1) / max_level
            elif direction == "descending":
                return (max_level - node.split_level + 1) / max_level
            else:
                raise ValueError(f"loss direction {direction} not supported")


def transform_cluster_tree_to_pred_tree(tree: Cluster_Tree) -> PredictionClusterTree:
    """
    Transforms a Cluster_Tree to a PredictionClusterTree.

    Parameters
    ----------
    tree : Cluster_Tree
        The cluster tree to transform.

    Returns
    -------
    PredictionClusterTree
        The transformed prediction cluster tree.
    """

    def transform_nodes(node: Cluster_Node):
        pred_node = PredictionClusterNode(node.id, node.split_id, None)
        if node.is_leaf_node():
            return pred_node
        pred_node.left_child = transform_nodes(node.higher_projection_child)
        pred_node.left_child.parent = pred_node
        pred_node.right_child = transform_nodes(node.lower_projection_child)
        pred_node.right_child.parent = pred_node
        return pred_node

    return PredictionClusterTree(transform_nodes(tree.root))


class _DipECT_Module(torch.nn.Module):
    """
    The _DipECT_Module. Contains most of the algorithm specific procedures like the loss and tree-grow functions.

    Parameters
    ----------
    init_centers : np.ndarray
        The initial cluster centers.
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False).

    Attributes
    ----------
    cluster_tree: Cluster_Node
        The cluster tree.
    augmentation_invariance : bool
        Is augmentation invariance used.
    """

    def __init__(
        self,
        trainloader: torch.utils.data.DataLoader,
        autoencoder: torch.nn.Module,
        projection_axis_optimizer: torch.optim.Optimizer,
        device: torch.device,
        random_state: np.random.RandomState,
        augmentation_invariance: bool = False,
    ):
        super().__init__()

        self.augmentation_invariance = augmentation_invariance
        self.device = device
        self.random_state = random_state

        # Create initial cluster tree
        self.cluster_tree = Cluster_Tree(
            trainloader,
            autoencoder,
            projection_axis_optimizer,
            device,
            self.random_state,
        )

    def fit(
        self,
        autoencoder: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        labels: np.ndarray,
        max_epochs: int,
        pruning_threshold: float,
        grow_interval: float,
        use_pvalue: bool,
        max_leaf_nodes: int,
        reconstruction_loss_weight: float,
        unimodality_treshhold: float,
        number_of_grow_steps: int,
        early_stopping: bool,
        refinement_epochs: int,
        optimizer: torch.optim.Optimizer,
        projection_axis_optimizer: torch.optim.Optimizer,
        rec_loss_fn: torch.nn.modules.loss._Loss,
        device: Union[torch.device | str],
        logging_active: bool,
        unimodal_loss_application,
        unimoal_loss_node_criteria_method,
        unimodal_loss_weight_function,
        unimodal_loss_weight_direction,
        unimodal_loss_weight,
        loss_weight_function_normalization,
        mulitmodal_loss_application,
        mulitmodal_loss_node_criteria_method,
        mulitmodal_loss_weight_function,
        mulitmodal_loss_weight_direction,
        multimodal_loss_weight,
        projection_axis_learning,
        pruning_strategy,
        pruning_factor,
        tree_growth_min_cluster_size,
        evaluate_after_n_epochs: int = 0,
    ) -> "_DipECT_Module":
        """
        Trains the _DipECT_Module in place.

        Parameters
        ----------
        autoencoder : torch.nn.Module
            The autoencoder used for training.
        trainloader : torch.utils.data.DataLoader
            DataLoader for training data.
        testloader : torch.utils.data.DataLoader
            DataLoader for testing data.
        labels : np.ndarray
            True labels of the data.
        max_epochs : int
            Maximum number of epochs for training.
        pruning_threshold : float
            Threshold for pruning the cluster tree.
        grow_interval : float
            Interval for growing the cluster tree.
        use_pvalue : bool
            Whether to use p-value for split criteria.
        max_leaf_nodes : int
            Maximum number of leaf nodes in the cluster tree.
        reconstruction_loss_weight : float
            Weight for the reconstruction loss.
        unimodality_treshhold : float
            Threshold for unimodality.
        number_of_grow_steps : int
            Number of grow steps.
        early_stopping : bool
            Whether to use early stopping.
        refinement_epochs : int
            Number of refinement epochs.
        optimizer : torch.optim.Optimizer
            Optimizer for training.
        projection_axis_optimizer : torch.optim.Optimizer
            Optimizer for projection axes.
        rec_loss_fn : torch.nn.modules.loss._Loss
            Loss function for reconstruction.
        device : Union[torch.device, str]
            Device for training (e.g., "cuda" or "cpu").
        logging_active : bool
            Whether to log metrics.
        unimodal_loss_application : str
            Specifies where the unimodal loss should be applied.
        unimoal_loss_node_criteria_method : str
            Specifies the method to use for node criteria.
        unimodal_loss_weight_function : str
            Specifies the weight function for unimodal loss.
        unimodal_loss_weight_direction : str
            Specifies the weight direction for unimodal loss.
        unimodal_loss_weight : float
            The weight for unimodal loss.
        loss_weight_function_normalization : float
            The normalization factor for the loss weight function.
        mulitmodal_loss_application : str
            Specifies where the multimodal loss should be applied.
        mulitmodal_loss_node_criteria_method : str
            Specifies the method to use for node criteria for multimodal loss.
        mulitmodal_loss_weight_function : str
            Specifies the weight function for multimodal loss.
        mulitmodal_loss_weight_direction : str
            Specifies the weight direction for multimodal loss.
        multimodal_loss_weight : float
            The weight for multimodal loss.
        projection_axis_learning : str
            Specifies the learning strategy for projection axes.
        pruning_strategy : str
            The strategy to use for pruning.
        pruning_factor : float
            The factor to use for pruning.
        tree_growth_min_cluster_size : int
            Minimum cluster size to allow tree growth.
        evaluate_after_n_epochs : int, optional
            Interval to evaluate the model, by default 0.

        Returns
        -------
        _DipECT_Module
            The instance of the _DipECT_Module.
        """

        self.device = device
        self = self.to(device)
        autoencoder = autoencoder.to(device)

        mov_rec_loss = 0.0
        mov_rec_loss_aug = 0.0
        mov_cluster_loss = 0.0
        mov_unimodal_loss = 0.0
        mov_multimodal_loss = 0.0
        mov_loss = 0.0

        growing_treshhold_reached = False
        refinement_counter = 0
        iterations_until_grow = math.ceil(len(trainloader) * grow_interval)
        iteration = 0
        # if grow_interval is 0, generate the whole tree before training
        metrics = {
            "acc": 0.0,
            "nmi": 0.0,
            "ari": 0.0,
            "dp": 0.0,
            "lp": 0.0,
        }
        if iterations_until_grow == 0:
            while not growing_treshhold_reached:
                growing_treshhold_reached = self.cluster_tree.grow_tree(
                    testloader,
                    autoencoder,
                    projection_axis_optimizer,
                    max_leaf_nodes,
                    unimodality_treshhold,
                    tree_growth_min_cluster_size=tree_growth_min_cluster_size,
                    use_pvalue=use_pvalue,
                    metrics=metrics,
                )
            iterations_until_grow = math.ceil(len(trainloader) * 2.0)
            growing_treshhold_reached = (
                False  # prevent from immediately ending training
            )
        classes = np.unique(labels).size
        for epoch in range(max_epochs):
            # evaluation
            if epoch > 0:
                if epoch % evaluate_after_n_epochs == 0:
                    pred_tree = self.predict(testloader, autoencoder)
                    metrics.update(
                        {
                            "acc": pred_tree.flat_accuracy(labels, classes),
                            "nmi": pred_tree.flat_nmi(labels, classes),
                            "ari": pred_tree.flat_ari(labels, classes),
                            "dp": pred_tree.dendrogram_purity(labels),
                            "lp": pred_tree.leaf_purity(labels)[0],
                        }
                    )
                    if logging_active:
                        logging.info(metrics)

            if pruning_strategy == "epoch_assessment" and epoch > 0:
                self.cluster_tree.prune_tree(pruning_threshold, metrics)
                self.cluster_tree.clear_pruning_values()

            with tqdm(
                trainloader, unit="batch", desc=f"Epoch {epoch+1}/{max_epochs}"
            ) as tepoch:
                for batch in tepoch:
                    if pruning_strategy == "moving_average":
                        self.cluster_tree.prune_tree(pruning_threshold, metrics)

                    if (
                        iteration % iterations_until_grow == 0 and iteration > 0
                    ) or self.cluster_tree.number_nodes < 3:
                        growing_treshhold_reached = self.cluster_tree.grow_tree(
                            testloader,
                            autoencoder,
                            projection_axis_optimizer,
                            max_leaf_nodes,
                            unimodality_treshhold,
                            number_of_grow_steps,
                            tree_growth_min_cluster_size,
                            use_pvalue,
                            metrics,
                        )
                        if growing_treshhold_reached and early_stopping:
                            logging.info(
                                "Stopped algorithm earlier since unimodality threshold is reached. Eventually refinement epochs starting..."
                            )
                            break

                    optimizer.zero_grad()

                    if self.augmentation_invariance:
                        idxs, M, M_aug = batch
                    else:
                        idxs, M = batch

                    # calculate autoencoder loss
                    rec_loss, embedded, reconstructed = autoencoder.loss(
                        [idxs, M], rec_loss_fn, self.device
                    )
                    assert any(torch.any(embedded.isnan(), dim=1)) == False

                    if self.augmentation_invariance:
                        rec_loss_aug, embedded_aug, reconstructed_aug = (
                            autoencoder.loss([idxs, M_aug], rec_loss_fn, self.device)
                        )
                        assert any(torch.any(embedded_aug.isnan(), dim=1)) == False

                    # calculate cluster loss
                    unimodal_loss, multimodal_loss = self.cluster_tree.improve_space(
                        embedded.cpu(),
                        embedded_aug.cpu() if self.augmentation_invariance else None,
                        projection_axis_optimizer,
                        unimodal_loss_application,
                        unimoal_loss_node_criteria_method,
                        unimodal_loss_weight_function,
                        unimodal_loss_weight_direction,
                        unimodal_loss_weight,
                        loss_weight_function_normalization,
                        mulitmodal_loss_application,
                        mulitmodal_loss_node_criteria_method,
                        mulitmodal_loss_weight_function,
                        mulitmodal_loss_weight_direction,
                        multimodal_loss_weight,
                        projection_axis_learning,
                        pruning_strategy,
                        pruning_factor,
                    )
                    # all_split_nodes = self.cluster_tree.number_nodes - len(
                    #     self.cluster_tree.leaf_nodes
                    # )
                    # only_split_nodes = len(
                    #     [
                    #         node
                    #         for node in self.cluster_tree.nodes
                    #         if not node.is_leaf_node()
                    #         and (
                    #             node.higher_projection_child.is_leaf_node()
                    #             or node.lower_projection_child.is_leaf_node()
                    #         )
                    #     ]
                    # )

                    # already scaled per node in improve space step
                    cluster_loss = unimodal_loss - multimodal_loss
                    # cluster_loss = (unimodal_loss - multimodal_loss) / all

                    if reconstruction_loss_weight is None:
                        reconstruction_loss_weight = 1 / (
                            1 * rec_loss.detach()
                        )  # /(4* rec_loss.detach()) # * 0.01

                    if self.augmentation_invariance:
                        loss = cluster_loss + reconstruction_loss_weight * 0.5 * (
                            rec_loss + rec_loss_aug
                        )
                        metrics["rec_loss_aug"] = rec_loss_aug.item()
                        mov_rec_loss_aug += metrics["rec_loss_aug"]
                    else:
                        loss = cluster_loss + rec_loss * reconstruction_loss_weight

                    loss.backward()
                    optimizer.step()
                    self.cluster_tree.clear_node_assignments()
                    metrics["multimodal_loss"] = multimodal_loss.item()
                    metrics["unimodal_loss"] = unimodal_loss.item()
                    metrics["cluster_loss"] = cluster_loss.item()
                    metrics["rec_loss"] = rec_loss.item()
                    metrics["loss"] = loss.item()
                    metrics["nodes"] = self.cluster_tree.number_nodes
                    metrics["leaf_nodes"] = len(self.cluster_tree.leaf_nodes)
                    metrics["combined_metrics"] = (
                        metrics["acc"] + metrics["dp"] + metrics["lp"]
                    )
                    iteration += 1
                    train.report(metrics)
                    mov_loss += metrics["loss"]
                    mov_rec_loss += metrics["rec_loss"]
                    mov_cluster_loss += metrics["cluster_loss"]
                    mov_unimodal_loss += metrics["unimodal_loss"]
                    mov_multimodal_loss += metrics["multimodal_loss"]

            if growing_treshhold_reached and early_stopping:
                refinement_counter += 1
                if refinement_counter > refinement_epochs:
                    break

            if logging_active:
                log_epoch_nr = epoch + 1  # to avoid division by zero
                logging.info(
                    f"epoch: {log_epoch_nr} - nodes: {len(self.cluster_tree.nodes)} - leaf nodes: {len(self.cluster_tree.leaf_nodes)} - moving averages: mov_rec_loss: {mov_rec_loss/log_epoch_nr} "
                    + f"mov_multimodal_loss: {mov_multimodal_loss/log_epoch_nr} mov_unimodal_loss: {mov_unimodal_loss/log_epoch_nr} "
                    + f"mov_cluster_loss: {mov_cluster_loss/log_epoch_nr} {f'rec_loss_aug: {mov_rec_loss_aug/log_epoch_nr}' if self.augmentation_invariance else ''} total_loss: {mov_loss/log_epoch_nr}",
                )

        return self

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        autoencoder: torch.nn.Module,
    ) -> PredictionClusterTree:
        """
        Batchwise prediction of the given samples in the dataloader for a
        given number of classes.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            DataLoader for the samples to be predicted.
        autoencoder: torch.nn.Module
            Autoencoder model for calculating the embeddings.

        Returns
        -------
        pred_tree : PredictionClusterTree
            The prediction cluster tree with assigned samples.
        """
        # get prediction tree
        pred_tree = transform_cluster_tree_to_pred_tree(self.cluster_tree)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predict"):
                # calculate embeddings of the samples which should be predicted
                batch_data = batch[1].to(self.device)
                indices = batch[0].cpu()
                embeddings = autoencoder.encode(batch_data)
                # assign the embeddings to the cluster tree
                self.cluster_tree.assign_to_tree(embeddings.cpu())
                # use assignment indices for prediction tree
                for node in self.cluster_tree.leaf_nodes:
                    pred_tree[node.id].assign_batch(indices, node.assignment_indices)
                self.cluster_tree.clear_node_assignments()

        return pred_tree


def _dipect(
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int,
    pretrain_optimizer_params: dict,
    clustering_optimizer_params: dict,
    projection_axis_optimizer_params: dict,
    pretrain_epochs: int,
    max_epochs: int,
    pruning_threshold: float,
    grow_interval: float,
    use_pvalue: bool,
    optimizer_class: torch.optim.Optimizer,
    rec_loss_fn: torch.nn.modules.loss._Loss,
    autoencoder: _AbstractAutoencoder,
    embedding_size: int,
    max_leaf_nodes: int,
    reconstruction_loss_weight: float,
    unimodality_treshhold: float,
    number_of_grow_steps: int,
    early_stopping: bool,
    refinement_epochs: int,
    custom_dataloaders: tuple,
    augmentation_invariance: bool,
    random_state: np.random.RandomState,
    logging_active: bool,
    autoencoder_save_param_path: str,
    unimodal_loss_application,
    unimoal_loss_node_criteria_method,
    unimodal_loss_weight_function,
    unimodal_loss_weight_direction,
    unimodal_loss_weight,
    loss_weight_function_normalization,
    mulitmodal_loss_application,
    mulitmodal_loss_node_criteria_method,
    mulitmodal_loss_weight_function,
    mulitmodal_loss_weight_direction,
    multimodal_loss_weight,
    projection_axis_learning,
    pruning_strategy,
    pruning_factor,
    tree_growth_min_cluster_size,
    evaluate_every_n_epochs,
):
    """
    Start the actual DipECT clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        The given data set. Can be a np.ndarray or a torch.Tensor.
    Y : np.ndarray
        The given labels for the data set.
    batch_size : int
        Size of the data batches.
    pretrain_optimizer_params : dict
        Parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate.
    clustering_optimizer_params : dict
        Parameters of the optimizer for the actual clustering procedure, includes the learning rate.
    projection_axis_optimizer_params : dict
        Parameters of the optimizer for the projection axes.
    pretrain_epochs : int
        Number of epochs for the pretraining of the autoencoder.
    max_epochs : int
        Number of epochs for the actual clustering procedure.
    pruning_threshold : float
        The threshold for pruning the tree.
    grow_interval : int
        Interval for growing the tree.
    use_pvalue : bool
        Whether to use p-value for split criteria.
    optimizer_class : torch.optim.Optimizer
        The optimizer class.
    rec_loss_fn : torch.nn.modules.loss._Loss
        Loss function for the reconstruction.
    autoencoder : torch.nn.Module
        The input autoencoder.
    embedding_size : int
        Size of the embedding within the autoencoder.
    max_leaf_nodes : int
        Maximum number of leaf nodes in the cluster tree.
    reconstruction_loss_weight : float
        Weight for the reconstruction loss.
    unimodality_treshhold : float
        Threshold for unimodality.
    number_of_grow_steps : int
        Number of grow steps.
    early_stopping : bool
        Whether to use early stopping.
    refinement_epochs : int
        Number of refinement epochs.
    custom_dataloaders : tuple
        Tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn cluster assignments that are invariant to the augmentation transformations.
    random_state : np.random.RandomState
        Use a fixed random state to get a repeatable solution.
    logging_active : bool
        Whether to log metrics.
    autoencoder_save_param_path : str
        Path to save the autoencoder parameters.
    unimodal_loss_application : str
        Specifies where the unimodal loss should be applied.
    unimoal_loss_node_criteria_method : str
        Specifies the method to use for node criteria.
    unimodal_loss_weight_function : str
        Specifies the weight function for unimodal loss.
    unimodal_loss_weight_direction : str
        Specifies the weight direction for unimodal loss.
    unimodal_loss_weight : float
        The weight for unimodal loss.
    loss_weight_function_normalization : float
        The normalization factor for the loss weight function.
    mulitmodal_loss_application : str
        Specifies where the multimodal loss should be applied.
    mulitmodal_loss_node_criteria_method : str
        Specifies the method to use for node criteria for multimodal loss.
    mulitmodal_loss_weight_function : str
        Specifies the weight function for multimodal loss.
    mulitmodal_loss_weight_direction : str
        Specifies the weight direction for multimodal loss.
    multimodal_loss_weight : float
        The weight for multimodal loss.
    projection_axis_learning : str
        Specifies the learning strategy for projection axes.
    pruning_strategy : str
        The strategy to use for pruning.
    pruning_factor : float
        The factor to use for pruning.
    tree_growth_min_cluster_size : int
        Minimum cluster size to allow tree growth.
    evaluate_every_n_epochs : int
        Interval to evaluate the model.

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by a final KMeans execution,
        The cluster centers as identified by a final KMeans execution,
        The labels as identified by DeepECT after the training terminated,
        The cluster centers as identified by DeepECT after the training terminated,
        The final autoencoder.
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    if os.path.exists(autoencoder_save_param_path):
        autoencoder.load_parameters(autoencoder_save_param_path)
    save_ae_state_dict = not hasattr(autoencoder, "fitted") or not autoencoder.fitted
    set_torch_seed(random_state)

    device = detect_device()
    # sample random mini-batches from the data -> shuffle = True
    if custom_dataloaders is None:
        trainloader = get_dataloader(X, batch_size, True, False)
        testloader = get_dataloader(X, batch_size, False, False)
    else:
        trainloader, testloader = custom_dataloaders
    # Get initial AE
    autoencoder = get_trained_network(
        trainloader=trainloader,
        optimizer_params=pretrain_optimizer_params,
        n_epochs=pretrain_epochs,
        device=device,
        optimizer_class=optimizer_class,
        loss_fn=rec_loss_fn,
        embedding_size=embedding_size,
        neural_network=autoencoder,
    )

    logging.info(device)
    if save_ae_state_dict:
        autoencoder.save_parameters(autoencoder_save_param_path)

    optimizer = optimizer_class(
        list(autoencoder.parameters()), **clustering_optimizer_params
    )

    # workaround to get an empty optimizer
    dummy_param = torch.nn.Parameter(torch.empty(0))
    projection_axis_optimizer = optimizer_class(
        [dummy_param], **projection_axis_optimizer_params
    )
    projection_axis_optimizer.param_groups = []

    # Setup DeepECT Module
    dipect_module = _DipECT_Module(
        trainloader,
        autoencoder,
        projection_axis_optimizer,
        device,
        random_state,
        augmentation_invariance,
    ).to(device)

    # DeepECT Training loop
    dipect_module.fit(
        autoencoder.to(device),
        trainloader,
        testloader,
        Y,
        max_epochs,
        pruning_threshold,
        grow_interval,
        use_pvalue,
        max_leaf_nodes,
        reconstruction_loss_weight,
        unimodality_treshhold,
        number_of_grow_steps,
        early_stopping,
        refinement_epochs,
        optimizer,
        projection_axis_optimizer,
        rec_loss_fn,
        device,
        logging_active,
        unimodal_loss_application,
        unimoal_loss_node_criteria_method,
        unimodal_loss_weight_function,
        unimodal_loss_weight_direction,
        unimodal_loss_weight,
        loss_weight_function_normalization,
        mulitmodal_loss_application,
        mulitmodal_loss_node_criteria_method,
        mulitmodal_loss_weight_function,
        mulitmodal_loss_weight_direction,
        multimodal_loss_weight,
        projection_axis_learning,
        pruning_strategy,
        pruning_factor,
        tree_growth_min_cluster_size,
        evaluate_after_n_epochs=evaluate_every_n_epochs,
    )
    # Get labels
    pred_tree: PredictionClusterTree = dipect_module.predict(testloader, autoencoder)
    classes = np.unique(Y).size
    metrics = {
        "acc": pred_tree.flat_accuracy(Y, classes),
        "nmi": pred_tree.flat_nmi(Y, classes),
        "ari": pred_tree.flat_ari(Y, classes),
        "dp": pred_tree.dendrogram_purity(Y),
        "lp": pred_tree.leaf_purity(Y)[0],
    }
    metrics["combined_metrics"] = metrics["acc"] + metrics["dp"] + metrics["lp"]
    train.report(metrics)
    if logging_active:
        logging.info(metrics)
    return pred_tree, autoencoder


class DipECT:
    """
    A deep hierarchical clustering algorithm based on the dip test for modality.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterward, a cluster tree will be grown and the AE will be optimized using the DipECT loss function.

    Parameters
    ----------
    batch_size : int
        Size of the data batches (default: 256).
    pretrain_optimizer_params : dict
        Parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3}).
    clustering_optimizer_params : dict
        Parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4}).
    projection_axis_learning_rate : float
        Learning rate for projection axes (default: 1e-5).
    projection_axis_learning : str
        Learning strategy for projection axes, options are None, "all", "only_leaf_nodes", "partial_leaf_nodes" (default: "all").
    optimizer_class : torch.optim.Optimizer
        The optimizer class (default: torch.optim.Adam).
    rec_loss_fn : torch.nn.modules.loss._Loss
        Loss function for the reconstruction (default: torch.nn.MSELoss()).
    autoencoder : torch.nn.Module
        The input autoencoder. If None, a new FeedforwardAutoencoder will be created (default: None).
    autoencoder_pretrain_n_epochs : int
        Number of epochs for pretraining the autoencoder (default: 50).
    reconstruction_loss_weight : float
        Weight for the reconstruction loss (default: None).
    autoencoder_param_path : str
        Path to save the autoencoder parameters (default: "pretrained_ae.pth").
    clustering_n_epochs : int
        Number of epochs for the actual clustering procedure (default: 40).
    embedding_size : int
        Size of the embedding within the autoencoder (default: 10).
    max_leaf_nodes : int
        Maximum number of leaf nodes in the cluster tree (default: 20).
    custom_dataloaders : tuple
        Tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position. If None, the default dataloaders will be used (default: None).
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn cluster assignments that are invariant to the augmentation transformations (default: False).
    pruning_threshold : float
        The threshold for pruning the tree (default: 0.1).
    pruning_strategy : str
        Strategy for pruning, options are "moving_average", "epoch_assessment" (default: "moving_average").
    pruning_factor : float
        Factor for pruning (default: 0.5).
    tree_growth_frequency : float
        Interval for growing the tree (default: 2.0).
    tree_growth_amount : int
        Number of grow steps (default: 1).
    tree_growth_upper_bound_leaf_nodes : int
        Maximum number of leaf nodes for tree growth (default: 20).
    tree_growth_use_unimodality_pvalue : bool
        Whether to use p-value for split criteria (default: False).
    tree_growth_unimodality_treshold : float
        Threshold for unimodality (default: 0.95).
    tree_growth_min_cluster_size : int
        Minimum cluster size to allow tree growth (default: 1000).
    unimodal_loss_application : str
        Specifies where the unimodal loss should be applied, options are None, "leaf_nodes", "all" (default: "leaf_nodes").
    unimodal_loss_node_criteria_method : str
        Method for node criteria, options are "tree_depth", "time_of_split" (default: "tree_depth").
    unimodal_loss_weight_function : str
        Weight function for unimodal loss, options are "linear", "exponential", None (default: "linear").
    unimodal_loss_weight_direction : str
        Weight direction for unimodal loss, options are "ascending", "descending" (default: "ascending").
    unimodal_loss_weight : float
        Weight for unimodal loss (default: 1.0).
    loss_weight_function_normalization : float
        Normalization factor for the loss weight function (default: -1).
    mulitmodal_loss_application : str
        Specifies where the multimodal loss should be applied, options are None, "leaf_nodes", "all" (default: "all").
    mulitmodal_loss_node_criteria_method : str
        Method for node criteria for multimodal loss, options are "tree_depth", "time_of_split" (default: "tree_depth").
    mulitmodal_loss_weight_function : str
        Weight function for multimodal loss, options are "linear", "exponential", None (default: None).
    mulitmodal_loss_weight_direction : str
        Weight direction for multimodal loss, options are "ascending", "descending" (default: "ascending").
    multimodal_loss_weight : float
        Weight for multimodal loss (default: 1.0).
    early_stopping : bool
        Whether to use early stopping (default: False).
    refinement_epochs : int
        Number of refinement epochs (default: 0).
    random_state : np.random.RandomState
        Use a fixed random state to get a repeatable solution. Can also be of type int (default: np.random.RandomState(42)).
    logging_active : bool
        Whether to log metrics (default: False).
    evaluate_every_n_epochs : int
        Interval to evaluate the model (default: 2).

    Attributes
    ----------
    tree_ : PredictionClusterTree
        The prediction cluster tree after training.
    autoencoder : torch.nn.Module
        The final autoencoder.
    """

    def __init__(
        self,
        # data
        batch_size: int = 256,
        custom_dataloaders: tuple = None,
        # optimizer
        pretrain_optimizer_params: dict = None,
        clustering_optimizer_params: dict = None,
        projection_axis_learning_rate: float = 1e-5,
        projection_axis_learning: str = "all",  # None, "all", "only_leaf_nodes", "partial_leaf_nodes"
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        # autoencoder
        rec_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        autoencoder: _AbstractAutoencoder = None,
        autoencoder_pretrain_n_epochs: int = 50,
        reconstruction_loss_weight: float = None,  # None, float(1e-4, 1e4)
        autoencoder_param_path: str = "",
        # clustering
        clustering_n_epochs: int = 40,
        embedding_size: int = 10,
        augmentation_invariance: bool = False,
        # pruning
        pruning_threshold: float = 0.1,
        pruning_strategy: str = "moving_average",  # "epoch_assessment", "moving_average",
        pruning_factor: float = 0.5,
        # tree growth
        tree_growth_frequency: float = 2.0,
        tree_growth_amount: int = 1,
        tree_growth_upper_bound_leaf_nodes: int = 20,
        tree_growth_use_unimodality_pvalue: bool = False,
        tree_growth_unimodality_treshold: float = 0.95,
        tree_growth_min_cluster_size: int = 1000,
        # unimodal
        unimodal_loss_application: str = "leaf_nodes",  # None, "leaf_nodes", "all"
        unimodal_loss_node_criteria_method: str = "tree_depth",  # "tree_depth", "time_of_split"
        unimodal_loss_weight_function: str = "linear",  # "linear", "exponential", None
        unimodal_loss_weight_direction: str = "ascending",  # "ascending", "descending"
        unimodal_loss_weight: float = 1.0,
        loss_weight_function_normalization=-1,  # -1 (no normalization), else normalization term ((np.log2(self.max_leaf_nodes) - 1) works good and was until now always used)
        # multimodal
        mulitmodal_loss_application: str = "all",  # None, "leaf_nodes", "all"
        mulitmodal_loss_node_criteria_method: str = "tree_depth",  # "tree_depth", "time_of_split"
        mulitmodal_loss_weight_function: str = None,  # "linear", "exponential", None
        mulitmodal_loss_weight_direction: str = "ascending",  # "ascending", "descending"
        multimodal_loss_weight: float = 1.0,
        # utility
        early_stopping: bool = False,
        refinement_epochs: int = 0,
        random_state: np.random.RandomState = np.random.RandomState(42),
        logging_active: bool = False,
        evaluate_every_n_epochs: int = 2,
    ):
        self.batch_size = batch_size
        self.pretrain_optimizer_params = (
            {"lr": 1e-3}
            if pretrain_optimizer_params is None
            else pretrain_optimizer_params
        )
        self.clustering_optimizer_params = (
            {"lr": 1e-4}
            if clustering_optimizer_params is None
            else clustering_optimizer_params
        )
        self.projection_axis_optimizer_params = {"lr": projection_axis_learning_rate}
        self.pretrain_epochs = autoencoder_pretrain_n_epochs
        self.max_epochs = clustering_n_epochs
        self.grow_interval = tree_growth_frequency
        self.use_pvalue = tree_growth_use_unimodality_pvalue
        self.pruning_threshold = pruning_threshold
        self.optimizer_class = optimizer_class
        self.rec_loss_fn = rec_loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.max_leaf_nodes = tree_growth_upper_bound_leaf_nodes
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.unimodality_treshhold = tree_growth_unimodality_treshold
        self.number_of_grow_steps = tree_growth_amount
        self.early_stopping = early_stopping
        self.refinement_epochs = refinement_epochs
        self.random_state = random_state
        self.autoencoder_param_path = autoencoder_param_path
        self.logging_active = logging_active
        self.unimodal_loss_application = unimodal_loss_application
        self.unimoal_loss_node_criteria_method = unimodal_loss_node_criteria_method
        self.unimodal_loss_weight_function = unimodal_loss_weight_function
        self.unimodal_loss_weight_direction = unimodal_loss_weight_direction
        self.unimodal_loss_weight = unimodal_loss_weight
        self.loss_weight_function_normalization = loss_weight_function_normalization
        self.mulitmodal_loss_application = mulitmodal_loss_application
        self.mulitmodal_loss_node_criteria_method = mulitmodal_loss_node_criteria_method
        self.mulitmodal_loss_weight_function = mulitmodal_loss_weight_function
        self.mulitmodal_loss_weight_direction = mulitmodal_loss_weight_direction
        self.multimodal_loss_weight = multimodal_loss_weight
        self.projection_axis_learning = projection_axis_learning
        self.pruning_strategy = pruning_strategy
        self.pruning_factor = pruning_factor
        self.tree_growth_min_cluster_size = tree_growth_min_cluster_size
        self.evaluate_every_n_epochs = evaluate_every_n_epochs

    def fit_predict(self, X: np.ndarray, Y: np.ndarray) -> "DipECT":
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            The given data set as a 2D-array of shape (#samples, #features).
        Y : np.ndarray
            The given labels for the data set.

        Returns
        -------
        self : DipECT
            This instance of the DipECT algorithm.
        """
        augmentation_invariance_check(
            self.augmentation_invariance, self.custom_dataloaders
        )
        os.environ["OMP_NUM_THREADS"] = "2"
        os.environ["MKL_NUM_THREADS"] = "2"
        os.environ["OPENBLAS_NUM_THREADS"] = "2"
        os.environ["BLIS_NUM_THREADS"] = "2"
        os.environ["SKLEARN_SEED"] = str(self.random_state.get_state()[1][0])
        tree, autoencoder = _dipect(
            X,
            Y,
            self.batch_size,
            self.pretrain_optimizer_params,
            self.clustering_optimizer_params,
            self.projection_axis_optimizer_params,
            self.pretrain_epochs,
            self.max_epochs,
            self.pruning_threshold,
            self.grow_interval,
            self.use_pvalue,
            self.optimizer_class,
            self.rec_loss_fn,
            self.autoencoder,
            self.embedding_size,
            self.max_leaf_nodes,
            self.reconstruction_loss_weight,
            self.unimodality_treshhold,
            self.number_of_grow_steps,
            self.early_stopping,
            self.refinement_epochs,
            self.custom_dataloaders,
            self.augmentation_invariance,
            self.random_state,
            self.logging_active,
            self.autoencoder_param_path,
            self.unimodal_loss_application,
            self.unimoal_loss_node_criteria_method,
            self.unimodal_loss_weight_function,
            self.unimodal_loss_weight_direction,
            self.unimodal_loss_weight,
            self.loss_weight_function_normalization,
            self.mulitmodal_loss_application,
            self.mulitmodal_loss_node_criteria_method,
            self.mulitmodal_loss_weight_function,
            self.mulitmodal_loss_weight_direction,
            self.multimodal_loss_weight,
            self.projection_axis_learning,
            self.pruning_strategy,
            self.pruning_factor,
            self.tree_growth_min_cluster_size,
            self.evaluate_every_n_epochs,
        )
        self.tree_ = tree
        self.autoencoder = autoencoder
        return self


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    start = datetime.datetime.now()
    dataset, labels = load_mnist(return_X_y=True)
    dataset = dataset / 255.0
    # dataset = dataset.reshape(-1, 32, 32, 3).astype("float32")
    # dataset = dataset.transpose((0, 3, 1, 2)) / 255.0
    # autoencoder = ConvolutionalAutoencoder(32, [512, 10])
    autoencoder = FeedforwardAutoencoder([dataset.shape[1], 500, 500, 2000, 10])

    dipect = DipECT(
        batch_size=256,
        autoencoder=autoencoder,
        autoencoder_param_path="practical/DeepClustering/DipECT/autoencoder/feedforward_mnist_100_21.pth",
        random_state=np.random.RandomState(21),
        autoencoder_pretrain_n_epochs=100,
        logging_active=True,
        clustering_n_epochs=60,
        clustering_optimizer_params={"lr": 1e-4},
        early_stopping=False,
        loss_weight_function_normalization=-1,
        mulitmodal_loss_application="all",
        mulitmodal_loss_node_criteria_method="tree_depth",  # "time_of_split",
        mulitmodal_loss_weight_direction="ascending",
        mulitmodal_loss_weight_function="linear",  # "linear",
        multimodal_loss_weight=1.6529335369273173,
        projection_axis_learning="only_leaf_nodes",
        projection_axis_learning_rate=5e-4,
        pruning_factor=1.0,
        pruning_strategy="epoch_assessment",
        pruning_threshold=dataset.shape[0] / 35,
        tree_growth_min_cluster_size=dataset.shape[0] / 35,
        reconstruction_loss_weight=1,
        refinement_epochs=0,
        tree_growth_amount=3,
        tree_growth_frequency=1.0,
        tree_growth_unimodality_treshold=0.975,
        tree_growth_upper_bound_leaf_nodes=100,
        tree_growth_use_unimodality_pvalue=True,
        unimodal_loss_application="all",
        unimodal_loss_node_criteria_method="tree_depth",
        unimodal_loss_weight=3.5,
        unimodal_loss_weight_direction="descending",
        unimodal_loss_weight_function="log",
    )
    dipect.fit_predict(dataset, labels)
    print(
        f"-------------------------------------------Time needed: {(datetime.datetime.now()-start).total_seconds()/60}min"
    )
