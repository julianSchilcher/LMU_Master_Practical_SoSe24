import logging
import os
import sys

sys.path.append(os.getcwd())

from typing import List, Union

import numpy as np
import torch
import torch.utils.data
from clustpy.deep._data_utils import (augmentation_invariance_check,
                                      get_dataloader)
from clustpy.deep._train_utils import get_trained_autoencoder
from clustpy.deep._utils import detect_device, encode_batchwise, set_torch_seed
from clustpy.deep.autoencoders._abstract_autoencoder import \
    _AbstractAutoencoder
from clustpy.deep.dipencoder import _Dip_Gradient
from clustpy.utils import dip_pval, dip_test
from sklearn.cluster import KMeans
from tqdm import tqdm

from practical.DeepClustering.DeepECT.metrics import (PredictionClusterNode,
                                                      PredictionClusterTree)


# replaces the dip module
class Cluster_Node: 
    """
    This class represents a cluster node within a binary cluster tree.
    Each node in a cluster tree represents a cluster. The cluster is stored through its center (self.center).
    During the assignment of a new minibatch to the cluster tree, each node stores the samples nearest to its center (self.assignments).
    The centers of leaf nodes are optimized through autograd, whereas the center of inner nodes is adapted analytically with weights for each of its children stored in self.weights.
    """

    def __init__(
        self,
        device: torch.device,
        id: int = 0,
        parent: "Cluster_Node" = None,
        split_id: int = 0, # keep for prediction phase
        split_level: int = 0, # used to adapt weight of L_uni depending on how deep we are in the tree
        number_assignments: int = 0, # used to initialise pruning_indicator
    ) -> "Cluster_Node":
        """
        Constructor for class Cluster_Node

        Parameters
        ----------
        center : np.ndarray
            The initial center for this node.
        device : torch.device
            The device to be trained on.
        id : int, optional
            The ID of the node, by default 0.
        parent : Cluster_Node, optional
            The parent node, by default None.
        split_id : int, optional
            The ID of the split, by default 0.
        weight : int, optional
            The weight of the node, by default 1.

        Returns
        -------
        Cluster_Node
            The initialized Cluster_Node object.
        """
        self.device = device
        self.pruning_indicator = number_assignments
        self.higher_projection_child = None # higher_projection_child
        self.lower_projection_child = None # lower_projection_child
        self.projection_axis = None
        self.assignments: Union[torch.Tensor, None] = None
        self.assignment_indices: Union[torch.Tensor, None] = None # important for PredictionClusterTree
        # self.sum_squared_dist: Union[torch.Tensor, None] = None
        self.id = id
        self.split_id = split_id
        self.split_level = split_level
        self.parent = parent
        self.check_invariant()
    
    def check_invariant(self):
        if not((self.is_leaf_node() and self.projection_axis is None) or (not self.is_leaf_node() and self.projection_axis is not None)): # leaf node <=> projection_axis=None
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
        self.sum_squared_dist = None

    def is_leaf_node(self) -> bool:
        """
        Checks if this node is a leaf node.

        Returns
        -------
        bool
            True if the node is a leaf node, else False.
        """
        return self.higher_projection_child is None and self.lower_projection_child is None

    def prune(self):
        """
        Prunes the tree by removing all nodes below this node.
        """
        if self.higher_projection_child is not None:
            self.higher_projection_child.prune()
        if self.lower_projection_child is not None:
            self.lower_projection_child.prune()

        if(not self.is_leaf_node()):
            self.projection_axis.requires_grad = False
        self.assignments = None
        self.assignment_indices = None
        self.sum_squared_dist = None

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
        optimizer : Union[torch.optim.Optimizer, None]
            The optimizer to be used.
        left_child_centers : np.ndarray
            Initial centers for the left child.
        left_child_weight : int
            Weight of the left child.
        right_child_centers : np.ndarray
            Initial centers for the right child.
        right_child_weight : int
            Weight of the right child.
        max_id : int, optional
            The maximum ID, by default 0.
        max_split_id : int, optional
            The maximum split ID, by default 0.
        """
        # set projection axis
        self.projection_axis = torch.nn.Parameter(torch.nn.Parameter(torch.from_numpy(projection_axis).float()))

        if projection_axis_optimizer is not None:
            # self.add_projection_axis_to_optimizer(optimizer, self.projection_axis) # using just one otptimizer
            projection_axis_optimizer.add_param_group({"params": self.projection_axis})
        self.higher_projection_child = Cluster_Node(
            self.device,
            max_id + 1,
            self,
            max_split_id + 1,
            self.split_level + 1,
            num_assignments_higher_projection_child
        )
        self.lower_projection_child = Cluster_Node(
            self.device,
            max_id + 2,
            self,
            max_split_id + 1,
            self.split_level + 1,
            num_assignments_lower_projection_child
        )
        # self.from_leaf_to_inner()

        self.check_invariant()

    def add_projection_axis_to_optimizer(self, optimizer: torch.optim.Optimizer, new_axis: torch.nn.Parameter):

        for param_group in optimizer.param_groups:
            if param_group.get('name') == "projection_axes":
                param_group['params'].extend([new_axis]) # optimizer expects a list of parameters
                return

        raise ValueError("Parameter group with with name projection_axes not initialised yet. Please initialise it by calling optimizer.add_param_group({'params': [], 'lr': desired_learning_rate, 'name': 'projection_axes'},)")
    
    def adapt_pruning_indicator(self, number_assignments: int):
        # adapt pruning indicator with EMA
        self.pruning_indicator = 0.5*(self.pruning_indicator + number_assignments)


class Cluster_Tree:
    """
    This class represents a binary cluster tree. It provides multiple
    functionalities used for improving the cluster tree, like calculating
    the DC and NC losses for the tree and assigning samples of a minibatch
    to the appropriate nodes. Furthermore, it provides methods for
    growing and pruning the tree as well as the analytical adaptation
    of the inner nodes.
    """

    def __init__(
        self,
        trainloader: torch.utils.data.DataLoader,
        autoencoder: torch.nn.Module,
        projection_axis_optimizer: torch.optim.Optimizer,
        device: torch.device,
        max_leaf_nodes: int
    ) -> "Cluster_Tree":
        """
        Constructor for the Cluster_Tree class.

        Parameters
        ----------
        init_leafnode_centers : np.ndarray
            The centers of the two initial leaf nodes of the tree
            given as an array of shape (2, #embedd_features).
        device : torch.device
            The device to be trained on.
        """
        # initialise cluster tree
        self.device = device
        self.max_leaf_nodes = max_leaf_nodes
        self.root = Cluster_Node(device)
        embedded_data = encode_batchwise(trainloader, autoencoder, self.device)
        axis, number_left_assignments, number_right_assignments = self.get_inital_projection_axis(embedded_data)
        self.root.expand_tree(axis, projection_axis_optimizer, number_left_assignments, number_right_assignments)
        

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
                1 + count_recursive(node.higher_projection_child) + count_recursive(node.lower_projection_child)
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
        Returns the inital projection axis for the data in the given trainloader. Furthermore, the size of the higher projection cluster and the lower projection cluster will be returned (e.g to initialise pruning indicator).
        """
        # init projection axis on full dataset
        kmeans = KMeans(n_clusters=2, n_init=10).fit(embedded_data)
        kmeans_centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        axis = kmeans_centers[1] - kmeans_centers[0]
        # higher projection by cluster 1 since axis points to cluster 1
        return kmeans_centers[0] - kmeans_centers[1], np.sum(labels == 0), np.sum(labels == 1)
    

    def clear_node_assignments(self):
        """
        Clears the assignments for all nodes in the tree.
        """
        self.root.clear_assignments()

    def assign_to_tree(
        self, data_embedded: torch.Tensor, set_pruning_incidator: bool = False
    ):
        """
        Assigns all samples in the minibatch to their nearest nodes in the cluster tree.
        It is performed bottom-up, so each sample is first assigned to its nearest
        leaf node. Afterwards, the samples are assigned recursively to
        the inner nodes by merging the assignments of the child node.

        Parameters
        ----------
        minibatch_embedded : torch.Tensor
            The minibatch with shape (#samples, #emb_features).
        compute_sum_dist : bool, optional
            Whether to compute the sum of squared distances, by default False.
        """
        # clear all assignments
        self.clear_node_assignments()
        # assign top-down
        self.assign_top_down(self.root, data_embedded, torch.tensor([i for i in range(len(data_embedded))]), set_pruning_incidator)

    def assign_top_down(self, node: Cluster_Node, embedded_data: torch.Tensor, embedded_data_indices: torch.Tensor, set_pruning_incidator: bool):
        
        if set_pruning_incidator:
                node.adapt_pruning_indicator(len(embedded_data))
        
        if embedded_data.numel() == 0:
            return

        node.assignments = embedded_data
        node.assignment_indices = embedded_data_indices
        if node.is_leaf_node():
            return
        
        labels = self.predict_subclusters(node)
        if node.higher_projection_child is not None:
            self.assign_top_down(node.higher_projection_child, embedded_data[labels == 1], embedded_data_indices[labels == 1], set_pruning_incidator)
        if node.lower_projection_child is not None:
            self.assign_top_down(node.lower_projection_child, embedded_data[labels == 0], embedded_data_indices[labels == 0], set_pruning_incidator)


    def predict_subclusters(self, node: Cluster_Node):
        if node.assignments.numel() == 1: ## !!!!!!!!!!!!!!!!!!!!!! Implement Pruning!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            return np.array([1])
        projections = torch.matmul(node.assignments.detach().clone().cpu().float(), node.projection_axis.detach().clone().reshape(-1,1)).numpy()[:,0] # remove second dimension after projection
        sorted_indices = projections.argsort()
        _, modal_interval, modal_triangle = dip_test(projections[sorted_indices], is_data_sorted=True, just_dip=False)
        index_lower, index_upper = modal_interval
        _, mid_point_triangle, _ = modal_triangle
        if projections[sorted_indices[mid_point_triangle]] > projections[sorted_indices[index_upper]]:
                threshold =  (projections[sorted_indices[mid_point_triangle]] + projections[sorted_indices[index_upper]])/2
        else:
                threshold =  (projections[sorted_indices[mid_point_triangle]] + projections[sorted_indices[index_lower]])/2
        labels = np.zeros(len(node.assignments))
        labels[projections >= threshold] = 1
        return labels
        

    def prune_tree(self, pruning_threshold: float):
        """
        Prunes the tree by removing nodes with pruning indicators below the pruning threshold.

        Parameters
        ----------
        pruning_threshold : float
            The threshold value for pruning. Nodes with pruning indicators below this threshold will be removed.

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
                "higher_projection_child" if child_attr == "lower_projection_child" else "lower_projection_child"
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

                print(
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
                        and self.root.higher_projection_child.pruning_indicator < pruning_threshold
                    ):
                        prune_node(self.root, "higher_projection_child")
                        result = True
                    elif (
                        self.root.lower_projection_child
                        and self.root.lower_projection_child.pruning_indicator < pruning_threshold
                    ):
                        prune_node(self.root, "higher_projection_child")
                        result = True
            return result

        return prune_recursive(self.root)

    def grow_tree(
        self,
        dataloader: torch.utils.data.DataLoader,
        autoencoder: torch.nn.Module,
        projection_axis_optimizer: torch.optim.Optimizer,
        unimodal_treshhold: float,
        use_pvalue: bool = False,
    ) -> bool:
        """
        Grows the tree at the leaf node with the highest squared distance
        between its assignments and center. The distance is not normalized,
        so larger clusters will be chosen.

        We transform the dataset (or a representative sub-sample of it)
        onto the embedded space. Then, we determine the leaf node
        with the highest sum of squared distances between its center
        and the assigned data points. We selected this rule because it
        provides a good balance between the number of data points
        and data variance for this cluster.
        Next, we split the selected node and attach two new leaf
        nodes to it as children. We determine the initial centers for
        these new leaf nodes by applying two-means (k-means with
        k = 2) to the assigned data points.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            The data loader for the dataset.
        autoencoder : torch.nn.Module
            The autoencoder model for embedding the data.
        optimizer : torch.optim.Optimizer
            The optimizer for the autoencoder.
        device : Union[torch.device, str]
            The device to perform calculations on.

        Returns
        ----------
        Returns True if the algorithm should be stopped (uniform criteria met) or False otherwise
        """
        X_embedd = encode_batchwise(dataloader, autoencoder, self.device)
        self.assign_to_tree(torch.from_numpy(X_embedd))
  
        if use_pvalue:
            best_value = np.inf
        else:
            best_value = -np.inf
            total_assignments = sum([len(node.assignments) for node in self.leaf_nodes if node.assignments is not None])

        best_node_to_split = None
        best_node_axis = None
        best_node_number_assign_lower_projection_cluster = None
        best_node_number_assign_higher_projection_cluster = None
        for node in self.leaf_nodes:
            if node.assignments is None:
                continue
            node_data = node.assignments.numpy()
            if len(node_data) < 2:
                continue
            print("#assignments: ", len(node.assignments))
            axis, number_assign_lower_projection_cluster, number_assign_higher_projection_cluster = self.get_inital_projection_axis(node_data)
            projections = np.matmul(node_data, axis)
            dip_value = dip_test(projections, just_dip=True, is_data_sorted=False)
            # pvalue gives the probability for unimodality (smaller dip value, higher p value)
            pvalue = dip_pval(dip_value, len(node.assignments))
            

            # the more samples, the smaller the dip value, consider this:
            if use_pvalue:
                current_value = pvalue
                better = current_value < best_value
            else:
                current_value = dip_value + 0.5*len(node.assignments)/(4*total_assignments)
                better = current_value > best_value

            if better and pvalue < unimodal_treshhold: 
                best_value = current_value
                best_node_axis = axis
                best_node_to_split = node
                best_node_number_assign_lower_projection_cluster = number_assign_lower_projection_cluster
                best_node_number_assign_higher_projection_cluster = number_assign_higher_projection_cluster
                    
        if np.abs(best_value) == np.inf:
            return True # stop algorithm
        else:
            print("#assignments best node: ", len(best_node_to_split.assignments))
            best_node_to_split.expand_tree(best_node_axis, projection_axis_optimizer, best_node_number_assign_higher_projection_cluster, best_node_number_assign_lower_projection_cluster, max([leaf.id for leaf in self.leaf_nodes]), max([node.split_id for node in self.nodes]))
            return False
    
    def improve_space(self, embedded_data: torch.Tensor, embedded_augmented_data: Union[torch.Tensor | None], projection_axis_optimizer: torch.optim.Optimizer, unimodal_loss_increase_method: str):
        self.assign_to_tree(embedded_data, set_pruning_incidator=True)
        loss = self._improve_space_recursive(self.root, projection_axis_optimizer, 0, unimodal_loss_increase_method, embedded_augmented_data)
        return loss

    def _improve_space_recursive(self, node: Cluster_Node, projection_axis_optimizer: torch.optim.Optimizer, loss: torch.Tensor, unimodal_loss_increase_method: str, embedded_augmented_data: Union[torch.Tensor | None]):
        if node.is_leaf_node():
            return loss

        self._adjust_axis(node, projection_axis_optimizer)

        axis = node.projection_axis.detach().clone()
        
        higher_projection_child_improvement = node.higher_projection_child.assignments is not None and node.higher_projection_child.assignments.numel() > 1
        lower_projection_child_improvement = node.lower_projection_child.assignments is not None and node.lower_projection_child.assignments.numel() > 1

        if higher_projection_child_improvement and lower_projection_child_improvement:
            unimodal_loss_weight = self._calc_unimodal_loss_weight(node, unimodal_loss_increase_method)
            if embedded_augmented_data  is not None:
                higher_projection_cluster = torch.cat((node.higher_projection_child.assignments, embedded_augmented_data[node.higher_projection_child.assignment_indices]), dim=0)
                lower_projection_cluster = torch.cat((node.lower_projection_child.assignments, embedded_augmented_data[node.lower_projection_child.assignment_indices]), dim=0)
            else:
                higher_projection_cluster = node.higher_projection_child.assignments
                lower_projection_cluster = node.lower_projection_child.assignments
            
            L_unimodal = (unimodal_loss_weight[0]*_Dip_Gradient.apply(higher_projection_cluster, axis) + unimodal_loss_weight[1]*_Dip_Gradient.apply(lower_projection_cluster, axis))/2
            L_multimodal = _Dip_Gradient.apply(torch.cat((higher_projection_cluster, lower_projection_cluster), dim=0), axis)
            loss = loss + L_unimodal - L_multimodal
        
        if higher_projection_child_improvement:
            loss_higher_projection_child = self._improve_space_recursive(node.higher_projection_child, projection_axis_optimizer, loss, unimodal_loss_increase_method, embedded_augmented_data)
        else:
            loss_higher_projection_child = 0
        if lower_projection_child_improvement:
            loss_lower_projection_child = self._improve_space_recursive(node.lower_projection_child, projection_axis_optimizer, loss, unimodal_loss_increase_method, embedded_augmented_data)
        else:
            loss_lower_projection_child = 0

        return loss_higher_projection_child + loss_lower_projection_child


    def _adjust_axis(self, node: Cluster_Node, projection_axis_optimizer: torch.optim.Optimizer):
        projection_axis_optimizer.zero_grad()
        # data gradients should not be stored
        data = node.assignments.detach().clone()
        loss = -_Dip_Gradient.apply(data, node.projection_axis)
        loss.backward()
        projection_axis_optimizer.step()
    
    def _calc_unimodal_loss_weight(self, node: Cluster_Node, method: str):
        max_depth_balanced_tree = np.log2(self.max_leaf_nodes)
        if method == "linear":
            weight = (1/(max_depth_balanced_tree - 1)) * node.split_level
            return (weight, weight)
        elif method == "exponential":
            weight = np.exp2(node.split_level - max_depth_balanced_tree - 1)
            return (weight, weight)
        elif method == "noUnimodalLoss":
            return (0, 0)
        elif method == "justLeafs":
            # turn unimodal loss off for split nodes
            f = lambda node: 1.0 if node.is_leaf_node() else 0
            return (f(node.higher_projection_child), f(node.lower_projection_child))



        


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
        pred_node = PredictionClusterNode(
            node.id, node.split_id, None
        )
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
    The _DeepECT_Module. Contains most of the algorithm specific procedures like the loss and tree-grow functions.

    Parameters
    ----------
    init_centers : np.ndarray
        The initial cluster centers
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn
        cluster assignments that are invariant to the augmentation transformations (default: False)

    Attributes
    ----------
    cluster_tree: Cluster_Node
    augmentation_invariance : bool
        Is augmentation invariance used
    """

    def __init__(
        self,
        trainloader: torch.utils.data.DataLoader,
        autoencoder: torch.nn.Module,
        projection_axis_optimizer: torch.optim.Optimizer,
        device: torch.device,
        random_state: np.random.RandomState,
        max_leaf_nodes: int,
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
            max_leaf_nodes
        )

    def fit(
        self,
        autoencoder: torch.nn.Module,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        max_epochs: int,
        pruning_threshold: float,
        grow_interval: int,
        use_pvalue: bool,
        unimodal_loss_increase_method: str,
        max_leaf_nodes: int,
        reconstruction_loss_weight: float,
        unimodal_treshhold: float,
        optimizer: torch.optim.Optimizer,
        projection_axis_optimizer: torch.optim.Optimizer,
        rec_loss_fn: torch.nn.modules.loss._Loss,
        device: Union[torch.device | str],
        logging_active: bool,
    ) -> "_DipECT_Module":
        """
        Trains the _DeepECT_Module in place.

        Parameters
        ----------
        autoencoder : torch.nn.Module
            The autoencoder used for training
        trainloader : torch.utils.data.DataLoader
            DataLoader for training data
        testloader : torch.utils.data.DataLoader
            DataLoader for testing data
        max_iterations : int
            Maximum number of iterations for training
        pruning_threshold : float
            Threshold for pruning the cluster tree
        grow_interval : int
            Interval for growing the cluster tree
        max_leaf_nodes : int
            Maximum number of leaf nodes in the cluster tree
        optimizer : torch.optim.Optimizer
            Optimizer for training
        rec_loss_fn : torch.nn.modules.loss._Loss
            Loss function for reconstruction
        device : Union[torch.device, str]
            Device for training (e.g., "cuda" or "cpu")

        Returns
        -------
        self : _DeepECT_Module
            This instance of the _DeepECT_Module
        """
        # mov_dc_loss = 0.0
        # mov_nc_loss = 0.0
        mov_rec_loss = 0.0
        mov_rec_loss_aug = 0.0
        mov_loss = 0.0

        # stop_algorithm = False
        # refinement_epochs = 2
        # refinement_counter = 0

        for epoch in range(max_epochs):

            with tqdm(trainloader, unit="batch") as tepoch:
                self.cluster_tree.prune_tree(pruning_threshold)

                if (epoch > 0 and epoch % grow_interval == 0) or self.cluster_tree.number_nodes < 3:
                        if len(self.cluster_tree.leaf_nodes) < max_leaf_nodes:
                            stop_algorithm = self.cluster_tree.grow_tree(testloader, autoencoder, projection_axis_optimizer, unimodal_treshhold, use_pvalue)
                            if stop_algorithm:
                                print("Stopped algorithm earlier since unimodality treshhold is reached")
                                break


                for batch in tepoch:
                    
                    tepoch.set_description(f"Epoch {epoch}/{max_epochs}")

                    optimizer.zero_grad()

                    if self.augmentation_invariance:
                        idxs, M, M_aug = batch
                    else:
                        idxs, M = batch

                    # calculate autoencoder loss
                    rec_loss, embedded, reconstructed = autoencoder.loss(
                        [idxs, M], rec_loss_fn, self.device
                    )
                    if self.augmentation_invariance:
                        rec_loss_aug, embedded_aug, reconstructed_aug = (
                            autoencoder.loss([idxs, M_aug], rec_loss_fn, self.device)
                        )

                    # calculate cluster loss
                    cluster_loss = self.cluster_tree.improve_space(embedded, embedded_aug if self.augmentation_invariance else None, projection_axis_optimizer, unimodal_loss_increase_method)

                    self.cluster_tree.clear_node_assignments()

                    # if self.cluster_tree.prune_tree(pruning_threshold):
                    #     cluster_loss = torch.tensor([0.0], dtype=torch.float, device=device)

                    if reconstruction_loss_weight is None:
                        reconstruction_loss_weight = 1 / rec_loss.detach() # /(4* rec_loss.detach())

                    if self.augmentation_invariance:
                        loss = cluster_loss + reconstruction_loss_weight*0.5*(rec_loss + rec_loss_aug)
                        mov_rec_loss_aug += rec_loss_aug.item()
                    else:
                        loss = cluster_loss  + rec_loss*reconstruction_loss_weight
                        

                    mov_rec_loss += rec_loss.item()
                    mov_loss += loss.item()


                    loss.backward()
                    optimizer.step()
            if logging_active:
                log_epoch_nr = epoch + 1 # to avoid division by zero 
                logging.info(
                  f"epoch: {log_epoch_nr} - moving averages: mov_rec_loss: {mov_rec_loss/log_epoch_nr} "
                  f"mov_loss: {mov_loss/log_epoch_nr} {f'rec_loss_aug: {mov_rec_loss_aug/log_epoch_nr}' if self.augmentation_invariance else ''} total_loss: {mov_loss/log_epoch_nr}"
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
            DataLoader for the samples to be predicted
        autoencoder: torch.nn.Module
            Autoencoder model for calculating the embeddings

        Returns
        -------
        pred_tree : PredictionClusterTree
            The prediction cluster tree with assigned samples
        """
        # get prediction tree
        pred_tree = transform_cluster_tree_to_pred_tree(self.cluster_tree)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predict"):
                # calculate embeddings of the samples which should be predicted
                batch_data = batch[1].to(self.device)
                indices = batch[0].to(self.device)
                embeddings = autoencoder.encode(batch_data)
                # assign the embeddings to the cluster tree
                self.cluster_tree.assign_to_tree(embeddings)
                # use assignment indices for prediction tree
                for node in self.cluster_tree.leaf_nodes:
                    pred_tree[node.id].assign_batch(indices, node.assignment_indices)
                self.cluster_tree.clear_node_assignments()

        return pred_tree

def _dipect(
    X: np.ndarray,
    batch_size: int,
    pretrain_optimizer_params: dict,
    clustering_optimizer_params: dict,
    projection_axis_optimizer_params: dict,
    pretrain_epochs: int,
    max_epochs: int,
    pruning_threshold: float,
    grow_interval: int,
    use_pvalue: bool,
    optimizer_class: torch.optim.Optimizer,
    rec_loss_fn: torch.nn.modules.loss._Loss,
    autoencoder: _AbstractAutoencoder,
    embedding_size: int,
    unimodal_loss_increase_method: str,
    max_leaf_nodes: int,
    reconstruction_loss_weight: float,
    unimodal_treshhold: float,
    custom_dataloaders: tuple,
    augmentation_invariance: bool,
    random_state: np.random.RandomState,
    logging_active: bool,
    autoencoder_save_param_path: str,
):
    """
    Start the actual DeepECT clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray
        The given data set. Can be a np.ndarray or a torch.Tensor
    batch_size : int
        Size of the data batches
    pretrain_optimizer_params : dict
        Parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate
    clustering_optimizer_params : dict
        Parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        Number of epochs for the pretraining of the autoencoder
    max_iterations : int
        Number of iterations for the actual clustering procedure
    pruning_threshold : float
        The threshold for pruning the tree
    grow_interval : int
        Interval for growing the tree
    optimizer_class : torch.optim.Optimizer
        The optimizer class
    rec_loss_fn : torch.nn.modules.loss._Loss
        Loss function for the reconstruction
    autoencoder : torch.nn.Module
        The input autoencoder
    embedding_size : int
        Size of the embedding within the autoencoder
    custom_dataloaders : tuple
        Tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn cluster assignments that are invariant to the augmentation transformations
    random_state : np.random.RandomState
        Use a fixed random state to get a repeatable solution
    autoencoder_save_param_path : str
        Path to save the autoencoder parameters

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by a final KMeans execution,
        The cluster centers as identified by a final KMeans execution,
        The labels as identified by DeepECT after the training terminated,
        The cluster centers as identified by DeepECT after the training terminated,
        The final autoencoder
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
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
    autoencoder = get_trained_autoencoder(trainloader, pretrain_optimizer_params, pretrain_epochs, device,
                                          optimizer_class, rec_loss_fn, embedding_size, autoencoder)

    # (
    #     device,
    #     trainloader,
    #     testloader,
    #     autoencoder,
    #     _,
    #     _,
    #     _,
    #     init_leafnode_centers,
    #     _,
    # ) = get_standard_initial_deep_clustering_setting(
    #     X,
    #     2,
    #     batch_size,
    #     pretrain_optimizer_params,
    #     pretrain_epochs,
    #     optimizer_class,
    #     rec_loss_fn,
    #     autoencoder,
    #     embedding_size,
    #     custom_dataloaders,
    #     KMeans,
    #     {"n_init": 20, "random_state": random_state},
    #     random_state,
    # )

    print(device)
    if save_ae_state_dict:
        autoencoder.save_parameters(autoencoder_save_param_path)

    
    optimizer = optimizer_class(
        list(autoencoder.parameters()), **clustering_optimizer_params
    )
    
    # workaround to get an empty optimizer
    dummy_param = torch.nn.Parameter(torch.empty(0))
    projection_axis_optimizer = optimizer_class([dummy_param], **projection_axis_optimizer_params)
    projection_axis_optimizer.param_groups = []
    # optimizer.add_param_group({'params': [], 'lr': projection_axis_lr, 'name': 'projection_axes'}) # using just one optimizer (with different lr for projection axis and data)

    # Setup DeepECT Module
    dipect_module = _DipECT_Module(
        trainloader,
        autoencoder,
        projection_axis_optimizer,
        device,
        random_state,
        max_leaf_nodes,
        augmentation_invariance,
    ).to(device)

    # DeepECT Training loop
    dipect_module.fit(
        autoencoder.to(device),
        trainloader,
        testloader,
        max_epochs,
        pruning_threshold,
        grow_interval,
        use_pvalue,
        unimodal_loss_increase_method, 
        max_leaf_nodes,
        reconstruction_loss_weight,
        unimodal_treshhold,
        optimizer,
        projection_axis_optimizer,
        rec_loss_fn,
        device,
        logging_active
    )
    # Get labels
    deepect_tree: PredictionClusterTree = dipect_module.predict(
        testloader, autoencoder
    )
    return deepect_tree, autoencoder



class DipECT:
    """
    The Deep Embedded Cluster Tree (DeepECT) algorithm.
    First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
    Afterward, a cluster tree will be grown and the AE will be optimized using the DeepECT loss function.

    Parameters
    ----------
    batch_size : int
        Size of the data batches (default: 256)
    pretrain_optimizer_params : dict
        Parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
    clustering_optimizer_params : dict
        Parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
    pretrain_epochs : int
        Number of epochs for the pretraining of the autoencoder (default: 50)
    max_iterations : int
        Number of iterations for the actual clustering procedure (default: 50000)
    grow_interval : int
        Interval for growing the tree (default: 500)
    pruning_threshold : float
        The threshold for pruning the tree (default: 0.1)
    optimizer_class : torch.optim.Optimizer
        The optimizer class (default: torch.optim.Adam)
    rec_loss_fn : torch.nn.modules.loss._Loss
        Loss function for the reconstruction (default: torch.nn.MSELoss())
    autoencoder : torch.nn.Module
        The input autoencoder. If None, a new FeedforwardAutoencoder will be created (default: None)
    embedding_size : int
        Size of the embedding within the autoencoder (default: 10)
    max_leaf_nodes : int
        Maximum number of leaf nodes in the cluster tree (default: 20)
    custom_dataloaders : tuple
        Tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position. If None, the default dataloaders will be used (default: None)
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn cluster assignments that are invariant to the augmentation transformations (default: False)
    random_state : np.random.RandomState
        Use a fixed random state to get a repeatable solution. Can also be of type int (default: None)
    autoencoder_param_path : str
        Path to save the autoencoder parameters (default: None)

    Attributes
    ----------
    tree_ : PredictionClusterTree
        The prediction cluster tree after training
    autoencoder : torch.nn.Module
        The final autoencoder
    """

    def __init__(
        self,
        batch_size: int = 256,
        pretrain_optimizer_params: dict = None,
        clustering_optimizer_params: dict = None,
        projection_axis_optimizer_params: dict = None,
        pretrain_epochs: int = 50,
        max_epochs: int = 40,
        grow_interval: int = 2,
        use_pvalue: bool = False,
        pruning_threshold: float = 0.1,
        optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
        rec_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(),
        autoencoder: _AbstractAutoencoder = None,
        embedding_size: int = 10,
        unimodal_loss_increase_method: str = "linear",
        max_leaf_nodes: int = 20,
        reconstruction_loss_weight: float = None,
        unimodal_treshhold: float = 0.95,
        custom_dataloaders: tuple = None,
        augmentation_invariance: bool = False,
        random_state: np.random.RandomState = np.random.RandomState(42),
        logging_active: bool = False,
        autoencoder_param_path: str = "pretrained_ae.pth",
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
        self.projection_axis_optimizer_params = (
            {"lr": 1e-5}
            if projection_axis_optimizer_params is None
            else projection_axis_optimizer_params
        )
        self.pretrain_epochs = pretrain_epochs
        self.max_epochs = max_epochs
        self.grow_interval = grow_interval
        self.use_pvalue = use_pvalue
        self.pruning_threshold = pruning_threshold
        self.optimizer_class = optimizer_class
        self.rec_loss_fn = rec_loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.unimodal_loss_increase_method = unimodal_loss_increase_method
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.max_leaf_nodes = max_leaf_nodes
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.unimodal_treshhold = unimodal_treshhold
        self.random_state = random_state
        self.autoencoder_param_path = autoencoder_param_path
        self.logging_active = logging_active

    def fit_predict(self, X: np.ndarray) -> "DipECT":
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            The given data set as a 2D-array of shape (#samples, #features)

        Returns
        -------
        self : DeepECT
            This instance of the DeepECT algorithm
        """
        augmentation_invariance_check(
            self.augmentation_invariance, self.custom_dataloaders
        )
        tree, autoencoder = _dipect(
            X,
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
            self.unimodal_loss_increase_method,
            self.max_leaf_nodes,
            self.reconstruction_loss_weight,
            self.unimodal_treshhold,
            self.custom_dataloaders,
            self.augmentation_invariance,
            self.random_state,
            self.logging_active,
            self.autoencoder_param_path,
        )
        self.tree_ = tree
        self.autoencoder = autoencoder
        return self
