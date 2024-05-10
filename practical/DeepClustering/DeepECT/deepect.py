import torch
import numpy as np
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from clustpy.deep._utils import set_torch_seed
from clustpy.deep._train_utils import get_standard_initial_deep_clustering_setting
from typing import List, Tuple


class Cluster_Node:
     
    def __init__(self, center: np.ndarray, device: torch.device):
        self.device = device
        self.left_child = None
        self.right_child = None
        self.weight = None
        self.center = torch.nn.Parameter(torch.tensor(center, requires_grad=True, device= self.device, dtype=torch.float))
        self.assignments = None

    def is_leaf_node(self):
        return self.left_child is None and self.right_child is None
    
    def from_leaf_to_inner(self):
        # inner node on cpu
        self.center = self.center.data.cpu() # retrive data tensor from nn.Parameters
        self.center.requires_grad = False
        self.weight = torch.tensor([1.0, 1.0]) # initialise weights for left and right child
    
    def set_childs(self, left_child: np.ndarray, right_child: np.ndarray):
        self.from_leaf_to_inner()
        self.left_child = Cluster_Node(left_child, self.device)
        self.right_child = Cluster_Node(right_child, self.device)

class Cluster_Tree:

    def __init__(self, init_leafnode_centers: np.ndarray, device: torch.device):
        self.root =  Cluster_Node(np.zeros(init_leafnode_centers.shape[1]), device)
        self.root.set_childs(init_leafnode_centers[0], init_leafnode_centers[1])
        self.pruning_nodes = [] # stores a list of nodes which weights fall below pruning treshold during current iteration and must be pruned in next iteration

    
    def get_all_leaf_nodes(self) -> List[Cluster_Node]:
        leafnodes = []
        self._collect_leafnodes(self.root, leafnodes)
        return leafnodes
        
    def _collect_leafnodes(self, node: Cluster_Node, leafnodes: list):
        if node.is_leaf_node():
            leafnodes.append(node)
        else:
            self._collect_leafnodes(node.left_child, leafnodes)
            self._collect_leafnodes(node.right_child, leafnodes)

    def assign_to_nodes(self, minibatch: torch.tensor):
        leafnodes = self.get_all_leaf_nodes()
        leafnode_centers = list(map(lambda node: node.center.data, leafnodes))
        leafnode_tensor = torch.stack(leafnode_centers, dim=0)

        with torch.no_grad():
            distance_matrix = torch.cdist(minibatch, leafnode_tensor, p=2) # kmeans uses L_2 norm (euclidean distance)
        distance_matrix = distance_matrix.squeeze()
        assignments = torch.argmin(distance_matrix, dim=1)
        
        for i, node in enumerate(leafnodes):
            indices = (assignments == i).nonzero()
            if len(indices) < 1:
                node.assignments = None
            else:
                leafnode_data = minibatch[indices.squeeze()]
                if leafnode_data.ndim == 1:
                    leafnode_data = leafnode_data[None]
            node.assignments = leafnode_data
        self._assign_to_splitnodes(self.root) # assign samples recursively bottom up from leaf nodes to inner nodes
    
    def _assign_to_splitnodes(self, node: Cluster_Node):
        if node.is_leaf_node():
            return node.assignments
        else:
            left_assignments = self._assign_to_splitnodes(node.left_child)
            right_assignments = self._assign_to_splitnodes(node.right_child)
            if left_assignments == None or right_assignments == None:
                node.assignments = left_assignments if right_assignments == None else right_assignments   
            else:
                node.assignments = torch.cat((left_assignments, right_assignments), dim=0)
            return node.assignments
    
    def nc_loss(self, autoencoder: torch.nn.Module) -> torch.tensor:
        leaf_nodes = self.get_all_leaf_nodes()
        # convert the list of leaf nodes to a list of the corresponding leaf node centers as tensors
        leafnode_centers = list(map(lambda node: node.center, leaf_nodes))
        # !!! Maybe here a problem of concatenating parameter tensors !!
        # reformat list of tensors to one sinlge tensor of shape (#leafnodes,#emb_features)
        leafnode_center_tensor = torch.stack(leafnode_centers, dim=0) 
        
        # get the assignments for each leaf node (from the current minibatch)
        leafnode_assignments = list(map(lambda node: node.assignments, leaf_nodes))
        # calculate the center of the assignments from the current minibatch for each leaf node
        with torch.no_grad(): # embedded space should not be optimized in this loss 
            leafnode_minibatch_centers = list(map(lambda assignments: torch.sum(autoencoder.encode(assignments), axis=0)/len(assignments), leafnode_assignments))
        # reformat list of tensors to one sinlge tensor of shape (#leafnodes,#emb_features)
        leafnode_minibatch_centers_tensor = torch.stack(leafnode_minibatch_centers, dim=0)
        
        # calculate the distance between the current leaf node centers and the center of its assigned embeddings averaged over all leaf nodes
        loss = torch.sum((leafnode_center_tensor - leafnode_minibatch_centers_tensor)**2)/len(leafnode_center_tensor)
        return loss

    def dc_loss(self):
        pass
    
    def adapt_inner_nodes(self):
        pass

    def pruning_necessary(self):
        return len(self._nodes_to_prune) != 0

    def prune_tree(self):
        # reset list of nodes to prune
        self._nodes_to_prune = []

    def grow_tree(self):
        pass


class _DeepECT_Module(torch.nn.Module):
        """
        The _DeepECT_Module. Contains most of the algorithm specific procedures like the loss and tree-gow functions.

        
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

        def __init__(self, init_leafnode_centers: np.ndarray, device: torch.device, augmentation_invariance: bool = False):
            super().__init__()
            self.augmentation_invariance = augmentation_invariance
            # Create initial cluster tree
            self.cluster_tree = Cluster_Tree(init_leafnode_centers, device)
            self.device = device

        def deepECT_loss(self, embedded: torch.Tensor, alpha: float) -> torch.Tensor:
            """
            Calculate the DeepECT loss of given embedded samples.

            Parameters
            ----------
            embedded : torch.Tensor
                the embedded samples
    
            Returns
            -------
            loss : torch.Tensor
                the final DeepECT loss
            """
            # squared_diffs = squared_euclidean_distance(embedded, self.centers)
            # probs = _dkm_get_probs(squared_diffs, alpha)
            # loss = (squared_diffs.sqrt() * probs).sum(1).mean()
            loss = None
            return loss
        

        def dkm_augmentation_invariance_loss(self, embedded: torch.Tensor, embedded_aug: torch.Tensor,
                                            alpha: float) -> torch.Tensor:
            """
            Calculate the DeepECT loss of given embedded samples with augmentation invariance.

            Parameters
            ----------
            embedded : torch.Tensor
                the embedded samples
            embedded_aug : torch.Tensor
                the embedded augmented samples
            alpha : float
                the alpha value

            Returns
            -------
            loss : torch.Tensor
                the final DKM loss
            """
            # # Get loss of non-augmented data
            # squared_diffs = squared_euclidean_distance(embedded, self.centers)
            # probs = _dkm_get_probs(squared_diffs, alpha)
            # clean_loss = (squared_diffs.sqrt() * probs).sum(1).mean()
            # # Get loss of augmented data
            # squared_diffs_augmented = squared_euclidean_distance(embedded_aug, self.centers)
            # aug_loss = (squared_diffs_augmented.sqrt() * probs).sum(1).mean()
            # # average losses
            # loss = (clean_loss + aug_loss) / 2
            loss = None
            return loss

        def _loss(self, batch: list, alpha: float, autoencoder: torch.nn.Module, cluster_loss_weight: float,
                rec_loss_fn: torch.nn.modules.loss._Loss, device: torch.device) -> torch.Tensor:
            """
            Calculate the complete DeepECT + Autoencoder loss.

            Parameters
            ----------
            batch : list
                the minibatch
            autoencoder : torch.nn.Module
                the autoencoder
            rec_loss_fn : torch.nn.modules.loss._Loss
                loss function for the reconstruction
            device : torch.device
                device to be trained on

            Returns
            -------
            loss : torch.Tensor
                the final DeepECT + AE loss
            """
            # # Calculate combined total loss
            # if self.augmentation_invariance:
            #     # Calculate reconstruction loss
            #     #batch[1] are augmented samples for training samples in batch[0]
            #     ae_loss, embedded, _ = autoencoder.loss([batch[0], batch[2]], loss_fn, device)
            #     ae_loss_aug, embedded_aug, _ = autoencoder.loss([batch[0], batch[1]], loss_fn, device)
            #     ae_loss = (ae_loss + ae_loss_aug) / 2
            #     # Calculate clustering loss
            #     cluster_loss = self.dkm_augmentation_invariance_loss(embedded, embedded_aug, alpha)
            # else:
            #     # Calculate reconstruction loss
            #     ae_loss, embedded, _ = autoencoder.loss(batch, loss_fn, device)
            #     # Calculate clustering loss
            #     cluster_loss = self.dkm_loss(embedded, alpha)
            # loss = ae_loss + cluster_loss * cluster_loss_weight
            loss = None
            return loss

        def fit(self, autoencoder: torch.nn.Module, trainloader: torch.utils.data.DataLoader, max_iterations: int,
                grow_interval: int, optimizer: torch.optim.Optimizer, 
                rec_loss_fn: torch.nn.modules.loss._Loss) -> '_DeepECT_Module':
            """
            Trains the _DeepECT_Module in place.

            Parameters
            ----------
            autoencoder : torch.nn.Module
                the autoencoder
            trainloader : torch.utils.data.DataLoader
                dataloader to be used for training
            max_iteratins : int
                number of iterations for the clustering procedure.
            optimizer : torch.optim.Optimizer
                the optimizer for training
            rec_loss_fn : torch.nn.modules.loss._Loss
                loss function for the reconstruction
            cluster_loss_weight : float
                weight of the clustering loss compared to the reconstruction loss

            Returns
            -------
            self : _DKM_Module
                this instance of the _DKM_Module
            """

            train_iterator = iter(trainloader)

            for e in range(max_iterations):
                if self.cluster_tree.pruning_necessary():
                    self.cluster_tree.prune_tree(self.pruning_nodes)                    
                if e % grow_interval == 0:
                    self.cluster_tree.grow_tree()
                
                # retrieve minibatch (endless)
                try:
                    # get next minibatch
                    M = next(train_iterator)
                except StopIteration:
                    # after full epoch shuffle again
                    train_iterator = iter(trainloader)
                    M = next(train_iterator)

                # assign data points to leafnodes and splitnodes 
                self.cluster_tree.assign_to_nodes(M)

                # calculate loss
                nc_loss = self.cluster_tree.nc_loss(autoencoder)
                dc_loss = self.cluster_tree.dc_loss()
                rec_loss, embedded, reconstructed = autoencoder.loss(M, rec_loss_fn, self.device)
                
                loss = nc_loss + dc_loss + rec_loss

                # optimizer_step with gradient descent
                # !!! make sure that optimizer contains autoencoder-params and all new leaf node centers (old leaf node centers must be deleted in optimizer params) 
                # ==> maybe new optimizer object after each tree grow step

                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                # adapt centers of split nodes analytically
                self.cluster_tree.adapt_inner_nodes()
            return self
        
        
        def predict(self, embedded: torch.Tensor, alpha: float = 1000) -> torch.Tensor:
            # """
            # Prediction of given embedded samples. Returns the corresponding soft labels.

            # Parameters
            # ----------
            # embedded : torch.Tensor
            #     the embedded samples
            # alpha : float
            #     the alpha value (default: 1000)

            # Returns
            # -------
            # pred : torch.Tensor
            #     The predicted soft labels
            # """
            # squared_diffs = squared_euclidean_distance(embedded, self.centers)
            # pred = _dkm_get_probs(squared_diffs, alpha)
            # return pred
            pass
    
def _deep_ect(X: np.ndarray,  batch_size: int, pretrain_optimizer_params: dict,
        clustering_optimizer_params: dict, pretrain_epochs: int, max_iterations: int, grow_interval: int,
        optimizer_class: torch.optim.Optimizer, rec_loss_fn: torch.nn.modules.loss._Loss, autoencoder: torch.nn.Module,
        embedding_size: int, custom_dataloaders: tuple, augmentation_invariance: bool,
        random_state: np.random.RandomState) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module):
    """
    Start the actual DeepECT clustering procedure on the input data set.

    Parameters
    ----------
    X : np.ndarray / torch.Tensor
        the given data set. Can be a np.ndarray or a torch.Tensor
    batch_size : int
        size of the data batches
    pretrain_optimizer_params : dict
        parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate
    clustering_optimizer_params : dict
        parameters of the optimizer for the actual clustering procedure, includes the learning rate
    pretrain_epochs : int
        number of epochs for the pretraining of the autoencoder
    max_iterations : int
        number of iterations for the actual clustering procedure.
    optimizer_class : torch.optim.Optimizer
        the optimizer
    rec_loss_fn : torch.nn.modules.loss._Loss
        loss function for the reconstruction
    autoencoder : torch.nn.Module
        the input autoencoder. If None a new FeedforwardAutoencoder will be created
    embedding_size : int
        size of the embedding within the autoencoder
    custom_dataloaders : tuple
        tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
        If None, the default dataloaders will be used
    augmentation_invariance : bool
        If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
        cluster assignments that are invariant to the augmentation transformations
    random_state : np.random.RandomState
        use a fixed random state to get a repeatable solution

    Returns
    -------
    tuple : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, torch.nn.Module)
        The labels as identified by a final KMeans execution,
        The cluster centers as identified by a final KMeans execution,
        The labels as identified by DKM after the training terminated,
        The cluster centers as identified by DKM after the training terminated,
        The final autoencoder
    """
    # Get initial setting (device, dataloaders, pretrained AE and initial clustering result)
    #TODO: Set clus
    device, trainloader, testloader, autoencoder, _, n_clusters, _, init_leafnode_centers, _ = get_standard_initial_deep_clustering_setting(
        X, 2, batch_size, pretrain_optimizer_params, pretrain_epochs, optimizer_class, rec_loss_fn, autoencoder,
        embedding_size, custom_dataloaders, KMeans, None, random_state)
    # Setup DKM Module
    dkm_module = _DeepECT_Module(init_leafnode_centers, device, augmentation_invariance).to(device)
    # Use DKM optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()),
                                **clustering_optimizer_params)
    # DKM Training loop
    dkm_module.fit(autoencoder, trainloader, max_iterations, grow_interval, optimizer, rec_loss_fn)
    # Get labels
    dkm_labels = predict_batchwise(testloader, autoencoder, dkm_module, device)
    dkm_centers = dkm_module.centers.detach().cpu().numpy()
    # Do reclustering with Kmeans
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(embedded_data)
    return kmeans.labels_, kmeans.cluster_centers_, dkm_labels, dkm_centers, autoencoder


class DeepECT:

    def __init__(self, batch_size: int = 256, pretrain_optimizer_params: dict = None, clustering_optimizer_params: dict = None,
                 pretrain_epochs: int = 50, max_iterations: int = 10000, grow_interval: int = 500,
                 optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
                 rec_loss_fn: torch.nn.modules.loss._Loss = torch.nn.MSELoss(), autoencoder: torch.nn.Module = None,
                 embedding_size: int = 10, custom_dataloaders: tuple = None,
                 augmentation_invariance: bool = False, random_state: np.random.RandomState = None):
        """
        The Deep Embedded Cluster Tree (DeepECT) algorithm.
        First, an autoencoder (AE) will be trained (will be skipped if input autoencoder is given).
        Afterward, a cluter tree will be grown and the AE will be optimized using the DeepECT loss function.

        Parameters
        ----------
        batch_size : int
            size of the data batches (default: 256)
        pretrain_optimizer_params : dict
            parameters of the optimizer for the pretraining of the autoencoder, includes the learning rate (default: {"lr": 1e-3})
        clustering_optimizer_params : dict
            parameters of the optimizer for the actual clustering procedure, includes the learning rate (default: {"lr": 1e-4})
        pretrain_epochs : int
            number of epochs for the pretraining of the autoencoder (default: 50)
        max_iterations : int
            number of iteratins for the actual clustering procedure (default: 1000)
        optimizer_class : torch.optim.Optimizer
            the optimizer class (default: torch.optim.Adam)
        rec_loss_fn : torch.nn.modules.loss._Loss
            loss function for the reconstruction (default: torch.nn.MSELoss())
        autoencoder : torch.nn.Module
            the input autoencoder. If None a new FeedforwardAutoencoder will be created (default: None)
        embedding_size : int
            size of the embedding within the autoencoder (default: 10)
        custom_dataloaders : tuple
            tuple consisting of a trainloader (random order) at the first and a test loader (non-random order) at the second position.
            If None, the default dataloaders will be used (default: None)
        augmentation_invariance : bool
            If True, augmented samples provided in custom_dataloaders[0] will be used to learn 
            cluster assignments that are invariant to the augmentation transformations (default: False)
        random_state : np.random.RandomState
            use a fixed random state to get a repeatable solution. Can also be of type int (default: None)

        Attributes
        ----------
        labels_ : np.ndarray
            The final labels (obtained by a final KMeans execution)
        cluster_centers_ : np.ndarray
            The final cluster centers (obtained by a final KMeans execution)
        dkm_labels_ : np.ndarray
            The final DKM labels
        dkm_cluster_centers_ : np.ndarray
            The final DKM cluster centers
        autoencoder : torch.nn.Module
            The final autoencoder
        """
        self.batch_size = batch_size
        self.pretrain_optimizer_params = {"lr": 1e-3} if pretrain_optimizer_params is None else pretrain_optimizer_params
        self.clustering_optimizer_params = {"lr": 1e-4} if clustering_optimizer_params is None else clustering_optimizer_params
        self.pretrain_epochs = pretrain_epochs
        self.max_iterations = max_iterations
        self.grow_interval = grow_interval
        self.optimizer_class = optimizer_class
        self.rec_loss_fn = rec_loss_fn
        self.autoencoder = autoencoder
        self.embedding_size = embedding_size
        self.custom_dataloaders = custom_dataloaders
        self.augmentation_invariance = augmentation_invariance
        self.random_state = check_random_state(random_state)
        set_torch_seed(self.random_state)

    def fit(self, X: np.ndarray) -> 'DeepECT':
        """
        Initiate the actual clustering process on the input data set.
        The resulting cluster labels will be stored in the labels_ attribute.

        Parameters
        ----------
        X : np.ndarray
            the given data set

        Returns
        -------
        self : DKM
            this instance of the DKM algorithm
        """
        # augmentation_invariance_check(self.augmentation_invariance, self.custom_dataloaders)

        kmeans_labels, kmeans_centers, dkm_labels, dkm_centers, autoencoder = _deep_ect(X,
                                                                                   self.batch_size,
                                                                                   self.pretrain_optimizer_params,
                                                                                   self.clustering_optimizer_params,
                                                                                   self.pretrain_epochs,
                                                                                   self.max_iterations,
                                                                                   self.grow_interval, 
                                                                                   self.optimizer_class, self.rec_loss_fn,
                                                                                   self.autoencoder,
                                                                                   self.embedding_size,
                                                                                   self.custom_dataloaders,
                                                                                   self.augmentation_invariance,
                                                                                   self.random_state)
        # self.labels_ = kmeans_labels
        # self.cluster_centers_ = kmeans_centers
        # self.dkm_labels_ = dkm_labels
        # self.dkm_cluster_centers_ = dkm_centers
        # self.autoencoder = autoencoder
        return self


    
    