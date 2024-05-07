import torch
import numpy as np
from sklearn.utils import check_random_state
from sklearn.cluster import KMeans
from clustpy.deep._utils import set_torch_seed
from clustpy.deep._train_utils import get_standard_initial_deep_clustering_setting




class Cluster_Node:
     
    def __init__(self, center: np.ndarray, leaf_node=True):
        if leaf_node:
            self.center = torch.nn.Parameter(torch.tensor(center), requires_grad=True)
        else: 
            self.center = torch.tensor(center)

        self.left_child = None
        self.right_child = None
        
    
    def from_leaf_to_inner(self):
        self.center.requires_grad(False)
    
    def set_childs(self, left_child: np.ndarray, right_child: np.ndarray):
        self.from_leaf_to_inner()
        self.left_child = torch.nn.Parameter(torch.tensor(left_child), requires_grad=True)
        self.right_child = torch.nn.Parameter(torch.tensor(right_child), requires_grad=True)



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

        def __init__(self, init_leafnode_centers: np.ndarray, augmentation_invariance: bool = False):
            super().__init__()
            self.augmentation_invariance = augmentation_invariance
            # Create initial cluster tree
            self.cluster_tree = Cluster_Node(torch.empty(), leaf_node=False).set_childs(init_leafnode_centers[0], init_leafnode_centers[1])

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
                device: torch.device, optimizer: torch.optim.Optimizer, 
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
            device : torch.device
                device to be trained on
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
            # for alpha in self.alphas:
            #     for e in range(n_epochs):
            #         for batch in trainloader:
            #             loss = self._loss(batch, alpha, autoencoder, cluster_loss_weight, rec_loss_fn, device)
            #             # Backward pass
            #             optimizer.zero_grad()
            #             loss.backward()
            #             optimizer.step()
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
        clustering_optimizer_params: dict, pretrain_epochs: int, max_iterations: int,
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
    dkm_module = _DeepECT_Module(init_leafnode_centers, augmentation_invariance).to(device)
    # Use DKM optimizer parameters (usually learning rate is reduced by a magnitude of 10)
    optimizer = optimizer_class(list(autoencoder.parameters()) + list(dkm_module.parameters()),
                                **clustering_optimizer_params)
    # DKM Training loop
    dkm_module.fit(autoencoder, trainloader, max_iterations, device, optimizer, rec_loss_fn)
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
                 pretrain_epochs: int = 50, max_iterations: int = 1000,
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


    
    