from sklearn.base import BaseEstimator
import numpy as np
from typing import Union
import torch

class MiniBatchKMeans(BaseEstimator):
    # Naming based on scikit-learn implementation: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
    def __init__(self, n_clusters: int=3, max_iter: int=1000, batch_size: int=1024, random_state: int=0, init: str='kmeans++') -> None:
        """MiniBatch k-Means Algorithm.
        

        Parameters
        ----------
        n_clusters: int, default=3
            The number of clusters to form as well as the number of centroids to generate.
        
        max_iter: int, default=1000
            Nnumber of iterations over the complete dataset.
        
        batch_size: int, default=1024
            Size of mini batches.
        
        random_state: int, default=0
            Determines random number generation for centroid initialization and random batch assignments.
        
        init: str, default='kmeans++'
            Method for initialization:
            
            'kmeans++': Selects inital cluster centroids using sampling based on the weighted distribution probability by contiously selecting the next centroid with the largest probability.
            
            'random': Randomly selects n_clusters points from the dataset as initial centroids.
            
        Attributes
        ----------
        cluster_centers_: torch.Tensor
            Coordinates of cluster centers.
        
        labels_: torch.Tensor
            Labels of each point.
        
        Notes
        -----
        Based on the "Web-Scale K-Means Clustering" by Scully, D. (2010).
        """
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.batch_size=batch_size
        self.random_state = random_state
        self.init_method = init
        self.labels_ = None
        self.cluster_centers_ = None
        
        # Set seeds
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        
        
    def fit(self, X: Union[torch.utils.data.DataLoader, np.ndarray]):
        """Compute the centroids on X by batching the dataset and assign their labels according to the clustering result.

        Parameter
        ---------
        X: Union[torch.utils.data.DataLoader, np.ndarray]
            Dataset for centroid computation and label assignment.
            
        Returns
        -------
        self: object
            Fitted estimator.
        """
        
        self._assert_dataset_type(X)
        
        if isinstance(X, np.ndarray):
            # Transform numpy.array dataset X to torch Object
            X = torch.utils.data.DataLoader(dataset=X, batch_size=self.batch_size, shuffle=True)

        # Assert correct values of torch dataset
        self._assert_torch_data(X)
            
        # Compute initial cluster centers
        if self.init_method == 'kmeans++':
            self.cluster_centers_ = self._kmeanspp_centroids(X)
        elif self.init_method == 'random':
            self.cluster_centers_ = self._random_centroids(X)
        else:
            raise NotImplementedError(f"Unsupported centroid initialization method given: {self.init_method}")
        
        # Apply MiniBatch k-Means algorithm, starting with the initial centroids
        self.cluster_centers_ = self._minibatchkmeans(X, self.cluster_centers_)
        
        # Assign the labels for each point
        self.labels_ = self._assign_labels(X)
        return self
    
    def _random_centroids(self, data: torch.utils.data.DataLoader) -> torch.Tensor:
        """Randomly selects k-amount points from a given dataset as their initial centroids.

        Parameters
        ----------
        data: torch.utils.data.DataLoader,
            The dataset from which the initial centroids are retrieved.
        
        Returns
        -------
        torch.Tensor: Tensor object with the randomly chosen k-amount centroids.
        
        Raises
        ------
        NotImplementedError: For not implemented data structures when retrieving the initial centroids.
        
        Notes
        -----
        Due to simplicity, numpy.random.choice has been selected as the main function for retrieving the random initial centroids.
        With increasing k, the likelihood of choosing a point more than once with torch.randint() increases.
        """        

        if isinstance(data, torch.utils.data.dataloader.DataLoader):
            centers = data.dataset[np.random.choice(len(data.dataset), size=self.n_clusters, replace=False)] 
        elif isinstance(data, np.ndarray):
            centers = data[np.random.choice(len(data), size=self.n_clusters, replace=False)] 
        else:
            raise NotImplementedError(f"Data of type {type(data)} is not implemented/supported.")
        return torch.tensor(centers)
    
    def _kmeanspp_centroids(self, data: torch.utils.data.DataLoader) -> torch.Tensor:
        """k-Means++ centroids computation, which uses a weighted probability method to calculate the optimal inital centroids.
        The main idea is to select initial centroids which are spread as far as possible to each other but remain close to other data points of potential clusters.

        Parameter
        ---------
        data: Union[torch.utils.data.DataLoader, np.ndarray] 
            Dataset used for the centroid calculation.

        Returns
        -------
        torch.Tensor: Tensor object with the computated centroids based on the k-Means++ algorithm.
        
        Notes
        -----
        The used method is based on "k-means++: The Advantages of Careful Seeding", Arthur, D. & Vassilvitskii, S.
        See https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf for more information.
        
        """
        cluster_centers = torch.empty((self.n_clusters, data.dataset.shape[1]), dtype=torch.float64) # Init an empty cluster centroid data struct as Tensor object
        for i in range(0, self.n_clusters):
            if i == 0:
                # The initial first centroid is uniformly at random selected.
                #cluster_centers[0] = torch.tensor(data.dataset[np.random.choice(len(data.dataset), 1, replace=False)])
                cluster_centers[0] = torch.tensor(data.dataset[torch.randint(low=0, high=data.dataset.shape[0], size=(1, ))])
        
            else:
                # Calculate all euclidean distances between the current centroids and all points
                distances = torch.cdist(torch.tensor(data.dataset), cluster_centers[:i], p=2.0)
                
                # Only keep the smallest distance values
                distances = torch.min(distances, dim=1).values
                
                # Square all values
                distances = torch.pow(distances, 2)

                # Calculate the weighted probibility values
                p_vals = torch.div(distances, torch.sum(distances, dim=0))
                
                # Add next optimal centroid to the cluster centers list
                cluster_centers[i] = torch.tensor(data.dataset[torch.argmax(p_vals)])
                        
        return cluster_centers
    
    def _minibatchkmeans(self, data: torch.utils.data.DataLoader, cluster_centers: torch.Tensor) -> torch.Tensor:
        """Minibatch k-Means Algorithm.
        Calculates the optimal cluster centers based in small batches and their continous updates.
        
        Parameters
        ----------
        data: torch.utils.data.Dataloader
            Dataset to be clustered
            
        Returns
        ------
        torch.Tensor: Optimal cluster centers after a certain amount of iterations.
        
        Notes
        -----
        For optimization, a check on the average overal minimization value of the distance between batch and current state of centroids has been added.
        If at any point the value increases compared to the previous value, the algorithm stops and returns the previous centroids as final centers.
        """
        
        # Initilize v
        v = np.zeros(self.n_clusters, dtype=np.int64)
        
        for _ in range(self.max_iter):
            batch = next(iter(data)) # Select data points of batch size
            
            # Calculate distances between current cluster centers and batch data and store the index of the closest point
            # torch.cdist returns all distances between each batch point to each cluster center. With p=2.0 the euclidean distance is calculated
            # torch.argmin returns the indexes of the closest center for each batch point
            batch_dist = torch.argmin(torch.cdist(batch, cluster_centers, p=2.0), dim=1)
            
            # Additional Optimization: Stop as soon as the update rule increases the average minimization value
            # Calculating the overall min distance vals for each data point to their respective cluster center
            min_vals = torch.min(torch.cdist(torch.tensor(data.dataset), cluster_centers, p=2.0), dim=1).values
            # And store the centroids with the according min vals
            old_centers = cluster_centers
            
            # Adjusting the cluster center
            for i in np.unique(batch_dist):
                v[i] += len(np.where(batch_dist == i)) # Add the occurences count to the overall count
                eta = 1.0 / v[i] # Calculate eta
                cluster_centers[i] = (1-eta)*cluster_centers[i] + eta*torch.mean(batch[np.where(batch_dist == i)], dim=0) # Adjust centroid
            
            if torch.mean(min_vals) < torch.mean(torch.min(torch.cdist(torch.tensor(data.dataset), cluster_centers, p=2.0), dim=1).values):
                return old_centers
            
                
        return cluster_centers
    
    def _assign_labels(self, data: torch.utils.data.DataLoader) -> torch.Tensor:
        """Calculates the closest (euclidean distance) center for each point and assigns the representing clustering label

        Parameter
        ---------
        data: torch.utils.data.DataLoader
            The full dataset.

        Returns
        -------
        torch.Tensor: The clustering results based on the shortest euclidean distance.
        """
        
        return torch.argmin(torch.cdist(torch.tensor(data.dataset), self.cluster_centers_, p=2.0), dim=1)
    
    def _assert_dataset_type(self, data) -> None:
        """Helper function to assert the given data set is of correct type.

        Parameter
        ---------
        data: object
            Dataset which is checked if its of appropiate type.
        """
        if not (isinstance(data, torch.utils.data.DataLoader) != isinstance(data, np.ndarray)):
            raise NotImplementedError(f"Dataset if wrong type: {type(data)}")
    
    def _assert_torch_data(self, data: torch.utils.data.DataLoader) -> None:
        """Helper function to assert that the necessary DataLoader values correspond to the class' settings.

        Parameter
        ---------
        data: torch.utils.data.DataLoader
            Dataset in the necessary DataLoader object
            
        Raises
        ------
        ValueError: Whenever a certain value is not corresponding to the class' settings when initalized.
        """
        
        if data.batch_size != self.batch_size:
            raise ValueError(f"The datasets batch size of {data.batch_size}")