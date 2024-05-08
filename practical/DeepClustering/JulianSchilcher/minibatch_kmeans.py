import numpy as np
import torch
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt


class MinibatchKmeans:
    """
    Implementation of minibatch-kmeans for a given number of clusters. If <pytorch_optimization> is set to True, it will perform
    the optimization with Pytorch. 
    """

    def __init__(self, k:int, batch_size: int, max_iterations=500, pytorch_optimization=True):
        """

        Args:
            k                    (int): number of clusters
            batch_size           (int): number of samples per minibatch
            max_iteration        (int): maximal number of iterations during optimization
            pytorch_optimization (bool): decides whether the optimization should be performed through pytorch 

        """
        if not isinstance(k, int):
            raise TypeError("k should be of type integer")
        if not isinstance(batch_size, int):
            raise TypeError("batch_size should be of type integer")
        if not isinstance(max_iterations, int):
            raise TypeError("max_iterations should be of type integer")
        if k < 1:
            raise ValueError("Number of classes must be bigger than 0!")
        if batch_size < 1:
            raise ValueError("Choose a batch size bigger than 0!")
        if max_iterations < 1:
            raise ValueError("Choose a maximum number of iterations bigger than 0!")
        
        self.k = k # number of clusters
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.pytorch_optimization = pytorch_optimization
        self.iter = 1 # can be used for analysis
        self.X = None # data matrix
        self.centers = None # representatives of the cluster
        self.init_centers = None # can be used for analysis

    def _get_minibatch(self, batch_size: int) -> torch.Tensor:
        """
        Returns a minibatch of the stored data matrix X

        Args:
            batch_size  (int): the number of samples in the minibatch

        Returns:
                      (Tensor): the minibatch of given size
        """
        if not isinstance(batch_size, int):
            raise TypeError("batch_size should be of type integer")
        if batch_size < 1:
            raise ValueError("Choose a batch size bigger than 0!")
        if X is None:
            raise ValueError("data matrix X is None. Make sure you call fit() before")
        indices = torch.randperm(len(self.X))[:batch_size]
        return self.X[indices].detach().clone()

    def _get_assignments(self,M: torch.Tensor) -> torch.Tensor:
        """
        Computes the assignments of the samples in the given data matrix M (<number of samples> x <number of features>)
        to the clusters represented by the cluster centers.

        Args:
            M   (Tensor): data tensor of dimension (<number of samples> x <number of features>)

        Returns:
                (Tensor): predicted class labels as a one dimensional tensor
        """
        if M is None:
            raise ValueError("Given data matrix cannot be None")
        if self.centers is None:

            raise ValueError("No centers available. Make sure you call fit() first")
        if not isinstance(M, torch.Tensor):
            raise TypeError("M must be a pytorch Tensor")
        
        with torch.no_grad():
            distance_matrix = torch.cdist(M,self.centers,p=2) # kmeans uses L_2 norm (euclidean distance)
            distance_matrix = distance_matrix.squeeze()
            if distance_matrix.ndim == 1: # catch case if batch size is 1
                assignments = torch.argmin(distance_matrix)
            else:
                assignments = torch.argmin(distance_matrix, dim=1)
        return assignments

    def _split_tensor(self, M: torch.Tensor, assignment: torch.Tensor) -> list:
        """
        Takes a minibatch M of size (number of samples x number of features) and the corresponing
        class labels for each sample as a 1 dimensional tensor of shape (n,). The function returns
        a list of tensors containing the samples for each class. List element i contains the Tensor 
        for class label i (or None, if no sample from this class exists). 
        (Cannot return a 3 dimensional tensors instead since the number of samples per class is not constant)
        
        Args:
            M              (Tensor): The data array of shape (<number of samples> x <number of features>).
            assignmet      (Tensor): Indicates the class label for each sample in M, shape (<number of samples>,)
            
        Returns:
                             (list): list of data tensors (number of samples x number of features) containing the samples for each class.
        """
        # Initialize a list to store tensors for each class
        output_tensors = []

        # Iterate over each class and retrieve samples belonging to this class
        for i in range(self.k):
            class_indices = (assignment == i).nonzero()
            if len(class_indices) < 1:
                class_data = None
            else:
                class_data = M[class_indices.squeeze()]
                if class_data.ndim == 1:
                    class_data = class_data[None]
            output_tensors.append(class_data)
        return output_tensors
    
    def _loss(self, M: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for a single class represented through the given center with its samples contained in M.
        
        Args:
            M              (Tensor): The data array of shape (<number of samples> x <number of features>).
            center         (Tensor): The center of the class (<number of features>, )
            
        Returns:
                           (Tensor): The tensor containing the loss for this class
        """
        assert M.shape[1] == len(center)
        return torch.sum((M - center)**2)

    def fit(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the Minibatch-Kmeans clustering on the given dataset X. 

        Args:
            X              (array): The data array of shape (<number of samples> x <number of features>).

        Returns:
            out            (array): The predicted class lables for each sample as a 1 dimensional numpy array
        """
        if X is None:
            raise ValueError("Given data matrix X is None")
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise TypeError("Data matrix X should be a 2 dimensional numpy array")
    
        self.X = torch.from_numpy(X)
        # initialise centers with k randomly picked samples from X
        self.centers = self._get_minibatch(self.k) 
        if self.pytorch_optimization:
            self.centers.requires_grad_(True)
            self.centers.retain_grad() # centers are leaf node in autograd-graph
            optimizer = torch.optim.Adam([self.centers], lr=0.1)
        self.init_centers = self.centers.detach().clone() # store initial centers seperatly for later analysis
        
        # initialise center counts used for adopting learnig rate
        center_counts = torch.zeros(self.k, dtype=torch.int)

        self.iter = 0
        # store centers of last iteration to check if centers are stable
        old_centers = torch.zeros(self.centers.shape, dtype=self.centers.dtype)
        stop_iteration = False

        while not stop_iteration and self.iter < self.max_iterations:

            old_centers = self.centers.detach().clone()
              
            M = self._get_minibatch(self.batch_size)
            # assign each sample in minibatch to its nearest center
            assignment = self._get_assignments(M)
            
            if self.pytorch_optimization:
                # get tensor of samples for each class
                tensor_list = self._split_tensor(M, assignment)
                optimizer.zero_grad() # reset gradients
                for label in range(self.k):
                    class_tensor = tensor_list[label]
                    if class_tensor == None:
                        continue
                    # calculate loss of whole minibatch
                    loss = self._loss(class_tensor, self.centers[label])
                    # calculate gradient of current center 
                    # (gradients of other centers are 0 but will become nonzero in following iterations of this for-loop)
                    # (gradients sum up trough backward(), so after the for loop each center representing a non empty class has a gradient)
                    loss.backward()
                # update the centers, learning rate decays linearly in "time"
                optimizer.param_groups[0]['lr'] = 0.8/(0.1*self.iter + 1)
                optimizer.step() 
            else:
                # optimization based on pseudo code

                # iteration over each sample is necessary in order to be able to adopt learning rate as stated in paper
                for i, sample in enumerate(M):
                    # label of current sample 
                    label = assignment[i]
                    # adopt learning rate
                    center_counts[label] += 1
                    learning_rate = 1/center_counts[label]  
                    # make gradient step
                    self.centers[label] = self.centers[label] + learning_rate*(sample - self.centers[label])
            self.iter += 1
            with torch.no_grad():
                stop_iteration = torch.allclose(self.centers, old_centers, rtol=0.01)
        return self._get_assignments(self.X).numpy()

# Test the implementation
k = 4
X, y_true = make_blobs(n_samples=300, centers=k, cluster_std=0.40, random_state=0)
kmeans = MinibatchKmeans(k, 50, pytorch_optimization=True)
labels_pred = kmeans.fit(X)
plt.scatter(X[:,0],X[:,1], c=labels_pred)
plt.scatter(kmeans.centers.detach().clone()[:,0], kmeans.centers.detach().clone()[:,1], marker="X", c="red", label="Final centers")
plt.scatter(kmeans.init_centers[:,0], kmeans.init_centers[:,1], c="darkgrey", label="Initial centers")
plt.legend()
plt.title(f"Minibatch-kmeans: {kmeans.iter} iterations needed, NMI={normalized_mutual_info_score(y_true, labels_pred)}")

