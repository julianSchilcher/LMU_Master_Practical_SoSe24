import torch


class MiniBatchKMeans:
    def __init__(self, num_clusters, batch_size, max_iterations, device='cpu'):
        """
        MiniBatch K-Means clustering using PyTorch.

        Parameters:
        num_clusters (int): Number of clusters.
        batch_size (int): Size of each mini-batch.
        max_iterations (int): Maximum number of iterations.
        device (str): Device to run the computations on ('cpu' or 'cuda').
        """
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.device = device
        self.cluster_centers_ = None

    def fit(self, dataset):
        """
        Compute the minibatch k-means clustering.

        Parameters:
        dataset (torch.Tensor): The dataset, a tensor of shape (num_samples, num_features).
        """

        if dataset.size(0) == 0:  # Check if the dataset is empty
            raise ValueError("The dataset cannot be empty.")
        dataset = dataset.to(self.device)
        num_samples, _ = dataset.shape

        # Initialize cluster centers randomly from the dataset
        initial_indices = torch.randperm(num_samples)[:self.num_clusters].to(self.device)
        self.cluster_centers_ = dataset[initial_indices]

        # Initialize counts for center update frequency
        update_counts = torch.zeros(self.num_clusters, device=self.device)

        # Main loop over the specified number of iterations
        for _ in range(self.max_iterations):
            # Randomly select indices for the mini-batch
            minibatch_indices = torch.randperm(num_samples)[:self.batch_size].to(self.device)
            minibatch = dataset[minibatch_indices]
            # Calculate distances and assign each point to the closest cluster
            distances = torch.cdist(minibatch, self.cluster_centers_)
            closest_centers_indices = torch.argmin(distances, dim=1)

            # Update cluster centers
            for idx in range(self.num_clusters):
                points_in_cluster = (closest_centers_indices == idx)
                if points_in_cluster.any():
                    selected_points = minibatch[points_in_cluster]
                    update_counts[idx] += selected_points.shape[0]
                    learning_rate = 1 / update_counts[idx]
                    self.cluster_centers_[idx] *= (1 - learning_rate)
                    self.cluster_centers_[idx] += learning_rate * selected_points.mean(dim=0)

    def predict(self, dataset):
        """
        Predict the closest cluster each sample in dataset belongs to.

        Parameters:
        dataset (torch.Tensor): New data to predict.

        Returns:
        labels (torch.Tensor): Index of the cluster each sample belongs to.
        """
        if dataset.size(0) == 0:
            raise ValueError("The dataset cannot be empty.")
        dataset = dataset.to(self.device)
        distances = torch.cdist(dataset, self.cluster_centers_)
        return torch.argmin(distances, dim=1)