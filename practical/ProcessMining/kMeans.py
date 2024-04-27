import numpy as np

class KMeans:
    """A simple clustering method that forms k clusters by iteratively reassigning
    samples to the closest centroids and after that moves the centroids to the center
    of the new formed clusters.


    Parameters:
    -----------
    k: int, optional
        The number of clusters the algorithm will form. Default is 3.
    max_iterations: int, optional
        The number of iterations the algorithm will run for if it does not converge before that.
        Default is 500.
    """
    def __init__(self, k: int = 3, max_iterations: int = 500):
        self.K = k
        self.max_iterations = max_iterations

    def run(self, X: np.ndarray) -> np.ndarray:
        """Runs the K-means algorithm and computes the final clusters.

        Parameters:
        -----------
        X: np.ndarray
            The input data matrix of shape (n_samples, n_features), where
            n_samples is the number of samples and n_features is the number
            of features.

        Returns:
        --------
        np.ndarray
            An array containing the indices of the clusters to which each sample belongs.
        """
        # Initialize centroids as k random samples from X
        centroids: np.ndarray = self._initialize_random_centroids(X)
        # Initialize clusters as one cluster for all samples
        clusters = np.zeros(np.shape(X)[0], dtype=int)
        # Iterate until convergence or for max iterations
        for _ in range(self.max_iterations):
            # Assign samples to the closest centroids (create clusters)
            clusters: np.ndarray = self._create_clusters(centroids, X)
            # Save current centroids for convergence check
            previous_centroids: np.ndarray = centroids
            # Calculate new centroids from the clusters
            centroids = self._compute_means(clusters, X)
            # If no centroids have changed => convergence
            diff: np.ndarray = previous_centroids - centroids
            if not diff.any():
                return clusters
        return clusters

    def _initialize_random_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initializes and returns k random centroids.

        Parameters:
        -----------
        X: np.ndarray
            The input data matrix of shape (n_samples, n_features), where
            n_samples is the number of samples and n_features is the number
            of features.

        Returns:
        --------
        np.ndarray
            An array containing the initial centroids.
        """
        m, n = np.shape(X)
        centroids: np.ndarray = np.empty((self.K, n))
        for i in range(self.K):
            centroids[i] = X[np.random.choice(range(m))]
        return centroids

    def _create_clusters(self, centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Returns an array of cluster indices for all the data samples.

        Parameters:
        -----------
        centroids: np.ndarray
            An array containing the centroids of the clusters.
        X: np.ndarray
            The input data matrix of shape (n_samples, n_features), where
            n_samples is the number of samples and n_features is the number
            of features.

        Returns:
        --------
        np.ndarray
            An array containing the indices of the clusters to which each sample belongs.
        """
        m, _ = np.shape(X)
        cluster_idx = np.empty(m, dtype=int)
        for i in range(m):
            cluster_idx[i] = self._closest_centroid(X[i], centroids)
        return cluster_idx

    def _compute_means(self, cluster_idx: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Computes and returns the updated centroids of the clusters.

        Parameters:
        -----------
        cluster_idx: np.ndarray
            An array containing the indices of the clusters to which each sample belongs.
        X: np.ndarray
            The input data matrix of shape (n_samples, n_features), where
            n_samples is the number of samples and n_features is the number
            of features.

        Returns:
        --------
        np.ndarray
            An array containing the updated centroids of the clusters.
        """
        _, n = np.shape(X)
        centroids: np.ndarray = np.empty((self.K, n))
        for i in range(self.K):
            points: np.ndarray = X[cluster_idx == i]
            centroids[i] = np.mean(points, axis=0)
        return centroids

    def _closest_centroid(self, x: np.ndarray, centroids: np.ndarray) -> int:
        """Finds and returns the index of the closest centroid for a given vector x.

        Parameters:
        -----------
        x: np.ndarray
            The data point for which the closest centroid needs to be found.
        centroids: np.ndarray
            An array containing the centroids of the clusters.

        Returns:
        --------
        int
            The index of the closest centroid to the given data point.
        """
        distances: np.ndarray = np.empty(self.K)
        for i in range(self.K):
            distances[i] = self._euclidean_distance(centroids[i], x)
        return np.argmin(distances)

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Calculates and returns the Euclidean distance between two vectors x1 and x2.

        Parameters:
        -----------
        x1: np.ndarray
            The first vector.
        x2: np.ndarray
            The second vector.

        Returns:
        --------
        float
            The Euclidean distance between the two input vectors.
        """
        return np.sqrt(np.sum(np.power(x1 - x2, 2)))
