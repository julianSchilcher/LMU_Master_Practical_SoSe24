import numpy as np

class Kmeans:
    def __init__(self, k=3, iterations=50):
        self.k = k
        self.iterations = iterations

    def kmeans(self, data):
        # Chose random centroids for k clusters
        self.centroid_indices = np.random.choice(data.shape[0], self.k, replace=False)
        self.centroids = data[self.centroid_indices]

        for _ in range(self.iterations):
            # Find nearest centroid for each data point
            distances = np.linalg.norm(data[:, None] - self.centroids, axis=2) # Calculate distance of each point to each centroid
            self.cluster_indices = np.argmin(distances, axis=1) # Get index of nearest centroid for each point
            new_centroids = np.array([data[self.cluster_indices == i].mean(axis=0) for i in range(self.k)])
            
            if np.all(self.centroids == new_centroids):
                break
            self.centroids = new_centroids