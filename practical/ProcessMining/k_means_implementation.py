import numpy as np
import matplotlib.pyplot as plt

#from practical.ProcessMining.k_means_implementation import KMeans

class KMeans:

    
    def __init__(self, n_clusters, max_iter=100):
         # initialization of the amount of clusters and the maximum amount of iterations
        self.n_clusters = n_clusters
        self.max_iter = max_iter

        #to show each step of k-means, keep track of the centroids history and their current assigned labels
        self.centroids_history = []
        self.labels_history = []
        
    def fit(self, X):

        # choose n_clusters amount of initial centroids, replace = False so you dont have the same initial centroids twice
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

       # print(self.centroids)
        # keep track of the current centroid
        self.centroids_history.append(self.centroids)
        
        for _ in range(self.max_iter):
            # Assign each of the data points to the nearest centroid
            labels = self._assign_labels(X)

            # keep track of label history
            self.labels_history.append(labels)
            
            # Update centroids based on the mean of the points assigned to each cluster
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            self.centroids_history.append(new_centroids)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
              #  print("convergence reached")
                break
            
            self.centroids = new_centroids
        
        return self
    
    def _assign_labels(self, X):
        # distance calculation
        # Step 1: Add a new axis to centroids
        centroids_expanded = self.centroids[:, np.newaxis]  # Shape: (n_clusters, 1, n_features)
      #  print(centroids_expanded)
        # Step 2: Calculate squared differences
        squared_diff = (X - centroids_expanded) ** 2  # Shape: (n_clusters, n_samples, n_features)
      #  print(squared_diff)
        
        # Step 3: Sum along feature axis
        sum_squared_diff = squared_diff.sum(axis=2)  # Shape: (n_clusters, n_samples)
        
        # Step 4: Calculate Euclidean distances
        distances = np.sqrt(sum_squared_diff)  # Shape: (n_clusters, n_samples)
        
        # Step 5: Potentially assign labels based on minimum distance
       # labels = np.argmin(distances, axis=0)  # Shape: (n_samples,)
        # determine the closest centroid for each data point
       # print(distances)
        return np.argmin(distances, axis=0)
    

#plotting of the final clustering result
    def plot_clusters(self, X):
            labels = self._assign_labels(X)

            plt.figure(figsize=(8, 6))

            for k in range(self.n_clusters):
                plt.scatter(X[labels == k][:, 0], X[labels == k][:, 1], label=f'Cluster {k}')

            plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='x', label='Centroids')
            plt.title('K-Means Clustering Result')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.show()

#plotting of each centroid update step of kmeans
    def plot_clusters_iteratively(self, X):
        num_iterations = len(self.centroids_history)


        num_rows = num_iterations // 2 + num_iterations % 2
        plt.figure(figsize=(10, 10))
        for i, (centroids, labels) in enumerate(zip(self.centroids_history, self.labels_history)):
            plt.subplot(num_rows, 2, i + 1)

            for k in range(self.n_clusters):
                plt.scatter(X[labels == k][:, 0], X[labels == k][:, 1], label=f'Cluster {k}')

            plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='x', label='Centroids')
            plt.title(f'Iteration {i}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
        plt.tight_layout()
        plt.show()



    

       
        
        
      