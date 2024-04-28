

from practical.ProcessMining.k_means_implementation import KMeans
import numpy as np
import unittest

from sklearn.datasets import make_blobs

class TestKMeans(unittest.TestCase):
    def test_kmeans(self):
        # Create synthetic data
        np.random.seed(2)
        X = np.random.rand(100, 2)
        
        # Fit KMeans model
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        
        # Check if centroids are initialized
        self.assertEqual(kmeans.centroids.shape[0], 3)
        self.assertEqual(kmeans.centroids.shape[1], 2)
        
        # Check if labels are assigned correctly
        labels = kmeans._assign_labels(X)
        self.assertEqual(labels.shape, (100,))
        
        # Check if labels are within expected range
        self.assertTrue((labels >= 0).all() and (labels < 3).all())
        
        # Check if centroids are updated correctly
        self.assertAlmostEqual(np.sum(kmeans.centroids - kmeans.centroids_history[-1]), 0, places=5)
        
        # Check if convergence is reached
        self.assertTrue(len(kmeans.centroids_history) < kmeans.max_iter)


  
if __name__ == '__main__':

   


    # generate data with sklearn using make_blobs 
    X,y = make_blobs(n_samples=1000, centers=3, n_features=2,
                    random_state=2)
    
    
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    kmeans.plot_clusters(X)
    #kmeans.plot_clusters_iteratively(X)
    unittest.main()
