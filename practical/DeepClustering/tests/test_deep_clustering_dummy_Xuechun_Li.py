from practical.DeepClustering.deep_clustering_dummy_Xuechun_Li import MiniBatchKmeans
import torch
import unittest

n_clusters = 3
batch_size = 3
n_iter = 10
tol = 0.001
class unitest_case(unittest.TestCase):
    def setUp(self):
        self.cluster=torch.tensor([
            [1, 2], [1, 3], [2, 3],  # Cluster 1\
            [8, 9], [7, 8], [9, 10],  # Cluster 2
            [0,0],[2,4],[5,7] # Cluster 3
        ], dtype=torch.float32)
        self.n_clusters = 3
        
    def test_fit(self):

        model = MiniBatchKmeans(n_clusters,batch_size,n_iter,tol)
        centers = model.forward(self.cluster)
        # check whether the centers has the same length with the number of clusters 
        # and the range of the centers
        self.assertEqual(len(centers),self.n_clusters)
        self.assertTrue(torch.min(centers) >= 0)
        self.assertTrue(torch.max(centers) <= 10)

    def test_check_convergence(self):
        model = MiniBatchKmeans(n_clusters,batch_size,n_iter,tol)
        model.forward(self.cluster)
        self.assertIsInstance(model._check_convergence(self.cluster), bool, " should return a boolean value.")

if __name__ == "__main__":
    unittest.main()



