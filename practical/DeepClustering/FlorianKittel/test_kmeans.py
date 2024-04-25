import unittest

import torch
from kmeans import MiniBatchKMeans
from scipy.spatial import ConvexHull
from utils import create_dataset


class TestKMeans(unittest.TestCase):

    def test_has_k_clusters(self):
        """Test wether functions returns k cluster centers"""
        k = 4
        batch_size = 100
        iterations = 10

        ds = create_dataset()

        model = MiniBatchKMeans(k=k, batch_size=batch_size, iterations=iterations)
        cluster_center = model.fit(ds)

        self.assertEqual(cluster_center.shape[0], k)

    def test_dimension_of_clusters(self):
        """Test wether function returns dim dimensional cluster centers"""
        k = 4
        batch_size = 100
        iterations = 10
        dim = 4

        ds = create_dataset(dimension=dim)
        model = MiniBatchKMeans(k=k, batch_size=batch_size, iterations=iterations)
        cluster_center = model.fit(ds)

        self.assertEqual(cluster_center.shape[1], dim)

    def test_different_centers(self):
        """Test wether function returns different cluster centers"""
        ds = create_dataset()
        k = 4
        batch_size = 100
        iterations = 10

        model = MiniBatchKMeans(k=k, batch_size=batch_size, iterations=iterations)
        cluster_center = model.fit(ds)

        self.assertFalse(torch.equal(cluster_center[0], cluster_center[1]))

    def test_center_in_convex_hull(self):
        """Test wether cluster centers are in convex hull of data set"""

        ds = create_dataset()
        k = 4
        batch_size = 100
        iterations = 10

        model = MiniBatchKMeans(k=k, batch_size=batch_size, iterations=iterations)
        cluster_center = model.fit(ds)

        ds_hull = ConvexHull(ds)
        ds_hull_vertices = torch.from_numpy(ds_hull.vertices)

        ds_cluster_hull = ConvexHull(torch.concat([ds, cluster_center]))
        ds_cluster_hull_vertices = torch.from_numpy(ds_cluster_hull.vertices)

        self.assertTrue(torch.equal(ds_cluster_hull_vertices, ds_hull_vertices))

    def test_k_lower_1(self):
        """Test wether function throws an error, if k is lower than 1"""
        with self.assertRaises(ValueError) as context:
            k = 0
            batch_size = 100
            iterations = 10
            _ = MiniBatchKMeans(k=k, batch_size=batch_size, iterations=iterations)

        self.assertEqual(str(context.exception), "k must be greater than 0")

    def test_batch_size_lower_1(self):
        """Test wether function throws an error, if batch_size is lower than 1"""
        with self.assertRaises(ValueError) as context:
            k = 2
            batch_size = 0
            iterations = 10
            _ = MiniBatchKMeans(k=k, batch_size=batch_size, iterations=iterations)

        self.assertEqual(str(context.exception), "batch_size must be greater than 0")

    def test_iterations_lower_0(self):
        """Test wether function throws an error, if iterations is lower than 0"""
        with self.assertRaises(ValueError) as context:
            k = 2
            batch_size = 100
            iterations = -1
            _ = MiniBatchKMeans(k=k, batch_size=batch_size, iterations=iterations)

        self.assertEqual(
            str(context.exception), "iterations must be greater than or equal to 0"
        )


if __name__ == "__main__":
    unittest.main()




























































