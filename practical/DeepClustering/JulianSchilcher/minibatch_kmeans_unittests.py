from minibatch_kmeans import MinibatchKmeans
import unittest
import torch
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score

class TestMinibatchKmeans(unittest.TestCase):
    """
    all test methods start with _test, the other (helper) methods will bes skipped by executing the tests
    """
    def test_get_minibatch(self):
        kmeans = MinibatchKmeans(3, 50, pytorch_optimization=False)
        kmeans.X = torch.rand((5,4))
        # generate a minibatch of size 2 out of X
        mb = kmeans._get_minibatch(batch_size=2)
        self.assertEqual(mb.shape, (2,4))
        self.assertTrue(torch.isin(mb,kmeans.X).all().item())

    def test_get_assignments(self):
        kmeans = MinibatchKmeans(4, 50, pytorch_optimization=False)
        kmeans.X = torch.tensor([[0,-5], [20, 15], [38, 2], [190, -100]], dtype=torch.float)
        kmeans.centers = torch.tensor([[200, -90], [40, 0], [22, 23], [-10, -3]], dtype=torch.float)
        labels = kmeans._get_assignments(kmeans.X)
        # ckeck if assignments are done right
        self.assertTrue(torch.eq(labels, torch.tensor([3,2,1,0])).all().item())

    def test_fit_outputformat(self):
        X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.20, random_state=0)
        kmeans = MinibatchKmeans(4, 50, pytorch_optimization=False)
        y_pred = kmeans.fit(X)
        # check if each sample got a label
        self.assertEqual(len(y_pred), len(X))
        # check if really 4 classes were outputed
        self.assertTrue(np.equal(np.unique(y_pred), np.array([0,1,2,3])).all().item())

    def helper_run_simulation(self, pytorch_optimization):
        X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.40, random_state=0)
        kmeans = MinibatchKmeans(4, 30, pytorch_optimization=pytorch_optimization)
        # mutual information should have high value averaged over multiple runs (depending on quality of random initialisation)
        nmi = 0
        for i in range(50):
            y_pred = kmeans.fit(X)
            nmi += normalized_mutual_info_score(y_true, y_pred)
        if pytorch_optimization:
            print("Performance Pytorch Optim: ", nmi/50)
        else:
            print("Performance Paper Optim: ", nmi/50)
        return nmi/50 > 0.8

    def test_fit_outputsemantics(self):
        self.assertTrue(self.helper_run_simulation(True) and self.helper_run_simulation(False))
    
    
    def test_split_tensor(self):
        kmeans = MinibatchKmeans(4, 50)
        M = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]])
        labels = torch.tensor([0,1,0,3],dtype=torch.int)
        tensor_list = kmeans._split_tensor(M, labels)
        true_tensor_list = [torch.tensor([[1,2,3],[7,8,9]]),torch.tensor([[4,5,6]]), None, torch.tensor([[10,11,12]])]
        # compare list of tensors elementwise (when both elements are None, this should be fine as well)
        cmp_tensor = lambda x,y: torch.equal(x,y) if x != None and y != None else x == y
        self.assertTrue(all(map(cmp_tensor,tensor_list, true_tensor_list)))


if __name__ == '__main__':
    unittest.main()