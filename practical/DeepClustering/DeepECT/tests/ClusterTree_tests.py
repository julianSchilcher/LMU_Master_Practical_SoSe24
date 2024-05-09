from practical.DeepClustering.DeepECT.deepect import Cluster_Node, _DeepECT_Module
import numpy as np
import torch

def test_Cluster_Node():
    root = Cluster_Node(np.array([0,0]), 'cpu')
    root.set_childs(np.array([-1,-1]), np.array([1,1]))
    root.left_child.set_childs(np.array([-2,-2]), np.array([-0.5,-0.5]))

    assert torch.all(torch.eq(root.center, torch.tensor([0,0])))
    assert torch.all(torch.eq(root.left_child.center, torch.tensor([-1,-1])))
    assert torch.all(torch.eq(root.left_child.right_child.center, torch.tensor([-0.5,-0.5])))
    assert root.left_child.weight != None
    assert not root.left_child.is_leaf_node()
    assert root.right_child.is_leaf_node()
    assert root.left_child.right_child.is_leaf_node()
    assert isinstance(root.left_child.right_child.center, torch.nn.Parameter)
    assert isinstance(root.left_child.center, torch.Tensor)

def test_cluster_tree():
    deep_ect = _DeepECT_Module(np.array([[0,0], [1,1]]), 'cpu')
    tree = deep_ect.cluster_tree
    tree.root.left_child.set_childs(np.array([-2,-2]), np.array([-0.5,-0.5]))
    assert len(tree.get_all_leaf_nodes()) == 3

    leaf_nodes = tree.get_all_leaf_nodes()
    assert torch.all(leaf_nodes[0].center == torch.nn.Parameter(torch.tensor([-2,-2], dtype=torch.float16))).item() and torch.all(leaf_nodes[1].center == torch.nn.Parameter(torch.tensor([-0.5,-0.5], dtype=torch.float16))).item() and torch.all(leaf_nodes[2].center ==  torch.nn.Parameter(torch.tensor([1,1], dtype=torch.float16))).item()

def test_cluster_tree_assignment():
    deep_ect = _DeepECT_Module(np.array([[0,0], [1,1]]), 'cpu')
    tree = deep_ect.cluster_tree
    tree.root.left_child.set_childs(np.array([-2,-2]), np.array([-0.5,-0.5]))
    tree.assign_to_nodes(torch.tensor([[-3,-3], [10,10], [-0.4,-0.4],[0.4,0.3]]))

    assert torch.all(torch.eq(tree.root.left_child.left_child.assignments, torch.tensor([[-3,-3]])))
    assert torch.all(torch.eq(tree.root.left_child.right_child.assignments, torch.tensor([[-0.4,-0.4]])))
    assert torch.all(torch.eq(tree.root.right_child.assignments, torch.tensor([[10,10], [0.4,0.3]])))
    assert torch.all(torch.eq(tree.root.assignments, torch.tensor([[-3,-3],[-0.4,-0.4], [10,10], [0.4,0.3]])))
    assert torch.all(torch.eq(tree.root.left_child.assignments, torch.tensor([[-3,-3],[-0.4,-0.4]])))