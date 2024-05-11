from practical.DeepClustering.DeepECT.deepect import Cluster_Node, _DeepECT_Module
import numpy as np
import torch

def test_Cluster_Node():
    root = Cluster_Node(np.array([0,0]), 'cpu')
    root.set_childs(np.array([-1,-1]), np.array([1,1]))
    root.left_child.set_childs(np.array([-2,-2]), np.array([-0.5,-0.5]))

    # check if centers are stored correctly
    assert torch.all(torch.eq(root.center, torch.tensor([0,0])))
    assert torch.all(torch.eq(root.left_child.center, torch.tensor([-1,-1])))
    assert torch.all(torch.eq(root.left_child.right_child.center, torch.tensor([-0.5,-0.5])))
    # since the left node of the root node changed to a inner node, its weights must be non zero
    assert root.left_child.weight != None
    assert not root.left_child.is_leaf_node()
    # right child of root is still a leaf node
    assert root.right_child.is_leaf_node()
    assert root.left_child.right_child.is_leaf_node()
    # centers of leaf nodes must be trainable
    assert isinstance(root.left_child.right_child.center, torch.nn.Parameter)
    # centers of inner nodes are just tensors
    assert isinstance(root.left_child.center, torch.Tensor)

def sample_cluster_tree():
    """ 
    Helper method for creating a sample cluster tree
    """
    deep_ect = _DeepECT_Module(np.array([[0,0], [1,1]]), 'cpu')
    tree = deep_ect.cluster_tree
    tree.root.left_child.set_childs(np.array([-2,-2]), np.array([-0.5,-0.5]))
    return tree

def sample_cluster_tree_with_assignments():
    """ 
    Helper method for creating a sample cluster tree with assignments
    """
    tree = sample_cluster_tree()
    tree.assign_to_nodes(torch.tensor([[-3,-3], [10,10], [-0.4,-0.4],[0.4,0.3]]))
    return tree

def test_cluster_tree():
    tree = sample_cluster_tree()
    # we should have 3 leaf nodes in this example
    assert len(tree.get_all_leaf_nodes()) == 3

    # check if the returned nodes are really the leaf nodes by checking the stored center 
    leaf_nodes = tree.get_all_leaf_nodes()
    assert torch.all(leaf_nodes[0].center == torch.nn.Parameter(torch.tensor([-2,-2], dtype=torch.float16))).item() and torch.all(leaf_nodes[1].center == torch.nn.Parameter(torch.tensor([-0.5,-0.5], dtype=torch.float16))).item() and torch.all(leaf_nodes[2].center ==  torch.nn.Parameter(torch.tensor([1,1], dtype=torch.float16))).item()

def test_cluster_tree_assignment():
    tree = sample_cluster_tree_with_assignments()

    # check if all assignments made correct
    assert torch.all(torch.eq(tree.root.left_child.left_child.assignments, torch.tensor([[-3,-3]])))
    assert torch.all(torch.eq(tree.root.left_child.right_child.assignments, torch.tensor([[-0.4,-0.4]])))
    assert torch.all(torch.eq(tree.root.right_child.assignments, torch.tensor([[10,10], [0.4,0.3]])))
    assert torch.all(torch.eq(tree.root.assignments, torch.tensor([[-3,-3],[-0.4,-0.4], [10,10], [0.4,0.3]])))
    assert torch.all(torch.eq(tree.root.left_child.assignments, torch.tensor([[-3,-3],[-0.4,-0.4]])))

def test_nc_loss():
    tree = sample_cluster_tree_with_assignments()

    # create mock-autoencoder, which represents just an identity function
    encode = lambda x: x
    autoencoder = type('Autoencoder', (), {'encode': encode})
    # calculate nc loss for the above example
    loss = tree.nc_loss(autoencoder)
    
    # calculate nc loss for the above example manually
    loss_left_left_node = torch.sqrt(torch.sum((torch.tensor([-2,-2]) - torch.tensor([-3,-3]))**2))
    loss_left_right_node = torch.sqrt(torch.sum((torch.tensor([-0.5,-0.5]) - torch.tensor([-0.4,-0.4]))**2))
    loss_right_node = torch.sqrt(torch.sum((torch.tensor([1,1]) - (torch.tensor([10,10])+torch.tensor([0.4,0.3]))/2)**2))
    loss_test = (loss_left_left_node + loss_left_right_node + loss_right_node)/3

    assert torch.all(torch.eq(loss, loss_test))

def test_dc_loss():
    tree = sample_cluster_tree_with_assignments()
    
    # calculate direction between left child of root and right child of root
    projection_l_r = (torch.tensor([0,0], dtype=torch.float32) - torch.tensor([1,1], dtype=torch.float32))/np.sqrt(2)
    projection_l_r = projection_l_r[None] # shape to 1x2 for matmul
    # calculate dc-loss for left child of root
    loss_l_r = torch.abs(torch.matmul(projection_l_r, (torch.tensor([0,0], dtype=torch.float32)[None] - torch.tensor([-3,-3], dtype=torch.float32)[None]).T)) + torch.abs(torch.matmul(projection_l_r, (torch.tensor([0,0], dtype=torch.float32)[None] -  torch.tensor([-0.4,-0.4])[None]).T))
    # calculate direction between right child of root and left child of root (just flip vector)
    projection_r_l = -projection_l_r
    # calculate dc-loss for right child of root
    loss_r_l = torch.abs(torch.matmul(projection_r_l, (torch.tensor([1,1], dtype=torch.float32)[None] - torch.tensor([10,10], dtype=torch.float32)[None]).T)) + torch.abs(torch.matmul(projection_r_l, (torch.tensor([1,1], dtype=torch.float32)[None] -  torch.tensor([0.4,0.3])[None]).T))
    # calculate direction between root.left.left and root.left.right
    projection_l_l_r = (torch.tensor([-2,-2], dtype=torch.float32) - torch.tensor([-0.5,-0.5]))/np.sqrt((-1.5)**2 + (-1.5)**2)
    # calculate dc-loss for root.left.left
    loss_l_l_r = torch.abs(torch.matmul(projection_l_l_r, (torch.tensor([-2,-2], dtype=torch.float32)[None] - torch.tensor([-3,-3], dtype=torch.float32)[None]).T))
    # calculate direction between root.left.right and root.left.left (just flip vector)
    projection_l_r_l = -projection_l_l_r
    # calculate dc-loss for root.left.right
    loss_l_r_l = torch.abs(torch.matmul(projection_l_r_l, (torch.tensor([-0.5,-0.5])[None] - torch.tensor([-0.4,-0.4])[None]).T))
    # calculate overall dc loss
    num_nodes = 4 # excluding root node
    batch_size = 4
    loss_manually = (loss_l_r + loss_r_l + loss_l_l_r + loss_l_r_l)/(num_nodes*batch_size)

    # create mock-autoencoder, which represents just an identity function
    encode = lambda x: x
    autoencoder = type('Autoencoder', (), {'encode': encode})
    # calculate dc loss of the tree 
    loss = tree.dc_loss(autoencoder, 4)

    assert torch.all(torch.eq(loss, loss_manually))

