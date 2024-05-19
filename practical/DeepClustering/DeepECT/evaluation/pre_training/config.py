import ml_collections
import torch 

def get_config():
    config = ml_collections.ConfigDict()
    
    # pretraining model
    config.training = training = ml_collections.ConfigDict()
<<<<<<< HEAD
    training.path = '/home/stud/xuechun/pratical/evaluation/pre_training/model'
    training.layer_wise = False
=======
    training.layer_wise = True
>>>>>>> 5bb5ea7211dbcb3a74f00159969a4f3ae16c9d14
    training.loss_fn = {
        "MSE": torch.nn.MSELoss(),
        "L1L": torch.nn.L1Loss(),
        "Smooth": torch.nn.SmoothL1Loss(),
        "HuberLoss": torch.nn.HuberLoss()
    }
<<<<<<< HEAD
    training.loss = "MSE"
    training.lr = 0.0001
    training.iter = 10
    training.pure_epochs = 50
    # path
    config.data = data= ml_collections.ConfigDict()
    data.root = '/home/stud/xuechun/pratical/evaluation/pre_training/datasets'
    data.paths = {
        "UPS": f"{data.root}/ups",
        "Reuters": f"{data.root}/reuters",
        "MNIST": f"{data.root}/mnist",
        "Fashion_MNIST": f"{data.root}/fashion_mnist"
    }

=======
    training.lr = 0.0001
    training.pure_epochs = 50
    # path
    config.data = data= ml_collections.ConfigDict()
    data.root = '../pre_training/datasets'
    data.paths = {
        "UPS":f"{data.root}/ups",
        "Reuters":f"{data.root}/reuters",
        "MNIST": f"{data.root}/mnist",
        "Fashion_MNIST":f"{data.root}/mnist"
    }
>>>>>>> 5bb5ea7211dbcb3a74f00159969a4f3ae16c9d14
    return config