import ml_collections
import torch 

def get_config():
    config = ml_collections.ConfigDict()
    
    # pretraining model
    config.training = training = ml_collections.ConfigDict()
    training.path = '/home/stud/xuechun/pratical/evaluation/pre_training/model'
    training.layer_wise = False
    training.loss_fn = {
        "MSE": torch.nn.MSELoss(),
        "L1L": torch.nn.L1Loss(),
        "Smooth": torch.nn.SmoothL1Loss(),
        "HuberLoss": torch.nn.HuberLoss()
    }
    training.loss = "MSE"
    training.lr = 0.0001
    training.iter = 10
    training.pure_epochs = 50
    # path
    config.data = data= ml_collections.ConfigDict()
    data.root = '/home/stud/xuechun/pratical/evaluation/pre_training/datasets'
    data.paths = {
        "USP": f"{data.root}/usp",
        "Reuters": f"{data.root}/reuters",
        "MNIST": f"{data.root}/mnist",
        "Fashion_MNIST": f"{data.root}/fashion_mnist"
    }

    return config