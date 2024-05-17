import ml_collections
import torch 

def get_config():
    config = ml_collections.ConfigDict()
    
    # pretraining model
    config.training = training = ml_collections.ConfigDict()
    training.layer_wise = True
    training.loss_fn = {
        "MSE": torch.nn.MSELoss(),
        "L1L": torch.nn.L1Loss(),
        "Smooth": torch.nn.SmoothL1Loss(),
        "HuberLoss": torch.nn.HuberLoss()
    }
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
    return config