import ml_collections
import torch 

def get_config():
    config = ml_collections.ConfigDict()
    
    # pretraining model
    config.training = training = ml_collections.ConfigDict()
    training.path = 'C:/Users/Li/Desktop/deep_clustering/DeepECT/DeepECT/evaluation/pre_training/evaluation/pre_training/model'
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
    data.root = 'C:/Users/Li/Desktop/deep_clustering/DeepECT/DeepECT/evaluation/pre_training/evaluation/pre_training/datasets'
    data.paths = {
        "UPS": f"{data.root}/ups",
        "Reuters": f"{data.root}/reuters",
        "MNIST": f"{data.root}/mnist",
        "Fashion_MNIST": f"{data.root}/fashion_mnist"
    }

    return config