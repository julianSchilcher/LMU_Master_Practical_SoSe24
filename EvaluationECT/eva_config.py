import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()
    
    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "MNIST"
    data.model = {
                "layer_wise":{ "USP": "EvaluationECT/experiments/pre_training/model/layer_wise/Fashion_MNIST_pre/vae_Fashion_MNIST_4/model.path",
                            "Reuters": "EvaluationECT/experiments/pre_training/model/layer_wise/MNIST_pre/vae_MNIST_0/model.path",
                            "MNIST": "EvaluationECT/experiments/pre_training/model/layer_wise/Reuters_pre/vae_Reuters_0/model.path",
                            "Fashion_MNIST": "EvaluationECT/experiments/pre_training/model/layer_wise/UPS_pre/vae_UPS_9/model.path" },
            
    }

    # training and test
    config.training = training = ml_collections.ConfigDict()
    # the training.pre is to choose pretraining model.
    training.pre = True
    training.lr = 0.0001
    training.iter = 5000
    training.loss = {
        "MSE": torch.nn.MSELoss(),
        "L1L": torch.nn.L1Loss(),
        "Smooth": torch.nn.SmoothL1Loss(),
        "HuberLoss": torch.nn.HuberLoss()
    }
    training.loss_fn = "MSE"
    
    
    return config