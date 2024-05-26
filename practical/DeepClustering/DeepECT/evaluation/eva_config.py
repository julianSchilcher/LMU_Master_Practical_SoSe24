import ml_collections
import torch

def get_config():
    config = ml_collections.ConfigDict()
    
    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "Reuters"
    data.model = {
                "layer_wise":{ "Fashion_MNIST": "EvaluationECT/experiments/pre_training/model/layer_wise/Fashion_MNIST_pre/vae_Fashion_MNIST_4/model.path",
                            "MNIST": "C:/Users/Li/Desktop/deepect/LMU_Master_Practical_SoSe24/EvaluationECT/experiments/pre_training/model/layer_wise/MNIST_pre/vae_MNIST_0/model.path",
                            "Reuters": "EvaluationECT/experiments/pre_training/model/layer_wise/Reuters_pre/vae_Reuters_0/model.path",
                            "USP": "EvaluationECT/experiments/pre_training/model/layer_wise/UPS_pre/vae_UPS_9/model.path" },
                "pure":{ "Fashion_MNIST": "C:/Users/Li/Desktop/deepect/LMU_Master_Practical_SoSe24/EvaluationECT/experiments/pre_training/model/pure_vae/Fashion_MNIST_pre/model.path",
                            "MNIST": "C:/Users/Li/Desktop/deepect/LMU_Master_Practical_SoSe24/EvaluationECT/experiments/pre_training/model/pure_vae/MNIST_pre/model.path",
                            "Reuters": "C:/Users/Li/Desktop/deepect/LMU_Master_Practical_SoSe24/EvaluationECT/experiments/pre_training/model/pure_vae/Reuters_pre/model.path",
                            "USP": "C:/Users/Li/Desktop/deepect/LMU_Master_Practical_SoSe24/EvaluationECT/experiments/pre_training/model/pure_vae/USP_pre/model.path" },
    }   

    # training and test
    config.training = training = ml_collections.ConfigDict()
    # the training.pre is to choose pretraining model.
    training.pre = False
    training.lr = 0.001
    training.iter = 5000
    training.loss = {
        "MSE": torch.nn.MSELoss(),
        "L1L": torch.nn.L1Loss(),
        "Smooth": torch.nn.SmoothL1Loss(),
        "HuberLoss": torch.nn.HuberLoss()
    }
    training.loss_fn = "MSE"
    
    
    return config