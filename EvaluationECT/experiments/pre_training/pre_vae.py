from load_datasets import usps_dataset, reuters_dataset, mnist_dataset, fashion_minist
import os 
import config
from vae_model import LayerwiseVae,  PureVae
from clustpy.data import load_usps, load_mnist, load_reuters,load_fmnist

cfg = config.get_config()

def main(dataset_name):

   
    
    if cfg.training.layer_wise:
        datasets = {
        "USP": usps_dataset,
        "Reuters": reuters_dataset,
        "MNIST": mnist_dataset,
        "Fashion_MNIST": fashion_minist
        }
        
        if dataset_name in datasets:
            data, _ = datasets[dataset_name]()
        else : raise ValueError
        assert data is not None
        model = LayerwiseVae(dataset_name)
        model.layerwise(data)
    else :
        datasets = {
        "USP": load_usps,
        "Reuters": load_reuters,
        "MNIST": load_mnist,
        "Fashion_MNIST": load_fmnist
        }
    
        if dataset_name in datasets:
            data, _ = datasets[dataset_name](return_X_y=True)
        
        assert data is not None
        model = PureVae(dataset_name)
        model.forward(data)

if __name__=="__main__":
    dataset_name = [ "USP", "Reuters", "MNIST","Fashion_MNIST"]
    for name in dataset_name:    
        main(name)