from load_datasets import load_usps, reuters_dataset, mnist_dataset, fashion_minist
import os 
import config
from vae_model import LayerwiseVae,  PureVae

cfg = config.get_config()

def main(dataset_name):

    datasets = {
        "UPS": load_usps,
        "Reuters": reuters_dataset,
        "MNIST": mnist_dataset,
        "Fashion_MNIST": fashion_minist
    }
    
    if dataset_name in datasets:
        data, _ = datasets[dataset_name]()
    else : raise ValueError
    assert data is not None
    
    if cfg.training.layer_wise:
        model = LayerwiseVae(dataset_name)
        model.layerwise(data)
    else :
        model = PureVae(dataset_name)
        model.forward(data)

if __name__=="__main__":
    dataset_name = [ "UPS","Reuters", "MNIST", "Fashion_MNIST"]
    for name in dataset_name:    
        main(name)