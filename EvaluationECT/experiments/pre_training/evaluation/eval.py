import torch
import config
from ect import Train, data_aug
cfg = config.get_train_config()

def main(aug, dataset_name):
    """
        the function is to train the cluster algorithm
        dataset: UPS, MNIST, Reuters, Fashion_MNIST
        args:
            aug: True: data augumentation, False: without data augumentation
            cfg.alg: use layer_wise pretraining, or pure pretraining 
    """
    data = data_aug(dataset_name, aug)
    Train(cfg.alg, dataset_name, data)
  


if __name__ =="__main__":
    for name in ["UPS", "Reuters", "MNIST", "Fashion_MNIST"]:
        main(cfg.aug, dataset_name=name)
