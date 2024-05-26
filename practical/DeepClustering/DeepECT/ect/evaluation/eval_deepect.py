import torch
import eva_config
from deepect import DeepECT
from experiments.pre_training.load_datasets import mnist_dataset, fashion_minist, usps_dataset, reuters_dataset
from clustpy.data import load_usps, load_mnist, load_reuters,load_fmnist
from clustpy.deep import  get_trained_autoencoder, get_dataloader, encode_batchwise
from clustpy.deep.autoencoders import FeedforwardAutoencoder, StackedAutoencoder
# from experiments.pre_training.vae.stacked_ae import stacked_ae
import torch.nn.functional as F
from clustpy.metrics import unsupervised_clustering_accuracy as acc
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from dengrom_leaf import dengrom_purity, leaf_purity
from hierachecial import dendrogram_purity, leaf_purity
cfg = eva_config.get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(true_labels, pred_labels):
  print("acc:", acc(true_labels, pred_labels))
  print("nmi:", nmi(true_labels, pred_labels))
  print("ari:", ari(true_labels, pred_labels))
 
def hiera_eval(clusters, labels):
    print(dengrom_purity(clusters, labels))  
def main():
    if cfg.training.pre: 
        # if True, then use load methods in load_datasets
        datasets = {
        "USP": usps_dataset,
        "Reuters": reuters_dataset,
        "MNIST": mnist_dataset,
        "Fashion_MNIST": fashion_minist
        }
        
        if cfg.data.dataset in datasets:
            data, labels = datasets[cfg.data.dataset]()
        else : raise ValueError
        assert data is not None
        feature_dim = data.shape[1]
        layer_dims = [500, 500, 2000]
        # training
        ae = StackedAutoencoder(feature_dim, 10, layer_dims,
                    weight_initalizer=torch.nn.init.xavier_normal_,
                    activation_fn=lambda x: F.relu(x),
                    optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=0.0001))
        ae.load_state_dict(torch.load(cfg.data.model["layer_wise"][cfg.data.dataset], map_location=torch.device(device)))
        deepect = DeepECT(number_classes=10, autoencoder=ae, max_leaf_nodes=20)
        deepect.fit(data)
        # test
        evaluate(labels,deepect.DeepECT_labels_)
         
    else: 
        # use the method from clustpy
        datasets = {
        "USP": load_usps,
        "Reuters": load_reuters,
        "MNIST": load_mnist,
        "Fashion_MNIST": load_fmnist
        }
        
        data, labels = datasets[cfg.data.dataset](return_X_y=True)
        ae = FeedforwardAutoencoder(layers=[data.shape[1], 500, 500, 2000, 10])
        ae.load_state_dict(torch.load(cfg.data.model["pure"][cfg.data.dataset], map_location=torch.device(device)))
        ae.fitted = True
        deepect = DeepECT(labels, number_classes=10, autoencoder=ae, max_leaf_nodes=20)
        deepect.fit(data)
        print(len(deepect.DeepECT_labels_))
        print(deepect.dendrogram)
        leaf_purity_value = deepect.leaf
        print(leaf_purity_value)
        
        # test the model
        evaluate(labels, deepect.DeepECT_labels_)
        
    """
    MNIST:  dend: 0.8167113909026195
            leaf: 0.9208714285714286
            acc: 0.9208714285714286
            nmi: 0.8327991828732338
            ari: 0.8311824687312072
            
    USP :   acc: 0.6893430256480593
            nmi: 0.7344101253005622
            ari: 0.6027551533369153
    Reuters:
    
    fMNIST: acc: 0.50545
            nmi: 0.558446373356682
            ari: 0.3901464714980103
    """
       
        
        
if __name__ == "__main__":
        main()
    
