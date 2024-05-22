import torch
import eva_config
from deepect import DeepECT
from experiments.pre_training.load_datasets import mnist_dataset, fashion_minist, usps_dataset, reuters_dataset
from clustpy.data import load_usps, load_mnist, load_reuters,load_fmnist
from clustpy.deep import  get_trained_autoencoder, get_dataloader, encode_batchwise
from clustpy.deep.autoencoders import FeedforwardAutoencoder
from experiments.pre_training.vae.stacked_ae import stacked_ae
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from clustpy.metrics import unsupervised_clustering_accuracy as acc
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari


cfg = eva_config.get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(true_labels, pred_labels):
  print("acc:", acc(true_labels, pred_labels))
  print("nmi:", nmi(true_labels, pred_labels))
  print("ari:", ari(true_labels, pred_labels))
  
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
        layer_dims = [500, 500, 2000, 10]
        # training
        train_loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True, pin_memory=True)
        ae = stacked_ae(feature_dim, layer_dims,
                    weight_initalizer=torch.nn.init.xavier_normal_,
                    activation_fn=lambda x: F.relu(x),
                    optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=0.0001))
        ae.load_state_dict(torch.load(cfg.data.model["layer_wise"][cfg.data.dataset]))
        
        X_train, X_test, _, y_test = train_test_split(train_loader, labels, test_size=0.2, random_state=42, stratify=labels)

        deepect = DeepECT(number_classes=10, autoencoder=ae, rec_loss_fn=cfg.training.loss[cfg.training.loss_fn], max_leaf_nodes=20)
        deepect.fit(X_train)
        # test
        pre = deepect.predict(X_test)
        evaluate(y_test,pre)
         
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
        deepect = DeepECT(number_classes=10, autoencoder=ae)
        deepect.fit(data)
        print(deepect.DeepECT_labels_)
        print(deepect.DeepECT_cluster_centers_)
        
        # test the model
        testloader = get_dataloader(data, 256, False, False)
        deepect.predict(10, testloader,ae)
        evaluate(_, deepect.DeepECT_labels_)
       
        
        
if __name__ == "__main__":
        main()
    