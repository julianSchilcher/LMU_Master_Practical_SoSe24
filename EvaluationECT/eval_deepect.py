import torch
import eva_config
from deepect import DeepECT
import argparse
from experiments.pre_training.load_datasets import mnist_dataset, fashion_minist, usps_dataset, reuters_dataset
from clustpy.data import load_usps, load_mnist, load_reuters,load_fmnist
from clustpy.deep import  get_trained_autoencoder, get_dataloader
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
            data, _ = datasets[cfg.data.dataset]()
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
        encoded = ae.encode(train_loader)
        X_train, X_test = train_test_split(encoded, test_size=0.2, random_state=42)

        deepect = DeepECT(number_classes=10, autoencoder=ae, rec_loss_fn=cfg.training.loss[cfg.training.loss_fn], max_leaf_nodes=20)
        deepect.fit(X_train)
        # test
        pre = deepect.predict(X_test)
        evaluate(pre)
         
    else: 
        # use the method from clustpy
        datasets = {
        "USP": load_usps,
        "Reuters": load_reuters,
        "MNIST": load_mnist,
        "Fashion_MNIST": load_fmnist
        }
        
        model_path = cfg.data.model["pure"][cfg.data.dataset]
        dataset, labels = datasets[cfg.data.dataset]("train", return_X_y=True)
        trainloader = get_dataloader(data, 256, True, False)  
        autoencoder = get_trained_autoencoder(trainloader, optimizer_params={"lr":1e-3}, n_epochs=50, device=device, optimizer_class=torch.optim.Adam, embedding_size=10)
        autoencoder.load_state_dict(torch.load(model_path))
       
        autoencoder.fitted = True
        deepect = DeepECT(number_classes=10, autoencoder=autoencoder, max_leaf_nodes=20)
        deepect.fit(dataset)
        print(deepect.DeepECT_labels_)
        print(deepect.DeepECT_cluster_centers_)
        
        # test the model
        test_data, _ = datasets[cfg.data.dataset]("test", return_X_y=True)
        testloader = get_dataloader(test_data, 256, False, False)
        autoencoder = get_trained_autoencoder(testloader, optimizer_params={"lr":1e-3}, n_epochs=50, device=device, optimizer_class=torch.optim.Adam, embedding_size=10)
        autoencoder.load_state_dict(torch.load(model_path))
       
        autoencoder.fitted = True
        pred = deepect.predict(testloader)
        evaluate(_, pred)
       
        
        
if __name__ == "__main__":
        main()
    