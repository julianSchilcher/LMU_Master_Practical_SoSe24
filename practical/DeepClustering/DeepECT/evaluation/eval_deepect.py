import torch
import eva_config
from deepect import DeepECT
from experiments.pre_training.load_datasets import mnist_dataset, fashion_minist, usps_dataset, reuters_dataset
from clustpy.data import load_usps, load_mnist, load_reuters,load_fmnist
from clustpy.deep import  get_trained_autoencoder, get_dataloader, encode_batchwise
from clustpy.deep.autoencoders import FeedforwardAutoencoder
import sys
sys.path.append("/Users/yy/LMU_Master_Practical_SoSe24/EvaluationECT/experiments/pre_training/")
from vae.stacked_ae import stacked_ae
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
        ae = stacked_ae(feature_dim, layer_dims,
                    weight_initalizer=torch.nn.init.xavier_normal_,
                    activation_fn=lambda x: F.relu(x),
                    optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=0.0001))
    
        deepect = DeepECT(labels, number_classes=10, autoencoder=ae, max_leaf_nodes=20)
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
        
        data, labels = datasets[cfg.data.dataset]("train",return_X_y=True)
        ae = stacked_ae(layers=[data.shape[1], 500, 500, 2000, 10])
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
    MNIST:  acc: 0.865
            nmi: 0.7861745965567448
            ari: 0.7417387078097011
            
    USP :   acc: 0.6893430256480593
            nmi: 0.7344101253005622
            ari: 0.6027551533369153
    Reuters:
    
    fMNIST: den: 0.4378096994105657
            leaf: 0.6662333333333333
            acc: 0.37025
            nmi: 0.49643987102875464
            ari: 0.29871404538799773
            """
       
        
        
# if __name__ == "__main__":
#         main()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def add_noise(batch):
#         mask = torch.empty(batch.shape, device=batch.device).bernoulli_(0.8)
#         return batch * mask
#     dataset, labels = load_reuters(return_X_y=True)
#     feature_dim = dataset.shape[1]
#     layer_dims = [500, 500, 2000, 10]
#     weight_initalizer=torch.nn.init.xavier_normal_
    
#     loss_fn = torch.nn.MSELoss()
#     steps_per_layer = 20000
#     refine_training_steps = 50000

#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)
#     model = stacked_ae(feature_dim, layer_dims,
#             weight_initalizer,
#             activation_fn=lambda x: torch.nn.functional.relu(x),
#             loss_fn=loss_fn,
#             optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=0.0001)).to(device)
#     model.pretrain(train_loader, steps_per_layer, corruption_fn=add_noise)
#     model.refine_training(train_loader, refine_training_steps, corruption_fn=add_noise)
#     path = f"layerWise_Reuters.pth"


#     model_path = path
#     os.makedirs(model_path,exist_ok=True)

#     torch.save(model.state_dict(), model_path+"/model.path")
#     print(f"Model saved to {model_path}")


#     model.fitted=True
#     deepect = DeepECT(autoencoder=model, max_leaf_nodes=20)
#     deepect.fit(dataset)
#     print(deepect.tree_.flat_accuracy(labels, n_clusters=10))
#     print(deepect.tree_.flat_nmi(labels, n_clusters=10))
#     print(deepect.tree_.flat_ari(labels, n_clusters=10))
#     print(deepect.tree_.dendrogram_purity(labels))
#     print(deepect.tree_.leaf_purity(labels))
    
