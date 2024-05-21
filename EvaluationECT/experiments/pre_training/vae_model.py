import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from vae.stacked_ae import stacked_ae
from clustpy.deep import get_dataloader, get_trained_autoencoder, get_dataloader
import config

cfg = config.get_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PureVae:
    def __init__(self, dataset_name):
        super(PureVae, self).__init__()
        self.lr = cfg.training.lr
        self.epochs = cfg.training.pure_epochs
        self.loss = cfg.training.loss_fn[cfg.training.loss]
        self.vae_pretraining = f"{dataset_name}_pre"
        os.makedirs(f"{cfg.training.path}/pure_vae/{self.vae_pretraining}", exist_ok=True)
    def forward(self, data):
        trainloader = get_dataloader(data, 256, True, False)  
        ae = get_trained_autoencoder(trainloader, optimizer_params={"lr":1e-3}, n_epochs=self.epochs, device=device, optimizer_class=torch.optim.Adam, loss_fn=self.loss, embedding_size=10)
        model_path = f"{cfg.training.path}/pure_vae/{self.vae_pretraining}"
        os.makedirs(model_path, exist_ok=True)
        torch.save(ae.state_dict(),model_path+"/model.path")
        return 
        
        
class LayerwiseVae:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.vae_pretraining = f"{dataset_name}_pre"
        os.makedirs(f"{cfg.training.path}/layer_wise/{self.vae_pretraining}", exist_ok=True)
        self.model = None
    def add_noise(self, batch):
        mask = torch.empty(batch.shape, device=batch.device).bernoulli_(0.8)
        return batch * mask

    def total_loss(self, train_loader, loss_fn):
        total_loss = 0.0
        for inputs in train_loader:
            total_loss += loss_fn(inputs.to(device),self.model.forward(inputs.to(device))[1]).cpu().item()
        return total_loss

    def layerwise(self, data):
        feature_dim = data.shape[1]
        layer_dims = [500, 500, 2000, 10]
        loss_fn = cfg.training.loss_fn[cfg.training.loss]
        steps_per_layer = 20000
        refine_training_steps = 50000
        iterations = cfg.training.iter

        for i in range(iterations):
            train_loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True, pin_memory=True)
            self.model = stacked_ae(feature_dim, layer_dims,
                    weight_initalizer=torch.nn.init.xavier_normal_,
                    activation_fn=lambda x: F.relu(x),
                    loss_fn=loss_fn,
                    optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=0.0001)).to(device)
            self.model.pretrain(train_loader, steps_per_layer, corruption_fn=self.add_noise)
            self.model.refine_training(train_loader, refine_training_steps, corruption_fn=self.add_noise)
            loss = self.total_loss(train_loader, loss_fn)
            path = f"{cfg.training.path}/layer_wise/{self.vae_pretraining}"
            with open(path+"_loss.log", "a") as f:
                f.write(f"Iter: {i}, Loss: {loss}\n")


            model_path = f"{cfg.training.path}/layer_wise/{self.vae_pretraining}/vae_{self.dataset_name}_{i}"
            os.makedirs(model_path,exist_ok=True)

            torch.save(self.model.state_dict(), model_path+"/model.path")
            print(f"Model saved to {model_path}")
            self.model = None
