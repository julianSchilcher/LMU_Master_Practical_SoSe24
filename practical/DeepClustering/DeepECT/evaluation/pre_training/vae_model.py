import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
from vae.stacked_ae import stacked_ae
import sys
sys.path.append("DeepECT/evaluation/pre_training/ClustPy")
from ClustPy.clustpy.deep import detect_device, get_trained_autoencoder, get_dataloader
import config

cfg = config.get_config()


class PureVae:
    def __init__(self,loss):
        super(PureVae, self).__init__()
        self.lr = cfg.training.lr
        self.epochs = cfg.training.pure_epochs
        # self.device = detect_device()
        self.loss = cfg.training.loss_fn(loss)
    def forward(self, data):
        train_loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True, pin_memory=True)
        ae = get_trained_autoencoder(train_loader, optimizer_params={"lr":self.lr}, n_epochs=self.epochs, device=self.device, optimizer_class=torch.optim.Adam, loss_fn=self.loss, embedding_size=10)
        return 
        
        
        
class LayerwiseVae:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.vae_pretraining = f"{dataset_name}_pre"
        os.makedirs(f"/model/layer_wise/{self.vae_pretraining}", exist_ok=True)

    def add_noise(self, batch):
        mask = torch.empty(batch.shape, device=batch.device).bernoulli_(0.8)
        return batch * mask

    def total_loss(self, train_loader, model, loss_fn):
        total_loss = 0.0
        for inputs, _ in train_loader:
            if isinstance(inputs, list):
                inputs = torch.tensor(inputs, dtype=torch.float32).to(next(model.parameters()).device)
            pred = model(inputs)
            total_loss += loss_fn(pred, inputs).item()
        return total_loss

    def layerwise(self, data):
        feature_dim = data.shape[1]
        layer_dims = [500, 500, 2000, 10]
        loss_fn = nn.MSELoss()
        steps_per_layer = 20000
        refine_training_steps = 50000
        iterations = 10

        for i in range(iterations):
            data = torch.utils.data.TensorDataset(data)
            train_loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True, pin_memory=True)
            model = stacked_ae(feature_dim, layer_dims,
                    weight_initalizer=torch.nn.init.xavier_normal_,
                    activation_fn=lambda x: F.relu(x),
                    loss_fn=loss_fn,
                    optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=0.0001)).cuda()
            model.pretrain(train_loader, steps_per_layer, corruption_fn=self.add_noise)
            model.refine_training(train_loader, refine_training_steps, corruption_fn=self.add_noise)
            loss = self.total_loss(train_loader, model, loss_fn)
            
            loss_file = os.path.join(self.vae_pretraining, 'loss.log')
            with open(loss_file, "a") as f:
                f.write(f"Iter: {i}, Loss: {loss}\n")
            
            model_path = f"model/layer_wise/{self.vae_pretraining}/vae_{self.dataset_name}_{i}.model"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")