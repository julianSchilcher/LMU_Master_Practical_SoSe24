from pre_training.vae import stacked_ae
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

def add_noise(batch):
    mask = torch.empty(batch.shape, device= batch.device).bernoulli_(0.8)
    return batch*mask
def total_loss(train_loader):
    loss = 0.0
    for input in train_loader:
        pre = model.forward(input)[1]
        loss+= loss_fn(pre, input).item()
    return loss

feature_dim=data.shape[0]
layer_dims = [500, 500, 2000, 10]
loss_fn = nn.MSELoss()
weight_init = torch.nn.init.xavier_normal_
optimizer = lambda parameters: torch.optim.Adam(parameters, lr=0.0001)
activation_fn = lambda x: F.relu(x)
# Nr of Training steps per layer
steps_per_layer = 20000
# Nr of Finetuning
refine_training_steps = 50000
iter = 10
for i in range(iter):
    del model
    model = stacked_ae(feature_dim, layer_dims, weight_init, loss_fn, activation_fn)
    model.pretrain(train_loder, steps_per_layer, corruption_fn=add_noise)
    model.refine_training(train_loder, refine_training_steps, corruption_fn=add_noise)
    loss = total_loss(train_loader)
    torch.save(model.state_dict(), f"{vae_pretraining}/vae_{dataset_name}_{i}.model")