from pre_training.vae import stacked_ae
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

data = ...  # Define the variable "data" with the appropriate value
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
        