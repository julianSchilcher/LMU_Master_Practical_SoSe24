import torch
import torch.nn.functional as F
import sys
import logging
from clustpy.deep.autoencoders._abstract_autoencoder import _AbstractAutoencoder
from clustpy.data import load_fmnist

# Ensure proper module import
sys.path.append('/Users/yy/LMU_Master_Practical_SoSe24/EvaluationECT/experiments/pre_training/vae/')
from functional import window

logger = logging.getLogger(__name__)

# Based on: https://gist.github.com/InnovArul/500e0c57e88300651f8005f9bd0d12bc
class stacked_ae(_AbstractAutoencoder):
    def __init__(self, feature_dim, layer_dims, weight_initializer,
                 loss_fn=lambda x, y: torch.mean((x - y) ** 2),
                 optimizer_fn=lambda parameters: torch.optim.Adam(parameters, lr=0.001),
                 tied_weights=False, activation_fn=None, bias_init=0.0, linear_embedded=True, linear_decoder_last=True):
        super().__init__()
        self.tied_weights = tied_weights
        self.loss_fn = loss_fn
        self.optimizer_fn = optimizer_fn
        self.linear_decoder_last = linear_decoder_last
        self.linear_embedded = linear_embedded
        self.n_layers = len(layer_dims)

        self.param_bias_encoder = []
        self.param_bias_decoder = []
        self.param_weights_encoder = []
        if tied_weights:
            self.param_weights_decoder = None
        else:
            self.param_weights_decoder = []

        layer_params = list(window([feature_dim] + layer_dims, 2))

        for l in range(self.n_layers):
            feature_dim, node_dim = layer_params[l]
            encoder_weight = torch.empty(node_dim, feature_dim)
            weight_initializer(encoder_weight)
            encoder_weight = torch.nn.Parameter(encoder_weight, requires_grad=True)
            self.register_parameter(f"encoder_weight_{l}", encoder_weight)
            self.param_weights_encoder.append(encoder_weight)
            encoder_bias = torch.empty(node_dim)
            encoder_bias.fill_(bias_init)
            encoder_bias = torch.nn.Parameter(encoder_bias, requires_grad=True)
            self.register_parameter(f"encoder_bias_{l}", encoder_bias)
            self.param_bias_encoder.append(encoder_bias)

            if not tied_weights:
                decoder_weight = torch.empty(feature_dim, node_dim)
                weight_initializer(decoder_weight)
                decoder_weight = torch.nn.Parameter(decoder_weight, requires_grad=True)
                self.register_parameter(f"decoder_weight_{l}", decoder_weight)
                self.param_weights_decoder.append(decoder_weight)
            decoder_bias = torch.empty(feature_dim)
            decoder_bias.fill_(bias_init)
            decoder_bias = torch.nn.Parameter(decoder_bias, requires_grad=True)
            self.register_parameter(f"decoder_bias_{l}", decoder_bias)
            self.param_bias_decoder.append(decoder_bias)
        if not tied_weights:
            self.param_weights_decoder.reverse()
        self.param_bias_decoder.reverse()
        self.activation_fn = activation_fn

    def forward_pretrain(self, input_data, stack, use_dropout=True, dropout_rate=0.2,
                         dropout_is_training=True):
        encoded_data = input_data
        if stack < 1 or stack > self.n_layers:
            raise RuntimeError(f"stack number {stack} is out or range (0,{self.n_layers})")
        for l in range(stack):
            weights = self.param_weights_encoder[l]
            bias = self.param_bias_encoder[l]
            encoded_data = F.linear(encoded_data, weights, bias)

            if self.activation_fn is not None:
                if self.linear_embedded is False or not (l == stack - 1 and stack == self.n_layers):
                    encoded_data = self.activation_fn(encoded_data)
            if use_dropout:
                if not (l == stack - 1 and stack == self.n_layers):
                    encoded_data = F.dropout(encoded_data, p=dropout_rate, training=dropout_is_training)
        reconstructed_data = encoded_data

        for ll in range(stack - 1, -1, -1):
            l = self.n_layers - ll - 1
            if self.tied_weights:
                weights = self.param_weights_encoder[self.n_layers - l - 1].t()
            else:
                weights = self.param_weights_decoder[l]
            bias = self.param_bias_decoder[l]
            reconstructed_data = F.linear(reconstructed_data, weights, bias)
            if self.activation_fn is not None:
                if self.linear_decoder_last is False or self.linear_decoder_last and ll > 0:
                    reconstructed_data = self.activation_fn(reconstructed_data)
            if use_dropout and ll > 0:
                reconstructed_data = F.dropout(reconstructed_data, p=dropout_rate)

        return encoded_data, reconstructed_data

    def encode(self, input_data):
        encoded_data = input_data
        for l in range(self.n_layers):
            weights = self.param_weights_encoder[l]
            bias = self.param_bias_encoder[l]
            encoded_data = F.linear(encoded_data, weights, bias)
            if self.activation_fn is not None and not (self.linear_embedded and l == self.n_layers - 1):
                encoded_data = self.activation_fn(encoded_data)
        return encoded_data

    def decode(self, encoded_data):
        reconstructed_data = encoded_data

        for l in range(self.n_layers):
            if self.tied_weights:
                weights = self.param_weights_encoder[self.n_layers - l - 1].t()
            else:
                weights = self.param_weights_decoder[l]
            bias = self.param_bias_decoder[l]
            reconstructed_data = F.linear(reconstructed_data, weights, bias)
            if self.activation_fn is not None and not (self.linear_decoder_last and l == self.n_layers - 1):
                reconstructed_data = self.activation_fn(reconstructed_data)
        return reconstructed_data

    def forward(self, input_data):
        encoded_data = self.encode(input_data)
        reconstructed_data = self.decode(encoded_data)

        return encoded_data, reconstructed_data

    def parameters_pretrain(self, stack):
        parameters = []
        for l in range(stack):
            parameters.append(self.param_weights_encoder[l])
            parameters.append(self.param_bias_encoder[l])
        for ll in range(stack - 1, -1, -1):
            l = self.n_layers - ll - 1
            if not self.tied_weights:
                parameters.append(self.param_weights_decoder[l])
            parameters.append(self.param_bias_decoder[l])
        return parameters

    def pretrain(self, dataset, rounds_per_layer=1000, dropout_rate=0.2, corruption_fn=None):
        for layer in range(1, self.n_layers + 1):
            logger.debug(f"Pretrain layer {layer}")
            optimizer = self.optimizer_fn(self.parameters_pretrain(layer))
            round = 0
            while True:
                for batch_data in dataset:
                    round += 1
                    if round > rounds_per_layer:
                        break

                    batch_data = batch_data[0].to(next(self.parameters()).device)
                    if corruption_fn is not None:
                        corrupted_batch = corruption_fn(batch_data)
                        _, reconstructed_data = self.forward_pretrain(corrupted_batch, layer, use_dropout=True,
                                                                      dropout_rate=dropout_rate,
                                                                      dropout_is_training=True)
                    else:
                        _, reconstructed_data = self.forward_pretrain(batch_data, layer, use_dropout=True,
                                                                      dropout_rate=dropout_rate,
                                                                      dropout_is_training=True)
                    loss = self.loss_fn(batch_data, reconstructed_data)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if round % 100 == 0:
                        logger.debug(f"Round {round} current loss: {loss.item()}")
                else:
                    continue
                break

    def refine_training(self, dataset, rounds, corruption_fn=None, optimizer_fn=None):
        logger.debug(f"Refine training")
        if optimizer_fn is None:
            optimizer = self.optimizer_fn(self.parameters())
        else:
            optimizer = optimizer_fn(self.parameters())

        index = 0
        while True:
            for batch_data in dataset:
                index += 1
                if index > rounds:
                    break
                batch_data = batch_data[0].to(next(self.parameters()).device)

                if corruption_fn is not None:
                    embedded_data, reconstructed_data = self.forward(corruption_fn(batch_data))
                else:
                    embedded_data, reconstructed_data = self.forward(batch_data)

                loss = self.loss_fn(reconstructed_data, batch_data)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if index % 100 == 0:
                    logger.debug(f"Round {index} current loss: {loss.item()}")
            else:
                continue
            break

def example_weight_initializer(tensor):
    if isinstance(tensor, torch.Tensor):
        torch.nn.init.xavier_uniform_(tensor)

if __name__ == "__main__":
    # Load dataset
    dataset, labels = load_fmnist(return_X_y=True)
    feature_dim = dataset.shape[1]
    layer_dims = [500, 500, 2000, 10]

    # Initialize autoencoder
    autoencoder = stacked_ae(
        feature_dim=feature_dim,
        layer_dims=layer_dims)

    autoencoder.fitted = True

    # Placeholder for training dataset loader
    # Assuming dataset is a DataLoader object
    # Example: dataset = DataLoader(TensorDataset(torch.from_numpy(dataset)), batch_size=64, shuffle=True)
    
    # Pretrain and refine training steps (assuming you have a DataLoader for dataset)
    # autoencoder.pretrain(dataset, rounds_per_layer=1000, dropout_rate=0.2)
    # autoencoder.refine_training(dataset, rounds=10000)

    print("Autoencoder initialized and ready for training.")
