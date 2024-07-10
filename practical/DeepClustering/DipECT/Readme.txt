##### Basic execution of DipECT on MNIST without having an pretrained autoencoder
from clustpy.data.real_torchvision_data import load_mnist
from practical.DeepClustering.DipECT.dipect import DipECT
# get dataset
data, labels = load_mnist(return_X_y=True)
# run algorithm with default settings (autoencoder will be pretrained)
dipect = DipECT(autoencoder_param_path="path/to/store/pretrained/autoencoder")
dipect.fit_predict(data, labels)
# evaluation
dipect.tree_.flat_accuracy(labels, 10)
dipect.tree_.flat_nmi(labels, 10)
dipect.tree_.dendrogram_purity(labels)



##### Basic execution of DeepECT on MNIST with an pretrained feedforward autoencoder
from clustpy.data.real_torchvision_data import load_mnist
from practical.DeepClustering.DipECT.dipect import DipECT
from clustpy.deep.autoencoders import FeedforwardAutoencoder
import torch
# get dataset
data, labels = load_mnist(return_X_y=True)
# load autoencoder
my_autoencoder = FeedforwardAutoencoder([data.shape[1], 500, 500, 2000, 10])
my_autoencoder.load_state_dict(torch.load("path/to/store/pretrained/autoencoder"))
my_autoencoder.fitted = True
# run algorithm with default settings
dipect = DipECT(autoencoder=my_autoencoder)
dipect.fit_predict(data, labels)
# evaluation
dipect.tree_.flat_accuracy(labels, 10)
dipect.tree_.flat_nmi(labels, 10)
dipect.tree_.dendrogram_purity(labels)



##### Basic execution of DeepECT on MNIST with augmentations
from clustpy.data.real_torchvision_data import load_mnist
from practical.DeepClustering.DipECT.dipect import DipECT
from practical.DeepClustering.DipECT.evaluation_pipeline import  DatasetType, get_custom_dataloader_augmentations
# get dataset
data, labels = load_mnist(return_X_y=True)
dataloaders_with_augmentation = get_custom_dataloader_augmentations(data, DatasetType.MNIST)
# run algorithm with default settings (autoencoder will be pretrained)
dipect = DipECT(autoencoder_param_path="path/to/store/pretrained/autoencoder", custom_dataloaders=dataloaders_with_augmentation, augmentation_invariance=True)
dipect.fit_predict(data, labels)
# evaluation
dipect.tree_.flat_accuracy(labels, 10)
dipect.tree_.flat_nmi(labels, 10)
dipect.tree_.dendrogram_purity(labels)