##### Basic execution of DeepECT on MNIST without having an pretrained autoencoder
from clustpy.data.real_torchvision_data import load_mnist
from practical.DeepClustering.DeepECT.deepect_paper import DeepECT # version adjusted to code from paper
# get dataset
data, labels = load_mnist(return_X_y=True)
# run algorithm with default settings (autoencoder will be pretrained)
deepect = DeepECT(autoencoder_param_path="path/to/store/pretrained/autoencoder")
deepect.fit_predict(data)
# evaluation
deepect.tree_.flat_accuracy(labels, 10)
deepect.tree_.flat_nmi(labels, 10)
deepect.tree_.dendrogram_purity(labels)



##### Basic execution of DeepECT on MNIST with an pretrained feedforward autoencoder
from clustpy.data.real_torchvision_data import load_mnist
from practical.DeepClustering.DeepECT.deepect_paper import DeepECT # version adjusted to code from paper
from clustpy.deep.autoencoders import FeedforwardAutoencoder
import torch
# get dataset
data, labels = load_mnist(return_X_y=True)
# load autoencoder
my_autoencoder = FeedforwardAutoencoder([data.shape[1], 500, 500, 2000, 10])
my_autoencoder.load_state_dict(torch.load("path/to/store/pretrained/autoencoder"))
my_autoencoder.fitted = True
# run algorithm with default settings
deepect = DeepECT(autoencoder=my_autoencoder)
deepect.fit_predict(data)
# evaluation
deepect.tree_.flat_accuracy(labels, 10)
deepect.tree_.flat_nmi(labels, 10)
deepect.tree_.dendrogram_purity(labels)



##### Basic execution of DeepECT on MNIST with augmentations
from clustpy.data.real_torchvision_data import load_mnist
from practical.DeepClustering.DeepECT.deepect_paper import DeepECT # version adjusted to code from paper
from practical.DeepClustering.DeepECT.evaluation_pipeline import  DatasetType, get_custom_dataloader_augmentations
# get dataset
data, labels = load_mnist(return_X_y=True)
dataloaders_with_augmentation = get_custom_dataloader_augmentations(data, DatasetType.MNIST)
# run algorithm with default settings (autoencoder will be pretrained)
deepect = DeepECT(autoencoder_param_path="path/to/store/pretrained/autoencoder", custom_dataloaders=dataloaders_with_augmentation, augmentation_invariance=True)
deepect.fit_predict(data)
# evaluation
deepect.tree_.flat_accuracy(labels, 10)
deepect.tree_.flat_nmi(labels, 10)
deepect.tree_.dendrogram_purity(labels)