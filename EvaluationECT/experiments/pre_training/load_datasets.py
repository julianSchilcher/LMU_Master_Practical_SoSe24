import torch
import torch.utils
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path
import nltk
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from experiments.pre_training.config import get_config
from nltk.tokenize import word_tokenize
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import os

transform = transforms.ToTensor()
cfg = get_config()

def mnist_dataset():
    if not os.path.exists(cfg.data.paths["MNIST"]):
        os.makedirs(cfg.data.paths["MNIST"])
    mnist_train = datasets.MNIST(cfg.data.paths["MNIST"], train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(cfg.data.paths["MNIST"], train=False, download=True, transform=transform)
    data = torch.cat([mnist_train.data, mnist_test.data], dim=0)
    labels = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)
    return  torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype= torch.float32)  

def usps_dataset():
    # Can be downloaded here: https://github.com/marionmari/Graph_stuff/tree/master/usps_digit_data
    dir_path = Path(cfg.data.paths["USP"])
    file_path = dir_path / 'usps_resampled.mat'

    # Ensure the directory exists
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    if not file_path.exists():
        import requests
        response = requests.get(
            "https://github.com/marionmari/Graph_stuff/blob/master/usps_digit_data/usps_resampled.mat?raw=true")
        with open(file_path, 'wb') as f:
            f.write(response.content)

    data_mat = loadmat(file_path)

    data = torch.cat([torch.tensor(data_mat['train_patterns'].T), torch.tensor(data_mat['test_patterns'].T)], 0)
    labels = torch.argmax(torch.cat([torch.tensor(data_mat['train_labels'].T), torch.tensor(data_mat['test_labels'].T)], 0), 1)
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype= torch.float32)

def reuters_dataset():
    if not os.path.exists(cfg.data.paths["Reuters"]):
        os.makedirs(cfg.data.paths["Reuters"])
    nltk.download('reuters', download_dir=cfg.data.paths["Reuters"])
    nltk.download('punkt', download_dir=cfg.data.paths["Reuters"])
    
    docs = reuters.fileids()
    docs_list = [word_tokenize(reuters.raw(doc_id)) for doc_id in docs]
    
    vocab = defaultdict(lambda:len(vocab))
    
    docs_indices = [[vocab[token] for token in doc] for doc in docs_list]
    
    docs_tensor = [torch.tensor(doc, dtype=torch.float32) for doc in docs_indices]
    docs_tensor_pad = pad_sequence(docs_tensor, batch_first=True, padding_value=0)
    
    labels = reuters.categories()
    
    return docs_tensor_pad, labels

def fashion_minist():
    if not os.path.exists(cfg.data.paths["Fashion_MNIST"]):
        os.makedirs(cfg.data.paths["Fashion_MNIST"])
    fashion_mnist_train = datasets.FashionMNIST(cfg.data.paths["Fashion_MNIST"], train=True, download=True, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(cfg.data.paths["Fashion_MNIST"], train=False, download=True, transform=transform)
    data = torch.cat([fashion_mnist_train.data, fashion_mnist_test.data], dim=0)
    labels = torch.cat([fashion_mnist_train.targets, fashion_mnist_test.targets], dim=0)
    return  torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype= torch.float32)  
