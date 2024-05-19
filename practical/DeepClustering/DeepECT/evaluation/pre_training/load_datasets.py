import torch
<<<<<<< HEAD
import torch.utils
import torch.utils.data
=======
>>>>>>> 5bb5ea7211dbcb3a74f00159969a4f3ae16c9d14
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pathlib import Path
import nltk
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import config 
<<<<<<< HEAD
from nltk.tokenize import word_tokenize
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
import os
=======
>>>>>>> 5bb5ea7211dbcb3a74f00159969a4f3ae16c9d14

transform = transforms.ToTensor()
cfg = config.get_config()

def mnist_dataset():
<<<<<<< HEAD
    if not os.path.exists(cfg.data.paths["MNIST"]):
        os.makedirs(cfg.data.paths["MNIST"])
    mnist_train = datasets.MNIST(cfg.data.paths["MNIST"], train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(cfg.data.paths["MNIST"], train=False, download=True, transform=transform)
    data = torch.cat([mnist_train.data, mnist_test.data], dim=0)
    labels = torch.cat([mnist_train.targets, mnist_test.targets], dim=0)
    return  torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype= torch.float32)  

def load_usps():
    # Can be downloaded here: https://github.com/marionmari/Graph_stuff/tree/master/usps_digit_data
    dir_path = Path(cfg.data.paths["UPS"])
    file_path = dir_path / 'usps_resampled.mat'

    # Ensure the directory exists
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
=======
    minist_train = datasets.MNIST(root=cfg.data.paths["MNIST"], train=True, download=True, transform=transform)
    minist_test = datasets.MNIST(root=cfg.data.paths["MNIST"], train=False, download=True, transform=transform)
    data = []
    labels = []
    for train, label in minist_train:
        data.append(train.view(train.size(0), -1))
        labels.append(label)
    for test, label in minist_test:
        data.append(test.view(test.size(0), -1))
        labels.append(label)
    data = torch.cat(data, dim=0)
    labels = torch.cat(labels,dim=0)
    return data, labels

def load_usps():
    # Can be downloaded here: https://github.com/marionmari/Graph_stuff/tree/master/usps_digit_data
    file_path = Path(cfg.data.paths["UPS"])

>>>>>>> 5bb5ea7211dbcb3a74f00159969a4f3ae16c9d14
    
    if not file_path.exists():
        import requests
        response = requests.get(
            "https://github.com/marionmari/Graph_stuff/blob/master/usps_digit_data/usps_resampled.mat?raw=true")
<<<<<<< HEAD
        with open(file_path, 'wb') as f:
            f.write(response.content)
=======
        f = open(file_path, 'wb+')
        f.write(response.content)
        f.close()
>>>>>>> 5bb5ea7211dbcb3a74f00159969a4f3ae16c9d14

    data_mat = loadmat(file_path)

    data = torch.cat([torch.tensor(data_mat['train_patterns'].T), torch.tensor(data_mat['test_patterns'].T)], 0)
    labels = torch.argmax(torch.cat([torch.tensor(data_mat['train_labels'].T), torch.tensor(data_mat['test_labels'].T)], 0), 1)
<<<<<<< HEAD
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
=======
    return data.type(torch.float32), labels.type(torch.int32)

def reuters_dataset():
    nltk.download('reuters')
    docs = reuters.fileids()
    categories = reuters.categories()
    docs_list = []
    labels_list = []
    for i in docs:
        docs_list.append(reuters.raw(i))
        labels_list.append(reuters.categories(i))
    
    # train_docs, test_docs, train_labels, test_labels = train_test_split(docs_list, labels_list, test_size=)
    #  need to determine the test_size
    return docs_list, labels_list

def fashion_minist():
>>>>>>> 5bb5ea7211dbcb3a74f00159969a4f3ae16c9d14
    fashion_mnist_train = datasets.FashionMNIST(cfg.data.paths["Fashion_MNIST"], train=True, download=True, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(cfg.data.paths["Fashion_MNIST"], train=False, download=True, transform=transform)
    data = torch.cat([fashion_mnist_train.data, fashion_mnist_test.data], dim=0)
    labels = torch.cat([fashion_mnist_train.targets, fashion_mnist_test.targets], dim=0)
<<<<<<< HEAD
    return  torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype= torch.float32)  
=======
    return data,labels    
>>>>>>> 5bb5ea7211dbcb3a74f00159969a4f3ae16c9d14
