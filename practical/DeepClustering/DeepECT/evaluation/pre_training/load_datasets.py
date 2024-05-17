import torch
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

transform = transforms.ToTensor()
cfg = config.get_config()

def mnist_dataset():
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

    
    if not file_path.exists():
        import requests
        response = requests.get(
            "https://github.com/marionmari/Graph_stuff/blob/master/usps_digit_data/usps_resampled.mat?raw=true")
        f = open(file_path, 'wb+')
        f.write(response.content)
        f.close()

    data_mat = loadmat(file_path)

    data = torch.cat([torch.tensor(data_mat['train_patterns'].T), torch.tensor(data_mat['test_patterns'].T)], 0)
    labels = torch.argmax(torch.cat([torch.tensor(data_mat['train_labels'].T), torch.tensor(data_mat['test_labels'].T)], 0), 1)
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
    fashion_mnist_train = datasets.FashionMNIST(cfg.data.paths["Fashion_MNIST"], train=True, download=True, transform=transform)
    fashion_mnist_test = datasets.FashionMNIST(cfg.data.paths["Fashion_MNIST"], train=False, download=True, transform=transform)
    data = torch.cat([fashion_mnist_train.data, fashion_mnist_test.data], dim=0)
    labels = torch.cat([fashion_mnist_train.targets, fashion_mnist_test.targets], dim=0)
    return data,labels    