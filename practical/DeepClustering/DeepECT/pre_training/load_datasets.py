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
transform = transforms.ToTensor()
def minist_dataset():
    minist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    minist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    return minist_train, minist_test

def load_usps():
    # Can be downloaded here: https://github.com/marionmari/Graph_stuff/tree/master/usps_digit_data
    file_path = Path(f"{dataset_dir}/usps_resampled.mat")

    if not file_path.exists():
        import requests
        response = requests.get(
            "https://github.com/marionmari/Graph_stuff/blob/master/usps_digit_data/usps_resampled.mat?raw=true")
        f = open(file_path, 'wb+')
        f.write(response.content)
        f.close()

    data_mat = loadmat(file_path)

    data = torch.cat([data_mat['train_patterns'].T, data_mat['test_patterns'].T], 0)
    labels = torch.argmax(torch.cat([data_mat['train_labels'].T, data_mat['test_labels'].T], 0), 1)
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

def fashion_minist():
    fashion_minist_train = datasets.FashionMNIST("./data", train=True, download=True, transform=transform)
    fashion_minist_test = datasets.FashionMNIST("./data", train=False, download=True, transform=transform)
    
    return fashion_minist_train,fashion_minist_test    