from pathlib import Path

import nltk
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from clustpy.deep import get_dataloader
from nltk.corpus import reuters
from scipy.io import loadmat

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


def get_augmentation_dataloaders_for_mnist(dataset: np.ndarray, batch_size: int = 256) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    dataset /= 255.0
    mean = dataset.mean()
    std = dataset.std()
    dataset = dataset.reshape(-1, 1, 28, 28)
    dataset = np.tile(dataset, (1, 3, 1, 1))
    # preprocessing functions
    normalize_fn = transforms.Normalize([mean], [std])
    flatten_fn = transforms.Lambda(torch.flatten)
    # augmentation transforms
    transform_list = [
        transforms.ToPILImage(),
        transforms.RandomAffine(degrees=(-16, +16), translate=(0.1, 0.1), shear=(-8, 8), fill=0),
        transforms.ToTensor(),
        normalize_fn,
        flatten_fn
    ]
    aug_transforms = transforms.Compose(transform_list)
    orig_transforms = transforms.Compose([normalize_fn, flatten_fn])
    # pass transforms to dataloader
    augmented_dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=True,
                            ds_kwargs={"aug_transforms_list":[aug_transforms], "orig_transforms_list":[orig_transforms]})
    original_dataloader = get_dataloader(dataset, batch_size=batch_size, shuffle=False,
                        ds_kwargs={"orig_transforms_list":[orig_transforms]})
    return augmented_dataloader, original_dataloader