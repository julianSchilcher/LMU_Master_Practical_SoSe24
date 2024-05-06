import numpy as np
import torch


def create_dataset(
    num_centers=2, dataset_size=1000, scattering_factor=0.3, dimension=2
):
    centers = torch.from_numpy(
        np.array([np.random.rand(dimension) * 10 for _ in range(num_centers)])
    )

    dataset = torch.empty((0, dimension))
    for center in centers:
        cluster = center + scattering_factor * torch.randn(
            dataset_size // len(centers), dimension
        )
        dataset = torch.concatenate((dataset, cluster))
    return dataset


