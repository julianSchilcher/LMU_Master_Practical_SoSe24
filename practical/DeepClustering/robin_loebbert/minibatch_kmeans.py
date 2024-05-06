import torch
from torch.utils.data import DataLoader, Dataset, Subset, RandomSampler, TensorDataset
import numpy as np


class MiniBatchKMeans:
    k: int
    b: int
    t: int
    c: torch.Tensor
    rng: torch.Generator
    distance_p_norm: int

    def __init__(
        self,
        k,
        mini_batch_size,
        iterations,
        distance_p_norm=2,
        seed=42,
        device="cpu",
    ) -> None:
        self.k = k
        self.b = mini_batch_size
        self.t = iterations
        self.rng = torch.random.manual_seed(seed)
        self.device = device
        self.distance_p_norm = distance_p_norm

    def fit(self, X: np.ndarray):
        dataset = TensorDataset(torch.tensor(X))
        dataloader = DataLoader(dataset, self.b, shuffle=True, generator=self.rng)
        # init C with k random entries from X
        random_idx = (
            torch.randint(
                high=len(X),
                size=(self.k,),
                device="cpu",
                generator=self.rng,
                requires_grad=False,
            )
            .numpy()
            .tolist()
        )
        self.c = torch.tensor(X[random_idx], device=self.device)
        # zero init per center counts
        v = torch.zeros((self.k,), dtype=torch.long, device=self.device)
        # epochs / iterations
        for i in range(self.t):
            # get random minibatch
            loss_per_epoch = torch.tensor(0.0, device=self.device)
            for batch in dataloader:
                # foreach x in minibatch
                (x,) = batch
                x = x.to(self.device)  # (b, d)
                # get nearest center index: dist = (b, k)
                dist = torch.cdist(x, self.c, p=self.distance_p_norm)
                d = dist.argmin(1)  # (b, 1)
                one_hot = torch.nn.functional.one_hot(d, num_classes=self.k)  # (b, k)
                cum_sum = torch.cumsum(one_hot.T, dim=1)  # (k, b)
                # update the center counts for this center
                v += cum_sum.max(dim=1).values
                # get per center learning rate
                eta = (1 / v).unsqueeze(1)
                eta[eta == float("inf")] = 0
                # take gradient step
                loss = eta * torch.sum(one_hot.T.unsqueeze(2) * x, dim=1)
                loss_per_epoch += torch.sum(loss)
                self.c += (- eta) * self.c + loss
            print(f"loss (epoch {i}):", float(loss_per_epoch.cpu()))
        return self

    def predict(self, X: np.ndarray):
        # dataloader for one iteration of minibatches
        dataset = TensorDataset(torch.tensor(X))
        dataloader = DataLoader(dataset, self.b, shuffle=False)
        results = []
        for batch in dataloader:
            (x,) = batch
            x = x.to(self.device)
            results.append(
                torch.cdist(x, self.c, p=self.distance_p_norm).argmin(1).cpu()
            )
        return torch.cat(results, dim=0).numpy()


if __name__ == "__main__":
    pass
