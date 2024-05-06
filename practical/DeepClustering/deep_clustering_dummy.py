import torch


def hello_world(string: str) -> str:
    return "Hello " + string


class MiniBatchKMeansVan:
    def __init__(self, k: int, batch_size: int, iterations: int, random_state: int):
        if not isinstance(k, int):
            raise TypeError("k should be of type integer")
        if not isinstance(batch_size, int):
            raise TypeError("batch_size should be of type integer")
        if not isinstance(iterations, int):
            raise TypeError("iterations should be of type integer")
        if k < 1:
            raise ValueError("number of clusters must be bigger than 0")
        if batch_size < 1:
            raise ValueError("batch size must be bigger than 0")
        if iterations < 1:
            raise ValueError("iteration must be bigger than 0")
        if random_state is not None:
            torch.manual_seed(random_state)

        self.k = k
        self.batch_size = batch_size
        self.iterations = iterations

    def distance(self, batch: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        return torch.argmin(torch.cdist(batch, centroids), dim=1)

    def fit(self, x: torch.Tensor) -> torch.Tensor:
        center_counts = torch.zeros(self.k)
        centroids = x[torch.randint(0, len(x), (self.k,))]
        print(centroids)

        for _ in range(self.iterations):
            batch_index = torch.randint(0, len(x), (self.batch_size,))
            batch = x[batch_index]
            print(batch)
            distance = self.distance(batch, centroids)
            print(distance)

            for index, point in enumerate(batch):
                center = distance[index]
                center_counts[center] += 1
                learning_rate = 1 / center_counts[center]
                centroids[center] = (1 - learning_rate) * centroids[center] + learning_rate * point

        return centroids
