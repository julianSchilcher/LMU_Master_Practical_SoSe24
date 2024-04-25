import torch


class MiniBatchKMeans:
    def __init__(self, k: int, batch_size: int, iterations: int):

        if k <= 0:
            raise ValueError("k must be greater than 0")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        if iterations < 0:
            raise ValueError("iterations must be greater than or equal to 0")

        self.k = k
        self.batch_size = batch_size
        self.iterations = iterations

    def __calculate_distance__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.sum((a - b) ** 2))

    def __get_random_indices__(self, k: int, X: torch.Tensor) -> torch.Tensor:
        return torch.randperm(len(X))[:k]

    def fit(self, X: torch.Tensor) -> torch.Tensor:
        clusters = X[self.__get_random_indices__(self.k, X)]
        per_center_counts = torch.zeros(self.k)

        for _ in range(self.iterations):

            mini_batch_indices = self.__get_random_indices__(self.batch_size, X)

            cluster_asignments = torch.empty(len(X))
            for id in mini_batch_indices:
                cluster_asignments[id] = torch.argmin(
                    torch.tensor(
                        [self.__calculate_distance__(X[id], c) for c in clusters]
                    )
                )

            for id in mini_batch_indices:
                cluster_id = int(cluster_asignments[id])
                per_center_counts[cluster_id] += 1
                lr = 1 / per_center_counts[cluster_id]
                clusters[cluster_id] = (1 - lr) * clusters[cluster_id] + lr * X[id]

        return clusters
















