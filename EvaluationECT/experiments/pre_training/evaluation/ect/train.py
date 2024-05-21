import torch

class Train:
    def __init__(self, algorithm, dataset_name, data):
        super.__init__(self, Train)
        self.algorithm = algorithm
        self.dataset_name = dataset_name
        self.data = data
        
    def load_model(self):
        
                