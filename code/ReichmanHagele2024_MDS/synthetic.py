import numpy as np
from sklearn.datasets import make_blobs

class Synthetic:
    def __init__(self, dimensionality: int):
        super().__init__()
        dataset, classes = self.generate_data(dimensionality)
        self.dataset = dataset
        self.classes = classes

        self.class_indices = [0, 1, 2, 3]
        self.classnames_as_string = [0, 1, 2, 3]
        self.dimensionality = dimensionality

        self.n_indices = self.get_classes().flatten().shape[0]
        self.unpicked_indices = np.arange(0, self.get_classes().flatten().shape[0])

    def get_dataset(self) -> np.ndarray:
        dataset = self.dataset
        dataset = np.array(dataset, dtype=float)
        return dataset

    def get_classes(self) -> np.ndarray:
        dataset = self.classes
        np_classes = np.array(dataset, dtype=int)
        return np_classes

    def pick(self, n_samples):
        picked_indices = np.random.choice(self.unpicked_indices, n_samples, replace=False)
        mask = np.isin(self.unpicked_indices, picked_indices, invert=True)
        self.unpicked_indices = self.unpicked_indices[mask]
        return picked_indices

    def draw_samples(self, n_samples: int):
        picked_indices = self.pick(n_samples)
        filtered_data = self.dataset[picked_indices]
        filtered_labels = self.classes[picked_indices]
        return filtered_data, filtered_labels

    def generate_data(self, dimensionality):
        X, y_true = make_blobs(n_samples=1_000_000, cluster_std=4, n_features=dimensionality, centers=4, random_state=0)
        return X, y_true
