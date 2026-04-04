import torch
from torch.utils.data import Dataset
from rtools.torch.preprocessing import StandardizedDataset


class ToyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def test_standardized_dataset_length_matches_base_dataset():
    features = torch.randn(10, 5)
    labels = torch.arange(10)
    dataset = ToyDataset(features, labels)
    std_dataset = StandardizedDataset(dataset, torch.zeros(5), torch.ones(5))

    assert len(std_dataset) == len(dataset)


def test_standardized_dataset_applies_standardization():
    features = torch.tensor([[1.0, 2.0, 3.0]])
    labels = torch.tensor([0])
    dataset = ToyDataset(features, labels)
    std_dataset = StandardizedDataset(dataset, torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 1.0, 1.0]))

    x, label = std_dataset[0]
    assert torch.allclose(x, torch.zeros(3))
    assert label == 0


def test_standardized_dataset_guards_against_zero_std():
    features = torch.tensor([[1.0, 2.0]])
    labels = torch.tensor([0])
    dataset = ToyDataset(features, labels)
    std_dataset = StandardizedDataset(dataset, torch.tensor([1.0, 1.0]), torch.tensor([1.0, 0.0]))

    assert std_dataset.std[1] == 1.0
