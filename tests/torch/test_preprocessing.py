import torch
import pytest
from torch.utils.data import Dataset
from rtools.torch.preprocessing import StandardizedDataset


class ToyDataset(Dataset):
    """Simple dataset that returns (features, label) tuples."""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class TestStandardizedDataset:
    """Tests for StandardizedDataset wrapper."""

    def test_standardized_dataset_length_matches_base_dataset(self):
        """Length should match the underlying dataset."""
        features = torch.randn(10, 5)
        labels = torch.arange(10)
        dataset = ToyDataset(features, labels)
        mean = torch.zeros(5)
        std = torch.ones(5)

        std_dataset = StandardizedDataset(dataset, mean, std)

        assert len(std_dataset) == len(dataset)
        assert len(std_dataset) == 10

    def test_standardized_dataset_returns_tuple_structure(self):
        """Should preserve tuple structure (features, *rest)."""
        features = torch.randn(5, 3)
        labels = torch.arange(5)
        dataset = ToyDataset(features, labels)
        mean = torch.zeros(3)
        std = torch.ones(3)

        std_dataset = StandardizedDataset(dataset, mean, std)
        result = std_dataset[0]

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)

    def test_standardized_dataset_applies_standardization(self):
        """Should apply (x - mean) / std transformation."""
        features = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        labels = torch.tensor([0, 1])
        dataset = ToyDataset(features, labels)
        mean = torch.tensor([1.0, 2.0, 3.0])
        std = torch.tensor([1.0, 1.0, 1.0])

        std_dataset = StandardizedDataset(dataset, mean, std)
        x, label = std_dataset[0]

        expected = torch.tensor([0.0, 0.0, 0.0])
        assert torch.allclose(x, expected)

    def test_standardized_dataset_with_nonunit_std(self):
        """Should correctly apply non-unit standard deviations."""
        features = torch.tensor([[2.0, 4.0]])
        labels = torch.tensor([0])
        dataset = ToyDataset(features, labels)
        mean = torch.tensor([1.0, 2.0])
        std = torch.tensor([2.0, 4.0])

        std_dataset = StandardizedDataset(dataset, mean, std)
        x, _ = std_dataset[0]

        expected = torch.tensor([0.5, 0.5])
        assert torch.allclose(x, expected)

    def test_standardized_dataset_guards_against_zero_std(self):
        """Should replace zero std with ones to avoid division by zero."""
        features = torch.tensor([[1.0, 2.0]])
        labels = torch.tensor([0])
        dataset = ToyDataset(features, labels)
        mean = torch.tensor([1.0, 1.0])
        std_with_zero = torch.tensor([1.0, 0.0])

        std_dataset = StandardizedDataset(dataset, mean, std_with_zero)

        # After guards, std should have 1.0 instead of 0.0
        assert std_dataset.std[1] == 1.0

    def test_standardized_dataset_preserves_multiple_return_values(self):
        """Should preserve all elements after the first (features)."""
        class MultiLabelDataset(Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return torch.randn(3), idx, f"label_{idx}"

        dataset = MultiLabelDataset()
        mean = torch.zeros(3)
        std = torch.ones(3)

        std_dataset = StandardizedDataset(dataset, mean, std)
        x, label1, label2 = std_dataset[0]

        assert isinstance(x, torch.Tensor)
        assert label1 == 0
        assert label2 == "label_0"

    def test_standardized_dataset_lazy_evaluation(self):
        """Transformation should be applied lazily at access time."""
        features = torch.randn(10, 5)
        labels = torch.arange(10)
        dataset = ToyDataset(features, labels)
        mean = torch.zeros(5)
        std = torch.ones(5)

        # Creating StandardizedDataset should not modify original data
        std_dataset = StandardizedDataset(dataset, mean, std)
        original_features = dataset.features.clone()

        # Access a sample
        std_dataset[0]

        # Original dataset should not be modified
        assert torch.equal(dataset.features, original_features)

    def test_standardized_dataset_consistent_access(self):
        """Accessing the same index multiple times should give same result."""
        features = torch.randn(5, 3)
        labels = torch.arange(5)
        dataset = ToyDataset(features, labels)
        mean = torch.randn(3)
        std = torch.ones(3) + torch.rand(3)

        std_dataset = StandardizedDataset(dataset, mean, std)
        x1, _ = std_dataset[2]
        x2, _ = std_dataset[2]

        assert torch.equal(x1, x2)

    def test_standardized_dataset_broadcasting(self):
        """Mean and std should broadcast correctly with features."""
        # Features with shape (batch_size, features)
        features = torch.randn(5, 1)
        labels = torch.arange(5)
        dataset = ToyDataset(features, labels)
        mean = torch.tensor([1.0])
        std = torch.tensor([2.0])

        std_dataset = StandardizedDataset(dataset, mean, std)
        x, _ = std_dataset[0]

        # Should have same shape as original features
        assert x.shape == features[0].shape

    def test_standardized_dataset_all_indices(self):
        """Should be able to access all indices without error."""
        n = 20
        features = torch.randn(n, 4)
        labels = torch.arange(n)
        dataset = ToyDataset(features, labels)
        mean = torch.randn(4)
        std = torch.ones(4) + torch.rand(4)

        std_dataset = StandardizedDataset(dataset, mean, std)

        # Access all indices
        for i in range(n):
            x, label = std_dataset[i]
            assert isinstance(x, torch.Tensor)
            assert label == i
