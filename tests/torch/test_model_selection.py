import pytest
from torch.utils.data import Dataset, Subset
from rtools.torch.model_selection import train_test_split


class ToyDataset(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


def _idx(subset: Subset) -> list[int]:
    return list(subset.indices)  # type: ignore


def test_raises_on_wrong_type():
    with pytest.raises(TypeError):
        train_test_split([1, 2, 3], test_size=0.2, random_state=0)  # type: ignore


def test_disjoint_and_full_coverage_for_dataset_input():
    n = 100
    ds = ToyDataset(n)
    tr, te = train_test_split(ds, test_size=0.2, random_state=42, shuffle=True)

    tr_idx = _idx(tr)
    te_idx = _idx(te)

    assert len(set(tr_idx).intersection(te_idx)) == 0
    assert sorted(tr_idx + te_idx) == list(range(n))
    assert len(tr_idx) + len(te_idx) == n


def test_subset_input_maps_indices_correctly_to_base_dataset():
    base = ToyDataset(50)
    subset_indices = list(range(10, 30))
    sub = Subset(base, subset_indices)

    tr, te = train_test_split(sub, test_size=0.2, random_state=0, shuffle=True)

    assert tr.dataset is base
    assert te.dataset is base
    assert set(_idx(tr)).issubset(subset_indices)
    assert set(_idx(te)).issubset(subset_indices)
    assert len(set(_idx(tr)).intersection(_idx(te))) == 0
