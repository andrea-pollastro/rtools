import numpy as np
import pytest
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split as tts
from rtools.torch.model_selection import train_test_split


class ToyDataset(Dataset):
    def __init__(self, n: int):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


def _idx(subset: Subset) -> list[int]:
    assert isinstance(subset, Subset)
    # torch Subset.indices can be list/sequence
    return list(subset.indices)  # type: ignore


def test_raises_on_wrong_type():
    with pytest.raises(TypeError):
        train_test_split([1, 2, 3], test_size=0.2, random_state=0)  # type: ignore


def test_returns_subsets_of_base_dataset_for_dataset_input():
    ds = ToyDataset(10)
    tr, te = train_test_split(ds, test_size=0.3, random_state=0)

    assert isinstance(tr, Subset)
    assert isinstance(te, Subset)
    assert tr.dataset is ds
    assert te.dataset is ds


def test_disjoint_and_full_coverage_for_dataset_input():
    n = 101
    ds = ToyDataset(n)
    tr, te = train_test_split(ds, test_size=0.2, random_state=123, shuffle=True)

    tr_idx = _idx(tr)
    te_idx = _idx(te)

    assert len(set(tr_idx).intersection(te_idx)) == 0, "train/test overlap detected"
    assert sorted(tr_idx + te_idx) == list(range(n)), "missing or duplicated indices"
    assert len(tr_idx) + len(te_idx) == n


def test_exact_match_with_sklearn_indices_for_dataset_input():
    n = 37
    ds = ToyDataset(n)
    indices = np.arange(n)

    exp_tr, exp_te = tts(indices, test_size=0.25, random_state=7, shuffle=True)
    tr, te = train_test_split(ds, test_size=0.25, random_state=7, shuffle=True)

    assert _idx(tr) == exp_tr.tolist()
    assert _idx(te) == exp_te.tolist()


def test_deterministic_given_same_random_state():
    ds = ToyDataset(50)

    tr1, te1 = train_test_split(ds, test_size=0.2, random_state=999, shuffle=True)
    tr2, te2 = train_test_split(ds, test_size=0.2, random_state=999, shuffle=True)

    assert _idx(tr1) == _idx(tr2)
    assert _idx(te1) == _idx(te2)


def test_different_random_state_changes_split_most_of_the_time():
    # Not purely probabilistic: use large n and compare whole index arrays.
    ds = ToyDataset(200)

    tr1, te1 = train_test_split(ds, test_size=0.2, random_state=1, shuffle=True)
    tr2, te2 = train_test_split(ds, test_size=0.2, random_state=2, shuffle=True)

    assert _idx(tr1) != _idx(tr2) or _idx(te1) != _idx(te2)


def test_shuffle_false_preserves_order_like_sklearn():
    n = 20
    ds = ToyDataset(n)
    indices = np.arange(n)

    exp_tr, exp_te = tts(indices, test_size=0.3, shuffle=False)
    tr, te = train_test_split(ds, test_size=0.3, shuffle=False)

    assert _idx(tr) == exp_tr.tolist()
    assert _idx(te) == exp_te.tolist()
    assert _idx(tr) == list(range(len(_idx(tr))))
    assert _idx(te) == list(range(len(_idx(tr)), n))


def test_subset_input_maps_indices_correctly_to_base_dataset():
    base = ToyDataset(100)
    subset_indices = list(range(10, 60))  # contiguous slice-like subset
    sub = Subset(base, subset_indices)

    tr, te = train_test_split(sub, test_size=0.2, random_state=0, shuffle=True)

    # Must always reference base dataset (not the Subset wrapper)
    assert tr.dataset is base
    assert te.dataset is base

    # Must only draw from the subset indices
    tr_idx = _idx(tr)
    te_idx = _idx(te)
    allowed = set(subset_indices)

    assert set(tr_idx).issubset(allowed)
    assert set(te_idx).issubset(allowed)

    # Must cover exactly the subset indices with no overlap
    assert len(set(tr_idx).intersection(te_idx)) == 0
    assert sorted(tr_idx + te_idx) == subset_indices


def test_stratify_behaves_like_sklearn_on_index_labels():
    # We stratify on a label per index. This is the only correct way to stratify here.
    n = 100
    ds = ToyDataset(n)
    indices = np.arange(n)

    # 80 zeros, 20 ones, stratify should preserve proportions in both splits.
    y = np.array([0] * 80 + [1] * 20)

    exp_tr, exp_te = tts(indices, test_size=0.25, random_state=0, stratify=y)
    tr, te = train_test_split(ds, test_size=0.25, random_state=0, stratify=y)

    assert _idx(tr) == exp_tr.tolist()
    assert _idx(te) == exp_te.tolist()

    # Severe: verify class proportions exactly match sklearn result
    tr_y = y[_idx(tr)]
    te_y = y[_idx(te)]
    exp_tr_y = y[exp_tr]
    exp_te_y = y[exp_te]

    assert tr_y.sum() == exp_tr_y.sum()
    assert te_y.sum() == exp_te_y.sum()


def test_raises_on_nested_subset():
    base = ToyDataset(200)
    sub1 = Subset(base, list(range(20, 120)))
    sub2 = Subset(sub1, list(range(10, 50)))

    with pytest.raises(ValueError, match="Nested Subset"):
        train_test_split(sub2, test_size=0.25, random_state=42)


def test_split_sizes_match_sklearn():
    n = 103
    ds = ToyDataset(n)
    indices = np.arange(n)

    exp_tr, exp_te = tts(indices, test_size=0.2, random_state=0, shuffle=True)
    tr, te = train_test_split(ds, test_size=0.2, random_state=0, shuffle=True)

    assert len(_idx(tr)) == len(exp_tr)
    assert len(_idx(te)) == len(exp_te)
