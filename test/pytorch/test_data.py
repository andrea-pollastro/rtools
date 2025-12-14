import numpy as np
import pytest
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split as tts
from research_utils.pytorch.data import (
    train_test_split, 
    compute_mean_std, 
    StandardizedDataset
)


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


########################################

class TupleDataset(Dataset):
    """
    Returns (x, y, meta) where x is a tensor and the rest must be preserved.
    x is generated deterministically from idx.
    """
    def __init__(self, n: int, feature_shape=(5,), dtype=torch.float32, constant_feature=False):
        self.n = n
        self.feature_shape = feature_shape
        self.dtype = dtype
        self.constant_feature = constant_feature

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # Deterministic, non-random features
        # Shape: feature_shape
        base = torch.arange(int(torch.tensor(self.feature_shape).prod().item()), dtype=torch.float32)
        x = base.reshape(self.feature_shape) + float(idx)

        # Optional: make last feature constant across all samples (std=0 along that feature)
        if self.constant_feature:
            # Set one feature element constant for all samples
            x = x.clone()
            x.view(-1)[-1] = 7.0

        x = x.to(self.dtype)

        y = idx % 3
        meta = {"idx": idx, "tag": "keep_me"}
        return x, y, meta


def _stack_all_x(dataset: Dataset) -> torch.Tensor:
    xs = []
    for i in range(len(dataset)):  # type: ignore
        x, *_ = dataset[i]
        xs.append(x.float().unsqueeze(0))
    return torch.cat(xs, dim=0)


def _finite(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()  # type: ignore


@pytest.mark.parametrize("feature_shape", [(7,), (3, 4)])
@pytest.mark.parametrize("batch_size", [1, 2, 7, 8, 1024])
def test_compute_mean_std_matches_torch_ground_truth(feature_shape, batch_size):
    ds = TupleDataset(n=23, feature_shape=feature_shape, dtype=torch.float32)

    mean, std = compute_mean_std(ds, batch_size=batch_size, num_workers=0)

    X = _stack_all_x(ds)
    # correction=0 => population std, same as unbiased=False
    std_gt, mean_gt = torch.std_mean(X, dim=0, correction=0)

    assert torch.allclose(mean, mean_gt, rtol=0, atol=1e-6)
    assert torch.allclose(std, std_gt, rtol=0, atol=1e-6)


def test_compute_mean_std_is_deterministic():
    ds = TupleDataset(n=50, feature_shape=(10,), dtype=torch.float32)

    mean1, std1 = compute_mean_std(ds, batch_size=7, num_workers=0)
    mean2, std2 = compute_mean_std(ds, batch_size=7, num_workers=0)

    assert torch.equal(mean1, mean2)
    assert torch.equal(std1, std2)


def test_compute_mean_std_handles_int_inputs_returns_float_tensors():
    ds = TupleDataset(n=10, feature_shape=(5,), dtype=torch.int64)

    mean, std = compute_mean_std(ds, batch_size=4, num_workers=0)

    assert isinstance(mean, torch.Tensor)
    assert isinstance(std, torch.Tensor)
    assert mean.dtype.is_floating_point
    assert std.dtype.is_floating_point


def test_compute_mean_std_works_on_subset():
    base = TupleDataset(n=100, feature_shape=(6,), dtype=torch.float32)
    sub = Subset(base, list(range(10, 40)))

    mean, std = compute_mean_std(sub, batch_size=8, num_workers=0)

    X = _stack_all_x(sub)
    std_gt, mean_gt = torch.std_mean(X, dim=0, correction=0)

    assert torch.allclose(mean, mean_gt, rtol=0, atol=1e-6)
    assert torch.allclose(std, std_gt, rtol=0, atol=1e-6)


def test_compute_mean_std_constant_feature_has_zero_std():
    ds = TupleDataset(n=30, feature_shape=(5,), dtype=torch.float32, constant_feature=True)

    mean, std = compute_mean_std(ds, batch_size=6, num_workers=0)

    # The last element is constant (7.0) -> std must be exactly 0 (up to floating precision)
    assert abs(std.view(-1)[-1].item()) < 1e-12
    assert abs(mean.view(-1)[-1].item() - 7.0) < 1e-6


@pytest.mark.parametrize("feature_shape", [(9,), (2, 5)])
def test_standardized_dataset_preserves_non_feature_fields(feature_shape):
    ds = TupleDataset(n=12, feature_shape=feature_shape, dtype=torch.float32)
    mean, std = compute_mean_std(ds, batch_size=4)

    sds = StandardizedDataset(ds, mean, std)

    x0, y0, meta0 = ds[3]
    x1, y1, meta1 = sds[3]

    assert y1 == y0
    assert meta1 == meta0


def test_standardized_dataset_outputs_are_finite_even_with_zero_std_feature():
    ds = TupleDataset(n=25, feature_shape=(5,), dtype=torch.float32, constant_feature=True)
    mean, std = compute_mean_std(ds, batch_size=5)

    sds = StandardizedDataset(ds, mean, std)

    for i in [0, 5, 10, 24]:
        x, *_ = sds[i]
        assert _finite(x), "Standardized features contain NaN/Inf"


def test_standardized_dataset_matches_manual_transform_exactly():
    ds = TupleDataset(n=20, feature_shape=(8,), dtype=torch.float32)
    mean, std = compute_mean_std(ds, batch_size=7)

    sds = StandardizedDataset(ds, mean, std)

    idx = 11
    x_raw, y_raw, meta_raw = ds[idx]
    x_std, y_std, meta_std = sds[idx]

    std_safe = torch.where(std == 0, torch.ones_like(std), std)
    x_expected = (x_raw.float() - mean) / std_safe

    assert y_std == y_raw
    assert meta_std == meta_raw
    assert torch.allclose(x_std, x_expected, rtol=0, atol=1e-6)


def test_standardized_dataset_fit_on_train_gives_zero_mean_unit_std_on_train():
    # Severe property check: if you fit on ds, standardized ds should have mean~0 and std~1
    ds = TupleDataset(n=60, feature_shape=(6,), dtype=torch.float32)
    mean, std = compute_mean_std(ds, batch_size=9)

    sds = StandardizedDataset(ds, mean, std)

    X = _stack_all_x(sds)
    std_z, mean_z = torch.std_mean(X, dim=0, correction=0)

    # For numerical reasons, allow tiny tolerances
    assert torch.allclose(mean_z, torch.zeros_like(mean_z), rtol=0, atol=1e-5)
    # Features with original std==0 become constant 0 after transform (since std_safe=1)
    # Here we don't have constant features, so std should be ~1
    assert torch.allclose(std_z, torch.ones_like(std_z), rtol=0, atol=1e-5)


def test_standardized_dataset_works_on_subset_and_keeps_base_indexing():
    base = TupleDataset(n=100, feature_shape=(4,), dtype=torch.float32)
    sub = Subset(base, [2, 5, 9, 20, 33, 70])

    mean, std = compute_mean_std(sub, batch_size=3)
    ssub = StandardizedDataset(sub, mean, std)

    # Compare a couple of elements explicitly
    for j in [0, 3, 5]:
        x_raw, y_raw, meta_raw = sub[j] # type: ignore
        x_std, y_std, meta_std = ssub[j]
        std_safe = torch.where(std == 0, torch.ones_like(std), std)
        assert torch.allclose(x_std, (x_raw.float() - mean) / std_safe, atol=1e-6)
        assert y_std == y_raw
        assert meta_std == meta_raw


def test_standardized_dataset_len_matches_wrapped_dataset():
    ds = TupleDataset(n=17, feature_shape=(5,), dtype=torch.float32)
    mean, std = compute_mean_std(ds, batch_size=4)
    sds = StandardizedDataset(ds, mean, std)

    assert len(sds) == len(ds)


def test_standardized_dataset_in_dataloader_batches_correctly():
    ds = TupleDataset(n=32, feature_shape=(3,), dtype=torch.float32)
    mean, std = compute_mean_std(ds, batch_size=8)
    sds = StandardizedDataset(ds, mean, std)

    loader = DataLoader(sds, batch_size=16, shuffle=False, num_workers=0)
    xb, yb, metab = next(iter(loader))

    assert xb.shape == (16, 3)
    assert yb.shape == (16,)
    assert isinstance(metab, dict)
    assert _finite(xb)


def test_standardized_dataset_does_not_modify_underlying_data():
    ds = TupleDataset(n=10, feature_shape=(5,), dtype=torch.float32)
    mean, std = compute_mean_std(ds, batch_size=5)
    sds = StandardizedDataset(ds, mean, std)

    x_before, *_ = ds[0]
    _ = sds[0]
    x_after, *_ = ds[0]

    assert torch.equal(x_before, x_after), "Underlying dataset appears mutated"