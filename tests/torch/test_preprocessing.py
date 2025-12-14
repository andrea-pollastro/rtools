import pytest
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from rtools.torch.preprocessing import (
    compute_mean_std, 
    StandardizedDataset
)


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