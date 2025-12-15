# rtools

Personal research utilities.

This package contains small, reusable helpers that I commonly use in experiments and research code.

## Installation

From the repository root:

```bash
git clone https://github.com/andrea-pollastro/rtools.git
cd rtools
pip install -e .
```

or, from GitHub:

```bash
pip install git+https://github.com/andrea-pollastro/rtools.git
```

## Contents

### `rtools.pytorch.random`

- **`set_seed`**  
  Sets random seeds for Python, NumPy, and PyTorch to enforce deterministic
  behavior across runs.

---

### `rtools.pytorch.model_selection`

- **`train_test_split`**  
  PyTorch-friendly equivalent of `sklearn.model_selection.train_test_split`.  
  Splits a `torch.utils.data.Dataset` or `Subset` into train/test subsets,
  preserving indices and avoiding data copying.  
  Supports all the parameters of `sklearn.model_selection.train_test_split`.

---

### `rtools.pytorch.preprocessing`

- **`compute_mean_std`**  
  Computes feature-wise mean and standard deviation of a dataset in a streaming,
  memory-efficient way. Equivalent to `X.mean()` and `X.std(unbiased=False)`
  without loading all data into memory.

- **`StandardizedDataset`**  
  Dataset wrapper that applies feature-wise standardization on-the-fly.  
  It applies `(x - mean) / std` lazily in `__getitem__`, without modifying the underlying dataset.
