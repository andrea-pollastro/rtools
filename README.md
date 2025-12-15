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

### rtools.pytorch.random

**set_seed**  
Sets random seeds for Python, NumPy, and PyTorch to enforce deterministic
behavior across runs.  

---

### rtools.pytorch.model_selection

**train_test_split**  
A PyTorch-friendly equivalent of `sklearn.model_selection.train_test_split`.

It splits a `torch.utils.data.Dataset` or `Subset` into training and test
subsets, preserving indices and avoiding data copying.  
The API mirrors sklearn’s behavior and supports arguments such as
`test_size`, `train_size`, `shuffle`, `random_state`, and `stratify`.

---

### rtools.pytorch.preprocessing

**compute_mean_std**  
Computes feature-wise mean and standard deviation of a PyTorch dataset in a
streaming and memory-efficient way.

The result is equivalent to computing `X.mean()` and
`X.std(unbiased=False)` over the full dataset, but without loading all data
into memory.  

**StandardizedDataset**  
A dataset wrapper that applies feature-wise standardization on-the-fly.

It mimics the behavior of `sklearn.preprocessing.StandardScaler` in a
PyTorch-native way by applying `(x - mean) / std` lazily in `__getitem__`,
without modifying the underlying dataset.  
