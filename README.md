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

### `rtools.torch.random`

- **`set_seed`**  
  Sets random seeds for Python, NumPy, and PyTorch to enforce deterministic
  behavior across runs.

---

### `rtools.torch.model_selection`

- **`train_test_split`**  
  PyTorch-friendly equivalent of `sklearn.model_selection.train_test_split`.  
  Splits a `torch.utils.data.Dataset` or `Subset` into train/test subsets,
  preserving indices and avoiding data copying.  
  Supports all the parameters of `sklearn.model_selection.train_test_split`.

---

### `rtools.torch.preprocessing`

- **`StandardizedDataset`**  
  Dataset wrapper that applies feature-wise standardization on-the-fly.  
  It applies `(x - mean) / std` lazily in `__getitem__`, without modifying the underlying dataset.

---

### `rtools.torch.training`

- **`RunningMean`**  
  Tracks running mean of multiple metrics without storing individual values.  
  Efficiently computes weighted running means for monitoring during training.

- **`History`**  
  Tracks history of metrics as lists for visualization and analysis.  
  Stores complete history and exports to CSV for plotting and reporting.

---

### `rtools.torch.checkpointing`

- **`CheckpointManager`**  
  Save and load PyTorch checkpoints with epoch-based organization.  
  Handles model and optimizer state serialization with optional metadata storage.

---

### `rtools.torch.utils`

- **`to_device()`**  
  Move tensors and nested structures (tuples, lists, dicts) to a specified device.  
  Recursively handles mixed data types, preserving non-tensor objects.

---

### `rtools.torch.regularization`

- **`EarlyStopping`**  
  Early stopping mechanism to interrupt training when improvement plateaus.  
  Monitors both validation and training loss to detect improvement and enforce patience-based stopping.

---

### `rtools.parallel`

- **`parallel_map(func, iterable, n_jobs=None, start_method="spawn")`**  
  Apply a function to items in an iterable using multiprocessing.  
  Equivalent to `multiprocessing.Pool.map()` with configurable worker count and start method.

- **`parallel_starmap(func, iterable, n_jobs=None, start_method="spawn")`**  
  Apply a function to unpacked argument tuples using multiprocessing.  
  Equivalent to `multiprocessing.Pool.starmap()` with configurable worker count and start method.

---

### `rtools.datasets.disentanglement`

- **`DSprites`**  
  The DSprites (Disentanglement Testing Sprites) dataset for evaluating disentangled
  representation learning algorithms. A large dataset of 64x64 images of 2D sprites
  with controlled disentangled factors of variation (shape, scale, orientation, position).  
  Supports both single sample and paired sample modes for contrastive learning.

- **`ThreeDShapes`**  
  The 3DShapes dataset for disentanglement research. It downloads and converts the
  original HDF5 archive, and requires `h5py` when initialized.

