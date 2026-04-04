"""Tests for checkpoint management utilities."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
from rtools.torch.checkpointing import CheckpointManager


class TinyModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


def test_checkpoint_manager_save_and_load():
    """Test basic save and load functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        manager = CheckpointManager(tmpdir)

        manager.save(model, optimizer, epoch=1)

        ckpt_files = list(Path(tmpdir).glob("*.pt"))
        assert len(ckpt_files) == 1

        # Load and verify
        new_model = TinyModel()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
        metadata = manager.load(new_model, new_optimizer, epoch=1)
        assert metadata == {}


def test_checkpoint_manager_save_with_metadata():
    """Test saving and retrieving metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        manager = CheckpointManager(tmpdir)

        manager.save(model, optimizer, epoch=5, loss=0.42, accuracy=0.95)

        new_model = TinyModel()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
        metadata = manager.load(new_model, new_optimizer, epoch=5)

        assert metadata["loss"] == 0.42
        assert metadata["accuracy"] == 0.95


def test_checkpoint_manager_multiple_epochs():
    """Test saving multiple checkpoints at different epochs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        manager = CheckpointManager(tmpdir)

        for epoch in [1, 5, 10]:
            manager.save(model, optimizer, epoch=epoch)

        ckpt_files = list(Path(tmpdir).glob("*.pt"))
        assert len(ckpt_files) == 3

        # Verify each can be loaded
        for epoch in [1, 5, 10]:
            new_model = TinyModel()
            new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
            manager.load(new_model, new_optimizer, epoch=epoch)


def test_checkpoint_manager_save_without_optimizer():
    """Test saving without optimizer."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        manager = CheckpointManager(tmpdir)

        manager.save(model, epoch=1)

        ckpt_files = list(Path(tmpdir).glob("*.pt"))
        assert len(ckpt_files) == 1

        # Load without optimizer
        new_model = TinyModel()
        metadata = manager.load(new_model, epoch=1)
        assert metadata == {}


def test_checkpoint_manager_states_preserved():
    """Test that model and optimizer states are properly preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Train for one step to modify state
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        loss_fn = torch.nn.MSELoss()

        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        # Save
        manager = CheckpointManager(tmpdir)
        manager.save(model, optimizer, epoch=1)

        # Create new model and load
        new_model = TinyModel()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
        manager.load(new_model, new_optimizer, epoch=1)

        # Verify states match
        old_params = list(model.parameters())
        new_params = list(new_model.parameters())
        for old_p, new_p in zip(old_params, new_params):
            assert torch.allclose(old_p, new_p)


def test_checkpoint_manager_load_missing_raises_error():
    """Test that loading non-existent checkpoint raises FileNotFoundError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = TinyModel()

        with pytest.raises(FileNotFoundError):
            manager.load(model, epoch=999)


def test_checkpoint_manager_load_requires_epoch_or_path():
    """Test that load requires either epoch or checkpoint_path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = TinyModel()

        with pytest.raises(ValueError, match="Must provide either epoch or checkpoint_path"):
            manager.load(model)

