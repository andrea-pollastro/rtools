import torch
import pytest
import os
from unittest.mock import patch
from io import StringIO
from rtools.torch.training import to_device, machine_summary, RunningStats


class TestToDevice:
    """Tests for to_device function."""

    def test_to_device_single_tensor_to_cpu(self):
        """Should move a single tensor to CPU."""
        tensor = torch.randn(3, 4)
        result = to_device(tensor, 'cpu')

        assert isinstance(result, torch.Tensor)
        assert result.device.type == 'cpu'

    def test_to_device_single_tensor_preserves_values(self):
        """Moving tensor should preserve values."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = to_device(tensor, 'cpu')

        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, tensor)

    def test_to_device_tuple_of_tensors(self):
        """Should convert tuple of tensors to tuple with all on device."""
        t1 = torch.randn(2, 3)
        t2 = torch.randn(4, 5)
        batch = (t1, t2)
        result = to_device(batch, 'cpu')

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(t, torch.Tensor) for t in result)
        assert all(t.device.type == 'cpu' for t in result)

    def test_to_device_list_of_tensors(self):
        """Should convert list of tensors to tuple with all on device."""
        t1 = torch.randn(2, 3)
        t2 = torch.randn(4, 5)
        batch = [t1, t2]
        result = to_device(batch, 'cpu')

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert torch.equal(result[0], t1)
        assert torch.equal(result[1], t2)

    def test_to_device_list_returns_tuple(self):
        """List input should return tuple, not list."""
        batch = [torch.randn(2, 3), torch.randn(4, 5)]
        result = to_device(batch, 'cpu')

        assert isinstance(result, tuple)
        assert not isinstance(result, list)

    def test_to_device_empty_list(self):
        """Should handle empty list/tuple."""
        result = to_device([], 'cpu')
        assert isinstance(result, tuple)
        assert len(result) == 0

    def test_to_device_single_element_list(self):
        """Should handle single-element list."""
        tensor = torch.randn(2, 3)
        result = to_device([tensor], 'cpu')

        assert isinstance(result, tuple)
        assert len(result) == 1
        assert torch.equal(result[0], tensor)

    def test_to_device_mixed_batch_sizes(self):
        """Should handle tensors with different shapes."""
        t1 = torch.randn(2, 3)
        t2 = torch.randn(10)
        t3 = torch.randn(5, 5, 5)
        batch = (t1, t2, t3)
        result = to_device(batch, 'cpu')

        assert len(result) == 3
        assert result[0].shape == (2, 3)
        assert result[1].shape == (10,)
        assert result[2].shape == (5, 5, 5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_to_cuda(self):
        """Should move tensor to CUDA if available."""
        tensor = torch.randn(3, 4)
        result = to_device(tensor, 'cuda')

        assert isinstance(result, torch.Tensor)
        assert result.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_device_batch_to_cuda(self):
        """Should move all tensors in batch to CUDA."""
        batch = [torch.randn(2, 3), torch.randn(4, 5)]
        result = to_device(batch, 'cuda')

        assert isinstance(result, tuple)
        assert all(t.device.type == 'cuda' for t in result)


class TestMachineSummary:
    """Tests for machine_summary function."""

    def test_machine_summary_produces_output(self, capsys, monkeypatch):
        """Should print to stdout."""
        # Mock os.get_terminal_size to avoid OSError in non-interactive environment
        from os import terminal_size
        monkeypatch.setattr(os, 'get_terminal_size', lambda fd=None: terminal_size((80, 24)))
        machine_summary()
        captured = capsys.readouterr()

        assert len(captured.out) > 0
        assert captured.out.startswith('=')

    def test_machine_summary_contains_expected_fields(self, capsys, monkeypatch):
        """Output should contain system information."""
        from os import terminal_size
        monkeypatch.setattr(os, 'get_terminal_size', lambda fd=None: terminal_size((80, 24)))
        machine_summary()
        captured = capsys.readouterr()
        output = captured.out.lower()

        assert "network name" in output or "computer" in output
        assert "machine" in output
        assert "processor" in output or "platform" in output
        assert "operating system" in output

    def test_machine_summary_has_decorative_lines(self, capsys, monkeypatch):
        """Output should have decorative equal sign lines."""
        from os import terminal_size
        monkeypatch.setattr(os, 'get_terminal_size', lambda fd=None: terminal_size((80, 24)))
        machine_summary()
        captured = capsys.readouterr()

        lines = captured.out.split('\n')
        # First and last lines should be decorative
        assert all(c == '=' for c in lines[0].strip())
        assert all(c == '=' for c in lines[-2].strip())

    def test_machine_summary_does_not_raise(self, monkeypatch):
        """Should not raise any exceptions."""
        from os import terminal_size
        monkeypatch.setattr(os, 'get_terminal_size', lambda fd=None: terminal_size((80, 24)))
        try:
            machine_summary()
        except Exception as e:
            pytest.fail(f"machine_summary raised {type(e).__name__}: {e}")


class TestRunningStats:
    """Tests for RunningStats class."""

    def test_running_stats_initialization(self):
        """Should initialize with empty totals."""
        stats = RunningStats(['loss', 'accuracy'])

        assert stats.metrics == ['loss', 'accuracy']
        assert stats.total == {'loss': 0.0, 'accuracy': 0.0}
        assert stats.count == 0

    def test_running_stats_reset(self):
        """Reset should clear totals and count."""
        stats = RunningStats(['metric1', 'metric2'])
        stats.update({'metric1': 1.0, 'metric2': 2.0})
        stats.reset()

        assert stats.total == {'metric1': 0.0, 'metric2': 0.0}
        assert stats.count == 0

    def test_running_stats_update_single_value(self):
        """Should accumulate values correctly."""
        stats = RunningStats(['loss'])
        stats.update({'loss': 5.0})

        assert stats.total['loss'] == 5.0
        assert stats.count == 1

    def test_running_stats_update_multiple_metrics(self):
        """Should update multiple metrics simultaneously."""
        stats = RunningStats(['loss', 'accuracy'])
        stats.update({'loss': 1.5, 'accuracy': 0.85})

        assert stats.total['loss'] == 1.5
        assert stats.total['accuracy'] == 0.85
        assert stats.count == 1

    def test_running_stats_update_accumulates(self):
        """Multiple updates should accumulate."""
        stats = RunningStats(['loss'])
        stats.update({'loss': 1.0})
        stats.update({'loss': 2.0})
        stats.update({'loss': 3.0})

        assert stats.total['loss'] == 6.0
        assert stats.count == 3

    def test_running_stats_update_with_elem_weight(self):
        """Should weight update by elem parameter."""
        stats = RunningStats(['loss'])
        stats.update({'loss': 1.0}, elem=5)

        assert stats.total['loss'] == 5.0
        assert stats.count == 5

    def test_running_stats_mean_single_update(self):
        """Mean should equal the value for single update."""
        stats = RunningStats(['loss'])
        stats.update({'loss': 10.0})

        mean = stats.mean()
        assert mean['loss'] == 10.0

    def test_running_stats_mean_multiple_updates(self):
        """Mean should compute average correctly."""
        stats = RunningStats(['loss'])
        stats.update({'loss': 1.0})
        stats.update({'loss': 2.0})
        stats.update({'loss': 3.0})

        mean = stats.mean()
        assert mean['loss'] == 2.0

    def test_running_stats_mean_with_weights(self):
        """Mean should respect elem weighting."""
        stats = RunningStats(['loss'])
        stats.update({'loss': 10.0}, elem=3)
        stats.update({'loss': 20.0}, elem=2)

        mean = stats.mean()
        # (10*3 + 20*2) / (3+2) = 70/5 = 14
        assert mean['loss'] == 14.0

    def test_running_stats_mean_multiple_metrics(self):
        """Should compute means for all metrics."""
        stats = RunningStats(['loss', 'accuracy'])
        stats.update({'loss': 1.0, 'accuracy': 0.8})
        stats.update({'loss': 2.0, 'accuracy': 0.9})

        mean = stats.mean()
        assert mean['loss'] == 1.5
        assert pytest.approx(mean['accuracy'], rel=1e-6) == 0.85

    def test_running_stats_unregistered_metric_raises(self):
        """Should raise KeyError for unregistered metrics."""
        stats = RunningStats(['loss'])

        with pytest.raises(KeyError, match="'accuracy' was not registered"):
            stats.update({'accuracy': 0.9})

    def test_running_stats_mean_before_update_raises(self):
        """Should raise RuntimeError if mean called before any update."""
        stats = RunningStats(['loss'])

        with pytest.raises(RuntimeError, match="before any update"):
            stats.mean()

    def test_running_stats_multiple_reset_cycles(self):
        """Should handle multiple reset cycles."""
        stats = RunningStats(['loss'])

        # First cycle
        stats.update({'loss': 5.0})
        assert stats.mean()['loss'] == 5.0

        # Reset and second cycle
        stats.reset()
        stats.update({'loss': 10.0})
        assert stats.mean()['loss'] == 10.0

    def test_running_stats_zero_elem_parameter(self):
        """Should handle elem=0 (no change)."""
        stats = RunningStats(['loss'])
        stats.update({'loss': 1.0})
        stats.update({'loss': 999.0}, elem=0)

        mean = stats.mean()
        assert mean['loss'] == 1.0

    def test_running_stats_large_elem_value(self):
        """Should handle large elem values."""
        stats = RunningStats(['loss'])
        stats.update({'loss': 1.0}, elem=1000000)

        assert stats.count == 1000000
        mean = stats.mean()
        assert mean['loss'] == 1.0

    def test_running_stats_partial_metric_update(self):
        """Should allow updating subset of metrics."""
        stats = RunningStats(['loss', 'accuracy'])

        # Only updating one metric should work
        stats.update({'loss': 1.0})

        assert stats.total['loss'] == 1.0
        assert stats.total['accuracy'] == 0.0

    def test_running_stats_empty_metrics_list(self):
        """Should handle empty metrics list."""
        stats = RunningStats([])
        assert stats.metrics == []
        assert stats.total == {}
        assert stats.count == 0

    def test_running_stats_mean_returns_all_metrics(self):
        """Mean should return all registered metrics."""
        stats = RunningStats(['m1', 'm2', 'm3'])
        stats.update({'m1': 1.0, 'm2': 2.0, 'm3': 3.0})
        mean = stats.mean()

        assert set(mean.keys()) == {'m1', 'm2', 'm3'}
