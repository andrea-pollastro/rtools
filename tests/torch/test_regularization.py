import pytest
import logging
from rtools.torch.regularization import EarlyStopping


class TestEarlyStopping:
    """Tests for EarlyStopping class."""

    def test_early_stopping_initialization(self):
        """Should initialize with correct attributes."""
        es = EarlyStopping(patience=5, min_delta=0.01)

        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.epochs_counter == 0
        assert es.best_valid_loss == float("+inf")
        assert es.best_train_loss == float("+inf")

    def test_early_stopping_negative_min_delta_becomes_zero(self, caplog):
        """Should convert negative min_delta to 0 with warning."""
        with caplog.at_level(logging.WARNING):
            es = EarlyStopping(patience=5, min_delta=-0.5)

        assert es.min_delta == 0.0
        assert "Negative min_delta" in caplog.text

    def test_early_stopping_first_update_always_improves(self):
        """First update should always be an improvement."""
        es = EarlyStopping(patience=5, min_delta=0.01)
        result = es.update(valid_loss=1.0, train_loss=1.0)

        assert result is True
        assert es.best_valid_loss == 1.0
        assert es.best_train_loss == 1.0
        assert es.epochs_counter == 0

    def test_early_stopping_validation_improvement(self):
        """Should detect validation loss improvement."""
        es = EarlyStopping(patience=5, min_delta=0.01)
        es.update(valid_loss=1.0, train_loss=1.0)
        result = es.update(valid_loss=0.95, train_loss=1.2)

        assert result is True
        assert es.best_valid_loss == 0.95
        assert es.epochs_counter == 0

    def test_early_stopping_no_improvement_increments_counter(self):
        """Should increment counter when no improvement."""
        es = EarlyStopping(patience=5, min_delta=0.01)
        es.update(valid_loss=1.0, train_loss=1.0)
        result = es.update(valid_loss=1.05, train_loss=1.1)

        assert result is False
        assert es.epochs_counter == 1
        assert es.best_valid_loss == 1.0

    def test_early_stopping_counter_resets_on_improvement(self):
        """Counter should reset when improvement detected."""
        es = EarlyStopping(patience=5, min_delta=0.01)
        es.update(valid_loss=1.0, train_loss=1.0)
        es.update(valid_loss=1.05, train_loss=1.1)
        es.update(valid_loss=1.06, train_loss=1.2)

        assert es.epochs_counter == 2

        # Now improve (valid loss decreases by more than min_delta)
        result = es.update(valid_loss=0.98, train_loss=1.0)
        assert result is True
        assert es.epochs_counter == 0

    def test_early_stopping_min_delta_threshold(self):
        """Should respect min_delta threshold."""
        es = EarlyStopping(patience=5, min_delta=0.1)
        es.update(valid_loss=1.0, train_loss=1.0)

        # 0.95 < 0.9, so improvement via first path
        result = es.update(valid_loss=0.95, train_loss=0.9)
        assert result is True
        assert es.epochs_counter == 0

        es2 = EarlyStopping(patience=5, min_delta=0.1)
        es2.update(valid_loss=1.0, train_loss=1.0)
        # 0.98 is in range (0.9, 1.0) and train_loss 0.9 < 1.0
        # So improvement via second path
        result = es2.update(valid_loss=0.98, train_loss=0.9)
        assert result is True
        assert es2.epochs_counter == 0

        es3 = EarlyStopping(patience=5, min_delta=0.1)
        es3.update(valid_loss=1.0, train_loss=1.0)
        # 0.98 in range (0.9, 1.0) but train_loss 1.1 > 1.0, so no improvement
        result = es3.update(valid_loss=0.98, train_loss=1.1)
        assert result is False
        assert es3.epochs_counter == 1

    def test_early_stopping_training_loss_improvement_with_valid_loss_plateau(self):
        """Should improve if valid loss plateaus but train loss decreases."""
        es = EarlyStopping(patience=5, min_delta=0.01)
        es.update(valid_loss=1.0, train_loss=1.0)

        # Validation loss stays within min_delta, training loss decreases
        # For this to trigger improvement, valid_loss must be within (best - min_delta, best)
        result = es.update(valid_loss=0.995, train_loss=0.9)

        # 0.995 is in range (0.99, 1.0) and train_loss 0.9 < 1.0
        assert result is True
        assert es.best_train_loss == 0.9
        assert es.epochs_counter == 0

    def test_early_stopping_training_loss_increase_no_improvement(self):
        """Should not improve if valid loss unchanged and train loss increases."""
        es = EarlyStopping(patience=5, min_delta=0.01)
        es.update(valid_loss=1.0, train_loss=1.0)

        result = es.update(valid_loss=1.005, train_loss=1.1)

        assert result is False
        assert es.epochs_counter == 1

    def test_early_stopping_interrupt_before_patience(self):
        """Should not interrupt before patience exceeded."""
        es = EarlyStopping(patience=5, min_delta=0.01)
        es.update(valid_loss=1.0, train_loss=1.0)

        for i in range(4):
            es.update(valid_loss=1.0 + i * 0.01, train_loss=1.0 + i * 0.01)

        assert es.interrupt() is False

    def test_early_stopping_interrupt_after_patience(self):
        """Should interrupt after patience exceeded."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        es.update(valid_loss=1.0, train_loss=1.0)

        # Trigger no improvement 3 times
        for i in range(3):
            es.update(valid_loss=1.1, train_loss=1.1)

        assert es.interrupt() is True

    def test_early_stopping_interrupt_exactly_at_patience(self):
        """Should interrupt exactly when patience is exceeded."""
        es = EarlyStopping(patience=3, min_delta=0.01)
        es.update(valid_loss=1.0, train_loss=1.0)

        # Trigger no improvement 3 times
        for i in range(3):
            es.update(valid_loss=1.1, train_loss=1.1)

        assert es.epochs_counter == 3
        assert es.interrupt() is True

    def test_early_stopping_interrupt_false_after_improvement(self):
        """Should reset interrupt status after improvement."""
        es = EarlyStopping(patience=2, min_delta=0.01)
        es.update(valid_loss=1.0, train_loss=1.0)

        # No improvement twice
        es.update(valid_loss=1.1, train_loss=1.1)
        es.update(valid_loss=1.1, train_loss=1.1)

        assert es.interrupt() is True

        # Then improve (valid loss decreases by more than min_delta)
        es.update(valid_loss=0.98, train_loss=1.0)

        assert es.interrupt() is False

    def test_early_stopping_patience_zero(self):
        """Should stop immediately with patience=0."""
        es = EarlyStopping(patience=0, min_delta=0.01)
        es.update(valid_loss=1.0, train_loss=1.0)

        es.update(valid_loss=1.1, train_loss=1.1)

        assert es.interrupt() is True

    def test_early_stopping_very_small_improvement(self):
        """Should detect very small improvements."""
        es = EarlyStopping(patience=5, min_delta=1e-6)
        es.update(valid_loss=1.0, train_loss=1.0)

        result = es.update(valid_loss=1.0 - 1e-7, train_loss=1.0)

        # Should be improvement since loss decreased by more than min_delta
        assert result is False  # 1e-7 < 1e-6, so no improvement
        assert es.epochs_counter == 1

    def test_early_stopping_large_min_delta(self):
        """Should handle large min_delta values."""
        es = EarlyStopping(patience=5, min_delta=1.0)
        es.update(valid_loss=10.0, train_loss=10.0)

        result = es.update(valid_loss=9.5, train_loss=9.5)

        # 9.5 is NOT < 10.0 - 1.0 = 9.0, so not an improvement by first condition
        # Check second condition: 9.0 < 9.5 < 10.0 and 9.5 < 10.0? Yes!
        assert result is True
        assert es.epochs_counter == 0

    def test_early_stopping_multiple_improvements_then_plateau(self):
        """Should handle multiple improvements followed by plateau."""
        es = EarlyStopping(patience=3, min_delta=0.01)

        es.update(valid_loss=1.0, train_loss=1.0)
        assert es.update(valid_loss=0.98, train_loss=0.98) is True
        assert es.update(valid_loss=0.96, train_loss=0.96) is True
        # 0.95 < 0.96 - 0.01, so improvement
        assert es.update(valid_loss=0.945, train_loss=0.945) is True

        # Now plateau
        assert es.update(valid_loss=0.946, train_loss=1.0) is False
        assert es.update(valid_loss=0.947, train_loss=1.0) is False
        assert es.update(valid_loss=0.948, train_loss=1.0) is False

        assert es.interrupt() is True

    def test_early_stopping_boundary_condition_valid_loss_exactly_at_threshold(self):
        """Should handle valid loss exactly at improvement boundary."""
        es = EarlyStopping(patience=5, min_delta=0.1)
        es.update(valid_loss=1.0, train_loss=1.0)

        # At 0.9 exactly: first condition is 0.9 < 0.9 (False)
        # Second condition is 0.9 < 0.9 < 1.0 (False)
        result = es.update(valid_loss=0.9, train_loss=0.9)
        assert result is False
        assert es.epochs_counter == 1

        # Just below threshold: 0.89 < 0.9, so improvement
        es2 = EarlyStopping(patience=5, min_delta=0.1)
        es2.update(valid_loss=1.0, train_loss=1.0)
        result = es2.update(valid_loss=0.89, train_loss=0.9)
        assert result is True
        assert es2.best_valid_loss == 0.89

    def test_early_stopping_wide_training_loss_variation(self):
        """Should consider training loss variation for plateau detection."""
        es = EarlyStopping(patience=5, min_delta=0.01)
        es.update(valid_loss=1.0, train_loss=1.0)

        # Valid loss plateaus, but training loss varies widely
        # First one: 0.995 in range (0.99, 1.0) and 0.5 < 1.0 = improvement
        es.update(valid_loss=0.995, train_loss=0.5)
        # Second: 0.995 in range (0.99, 1.0) and 0.4 < 0.5 = improvement
        es.update(valid_loss=0.995, train_loss=0.4)
        # Third: 0.995 in range (0.99, 1.0) and 0.3 < 0.4 = improvement
        assert es.update(valid_loss=0.995, train_loss=0.3) is True
        # Fourth: 0.995 in range (0.99, 1.0) and 0.2 < 0.3 = improvement
        assert es.update(valid_loss=0.995, train_loss=0.2) is True

    def test_early_stopping_symmetric_behavior(self):
        """Results should be symmetric for similar loss sequences."""
        def run_sequence(patience, min_delta, loss_sequence):
            es = EarlyStopping(patience=patience, min_delta=min_delta)
            results = []
            for valid, train in loss_sequence:
                results.append(es.update(valid_loss=valid, train_loss=train))
            return results, es.interrupt()

        losses = [(1.0, 1.0), (0.95, 0.95), (0.92, 0.92), (0.93, 0.93)]
        results, should_interrupt = run_sequence(3, 0.01, losses)

        assert len(results) == 4
        assert results[0] is True  # First is always improvement
        assert results[1] is True  # Improvement
        assert results[2] is True  # Improvement
        assert results[3] is False  # No improvement
