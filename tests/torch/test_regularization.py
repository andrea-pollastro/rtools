import logging
import pytest
from rtools.torch.regularization import EarlyStopping


def test_early_stopping_initialization():
    es = EarlyStopping(patience=5, min_delta=0.01)

    assert es.patience == 5
    assert es.min_delta == 0.01
    assert es.epochs_counter == 0
    assert es.best_valid_loss == float("+inf")
    assert es.best_train_loss == float("+inf")


def test_early_stopping_negative_min_delta_becomes_zero(caplog):
    with caplog.at_level(logging.WARNING):
        es = EarlyStopping(patience=5, min_delta=-0.5)

    assert es.min_delta == 0.0
    assert "Negative min_delta" in caplog.text


def test_early_stopping_validation_improvement_resets_counter():
    es = EarlyStopping(patience=5, min_delta=0.01)
    es.update(valid_loss=1.0, train_loss=1.0)
    result = es.update(valid_loss=0.95, train_loss=1.2)

    assert result is True
    assert es.best_valid_loss == 0.95
    assert es.epochs_counter == 0


def test_early_stopping_interrupt_after_patience():
    es = EarlyStopping(patience=3, min_delta=0.01)
    es.update(valid_loss=1.0, train_loss=1.0)

    for _ in range(3):
        es.update(valid_loss=1.2, train_loss=1.2)

    assert es.interrupt() is True
