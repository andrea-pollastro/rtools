import logging
logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stopping mechanism to interrupt training when improvement plateaus.

    This class monitors validation and training losses to determine when to stop
    training. It tracks the best validation loss and allows improvement either
    through reduction in validation loss or by reducing training loss while
    maintaining validation loss within a tolerance.

    Parameters
    ----------
    patience : int
        Number of epochs without improvement after which training should be
        interrupted.
    min_delta : float, default=0.0
        Minimum change in validation loss to qualify as an improvement.
        If negative, it is automatically set to 0.0 with a warning.

    Attributes
    ----------
    best_valid_loss : float
        Best validation loss seen so far.
    best_train_loss : float
        Training loss corresponding to the best validation loss.
    epochs_counter : int
        Number of consecutive epochs without improvement.

    Notes
    -----
    - An epoch is considered an improvement if:
      1. Validation loss decreases by at least min_delta, OR
      2. Validation loss is within min_delta of the best and training loss decreases
    - The counter resets to 0 when improvement is detected.
    - Call `interrupt()` to check if training should be stopped.
    """
    def __init__(self, patience: int, min_delta: float = 0.0):
        """
        Initialize the EarlyStopping criterion.

        Parameters
        ----------
        patience : int
            Number of epochs without improvement to tolerate.
        min_delta : float, default=0.0
            Minimum change in validation loss to qualify as improvement.
        """
        if min_delta < 0:
            logger.warning('Negative min_delta; min_delta will be set to 0.')
            min_delta = 0
        self.min_delta = min_delta
        self.patience = patience
        self.epochs_counter = 0
        self.best_valid_loss = float("+inf")
        self.best_train_loss = float("+inf")

    def update(self, valid_loss: float, train_loss: float) -> bool:
        """
        Update the early stopping criterion with new loss values.

        Parameters
        ----------
        valid_loss : float
            Validation loss for the current epoch.
        train_loss : float
            Training loss for the current epoch.

        Returns
        -------
        bool
            True if improvement was detected (counter reset), False otherwise.
        """
        if valid_loss < self.best_valid_loss - self.min_delta:
            self.best_valid_loss = valid_loss
            self.best_train_loss = train_loss
            self.epochs_counter = 0
            return True

        if (self.best_valid_loss - self.min_delta < valid_loss < self.best_valid_loss) \
                and (train_loss < self.best_train_loss):
            self.best_train_loss = train_loss
            self.epochs_counter = 0
            return True

        self.epochs_counter += 1
        return False

    def interrupt(self) -> bool:
        """
        Check if training should be interrupted.

        Returns
        -------
        bool
            True if the number of epochs without improvement has exceeded patience,
            False otherwise.
        """
        return self.epochs_counter >= self.patience