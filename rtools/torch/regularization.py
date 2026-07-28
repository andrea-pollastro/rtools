import logging
logger = logging.getLogger(__name__)

class EarlyStopping:
    """
    Early stopping mechanism to interrupt training when improvement plateaus.

    Monitors validation and training losses. Stops if no improvement for 'patience' epochs.

    Parameters
    ----------
    patience : int
        Number of epochs without improvement after which training should be interrupted.
    min_delta : float, default=0.0
        Minimum change in validation loss to qualify as an improvement.
    """
    def __init__(self, patience: int, min_delta: float = 0.0):
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

        Notes
        -----
        Two cases count as improvement and reset the patience counter:
        1. `valid_loss` improves on `best_valid_loss` by more than `min_delta`.
        2. `valid_loss` regresses by less than `min_delta` (within noise
           tolerance) but `train_loss` still improves on `best_train_loss`.
           This keeps training going through validation-loss noise as long
           as the model keeps fitting the training data better.
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

