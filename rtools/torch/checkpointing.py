import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage saving and loading PyTorch checkpoints.

    Saves checkpoints on request with epoch-based organization.

    Parameters
    ----------
    checkpoint_dir : str or Path
        Directory to save checkpoints in.

    Examples
    --------
    >>> import torch
    >>> from rtools.torch.checkpointing import CheckpointManager
    >>> model = torch.nn.Linear(10, 5)
    >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    >>> manager = CheckpointManager("./checkpoints")
    >>> manager.save(model, optimizer, epoch=1)
    >>> manager.load(model, optimizer, epoch=1)
    """

    def __init__(self, checkpoint_dir: Union[str, Path]) -> None:
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        **kwargs: Any,
    ) -> Path:
        """
        Save checkpoint to disk.

        Parameters
        ----------
        model : torch.nn.Module
            Model to save.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to save.
        epoch : int, optional
            Epoch number for this checkpoint. Used in filename if provided.
        **kwargs
            Additional metadata to save with checkpoint (e.g., loss, metrics).

        Returns
        -------
        Path
            Path to saved checkpoint.
        """
        if epoch is not None:
            name = f"checkpoint_epoch_{epoch:05d}"
        else:
            name = f"checkpoint"

        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            **kwargs,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info("Saved checkpoint to %s", checkpoint_path)

        return checkpoint_path

    def load(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint from disk.

        Parameters
        ----------
        model : torch.nn.Module
            Model to load state into.
        optimizer : torch.optim.Optimizer, optional
            Optimizer to load state into.
        epoch : int, optional
            Epoch number to load. If None, uses checkpoint_path.
        checkpoint_path : str or Path, optional
            Path to checkpoint. If None, derives from epoch.

        Returns
        -------
        dict
            Checkpoint metadata (excluding model and optimizer state).

        Raises
        ------
        FileNotFoundError
            If checkpoint file does not exist.
        ValueError
            If neither epoch nor checkpoint_path is provided.
        """
        if checkpoint_path is None:
            if epoch is not None:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:05d}.pt"
            else:
                checkpoint_path = self.checkpoint_dir / "checkpoint.pt"

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=False)

        model.load_state_dict(checkpoint["model_state"])
        if optimizer and checkpoint.get("optimizer_state"):
            optimizer.load_state_dict(checkpoint["optimizer_state"])

        logger.info("Loaded checkpoint from %s", checkpoint_path)

        # Return all metadata except model/optimizer state
        return {k: v for k, v in checkpoint.items() if k not in ("model_state", "optimizer_state")}


