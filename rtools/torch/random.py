import random
import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)

def set_seed(seed: int, deterministic: bool = False):
    """
    Set seeds for Python, NumPy, and PyTorch.
    """
    logger.debug("Setting global seed to %d", seed)
    
    random.seed(seed)
    logger.debug("Python random seed set")
    
    np.random.seed(seed)
    logger.debug("NumPy random seed set")

    torch.manual_seed(seed)
    logger.debug("PyTorch CPU seed set")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(
            "PyTorch CUDA seed set (GPUs: %d)",
            torch.cuda.device_count()
        )
    else:
        logger.info("CUDA not available — GPU seeding skipped")

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.debug("cuDNN deterministic=True, benchmark=False")

        # warn_only=True logga un warning invece di crashare
        # quando un'operazione non ha implementazione deterministica
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info("Deterministic algorithms enforced (warn_only=True)")
    else:
        # benchmark=True può accelerare l'addestramento su GPU
        torch.backends.cudnn.benchmark = True
        logger.debug("cuDNN benchmark=True (non-deterministic, faster)")
