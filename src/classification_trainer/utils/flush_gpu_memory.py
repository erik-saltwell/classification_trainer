import gc

import torch


def flush_gpu_memory() -> None:
    """Run garbage collection and free cached GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
