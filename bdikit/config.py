import os
import torch


BDIKIT_DEVICE: str = os.getenv("BDIKIT_DEVICE", default="cpu")


def get_device() -> str:
    if BDIKIT_DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return BDIKIT_DEVICE
