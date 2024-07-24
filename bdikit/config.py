import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"


BDIKIT_DEVICE: str = os.getenv("BDIKIT_DEVICE", default="cpu")
VALUE_MATCHING_THRESHOLD = 0.3


def get_device() -> str:
    if BDIKIT_DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return BDIKIT_DEVICE
