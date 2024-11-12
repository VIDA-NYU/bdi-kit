import os
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

BDIKIT_DEVICE: str = os.getenv("BDIKIT_DEVICE", default="cpu")

VALUE_MATCHING_THRESHOLD = 0.3

default_os_cache_dir = os.getenv(
    "XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")
)

BDIKIT_CACHE_DIR = os.getenv(
    "BDIKIT_CACHE", os.path.join(default_os_cache_dir, "bdikit")
)


def get_device() -> str:
    if BDIKIT_DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return BDIKIT_DEVICE
