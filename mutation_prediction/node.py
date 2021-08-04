import os
import socket
from typing import Optional

import torch


def get_pl_gpus() -> int:
    return torch.cuda.device_count()


def get_node_name() -> str:
    if "MUTATION_PREDICTION_NODE_NAME" in os.environ:
        return os.environ["MUTATION_PREDICTION_NODE_NAME"]
    else:
        return socket.gethostname()


def get_num_cpus() -> int:
    return len(os.sched_getaffinity(0))


def get_redis_url() -> str:
    return os.environ["MUTATION_PREDICTION_REDIS"]


def get_precomputed_path() -> str:
    return os.getenv("MUTATION_PREDICTION_PRECOMPUTED", default="precomputed")


def get_precomputed_dataset() -> Optional[str]:
    return os.getenv("MUTATION_PREDICTION_PRECOMPUTED_DATASET", None)


def get_storage_url() -> str:
    return os.environ["MUTATION_PREDICTION_STORAGE"]
