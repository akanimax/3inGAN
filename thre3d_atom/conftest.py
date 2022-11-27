from pathlib import Path

import numpy as np
import pytest
import torch

from thre3d_atom.utils.constants import SEED

current_path = Path(__file__).parent.absolute()


@pytest.fixture
def data_path() -> Path:
    return current_path.parent / "projects/data/lego"


@pytest.fixture(autouse=True)
def execute_before_every_test():
    torch.manual_seed(SEED)
    np.random.seed(SEED)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size() -> int:
    return 32


@pytest.fixture
def num_samples() -> int:
    return 64
