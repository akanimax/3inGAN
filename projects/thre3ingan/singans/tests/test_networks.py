import torch

from projects.thre3ingan.singans.networks import Thre3dGenerator
from torch.backends import cudnn

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_thre3d_generator() -> None:
    batch_size = 1
    random_input = torch.randn(batch_size, 128, 64, 64, 64).to(device)

    network = Thre3dGenerator().to(device)
    print(network)

    output = network(random_input)

    assert output.shape == (batch_size, 8, 64, 64, 64)
