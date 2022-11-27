from typing import Any, Callable, Optional, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToTensor


class NoOp(object):
    """A NoOp image transform utility. Does nothing, but makes the code cleaner"""

    def __call__(self, whatever: Any) -> Any:
        return whatever

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


def get_torch_vision_image_transform(
    new_size: Optional[Tuple[int, int]] = None, flip_horizontal: bool = False
) -> Callable[[Image.Image], Tensor]:
    """
    obtain the image transforms required for the input data
    Args:
        new_size: size of the resized images (if needed, could be None)
        flip_horizontal: whether to randomly mirror input images during training
    Returns: requested transform object from TorchVision
    """
    return Compose(
        [
            ToTensor(),
            RandomHorizontalFlip(p=0.5) if flip_horizontal else NoOp(),
            Resize(new_size) if new_size is not None else NoOp(),
        ]
    )


def infinite_dataloader(data_loader: DataLoader) -> Any:
    while True:
        for data in data_loader:
            yield data
