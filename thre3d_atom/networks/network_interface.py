import abc
from typing import Any, Dict, Sequence, Tuple

from torch.nn import Module


class Network(Module, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        """Returns the required shapes of the inputs to the network. None represents
        any size"""

    @property
    @abc.abstractmethod
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        """Returns the shapes of the outputs of the network. None represents
        any size"""

    @abc.abstractmethod
    def get_save_info(self) -> Dict[str, Any]:
        """Returns the save info of the module. This should contain the model
        weights (i.e. state_dict) and any configuration needed to create this object"""
