""" Module contains variants of fully connected (dense) networks """
import dataclasses
from typing import Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import Dropout, Linear, ModuleList, Sequential, Softplus, Module, Identity

from thre3d_atom.networks.network_interface import Network
from thre3d_atom.networks.shared.equalized_layers import EqualizedLinear
from thre3d_atom.utils.config_utils import validity
from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS, NUM_COLOUR_CHANNELS


def _init_weights_glorot_uniform(m: torch.nn.Module):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0.0)


@dataclasses.dataclass
class SkipMLPConfig:
    input_dims: int = NUM_COORD_DIMENSIONS
    output_dims: int = NUM_COLOUR_CHANNELS
    layer_depths: Sequence[int] = (128, 128, 128)
    skips: Sequence[bool] = (False, False, False)
    dropout_prob: float = 0.0
    use_equalized_learning_rate: bool = False
    # Using soft-plus instead of ReLU as suggested in the pytorch3d demo
    activation_fn: Module = Softplus()
    out_activation_fn: Module = Identity()

    def __post_init__(self):
        validity(len(self.layer_depths) == len(self.skips))
        validity(self.input_dims > 0)
        validity(self.output_dims > 0)
        validity(0.0 <= self.dropout_prob <= 1.0)


class SkipMLP(Network):
    def __init__(
        self,
        config: SkipMLPConfig,
    ) -> None:
        super().__init__()
        self._config = config
        self.network_layers = self._get_sequential_layers_with_skips()

        # initialize the layers according to the glorot_uniform
        if not self._config.use_equalized_learning_rate:
            self.apply(_init_weights_glorot_uniform)

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, self._config.input_dims

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, self._config.output_dims

    def _get_sequential_layers_with_skips(self) -> ModuleList:
        linear_module = (
            EqualizedLinear if self._config.use_equalized_learning_rate else Linear
        )
        modules, in_features = [], self._config.input_dims
        for skip, layer_depth in zip(self._config.skips, self._config.layer_depths):
            modules.append(
                Sequential(
                    linear_module(
                        in_features=in_features,
                        out_features=layer_depth,
                        bias=True,
                    ),
                    self._config.activation_fn,
                    Dropout(self._config.dropout_prob),
                )
            )
            in_features = layer_depth + self._config.input_dims if skip else layer_depth

        # add the final output layer separately:
        modules.append(linear_module(in_features, self._config.output_dims, bias=True))
        modules.append(self._config.out_activation_fn)
        return ModuleList(modules)

    def get_save_info(self):
        return {
            "config": dataclasses.asdict(self._config),
            "state_dict": self.state_dict(),
        }

    def forward(self, x: Tensor) -> Tensor:
        y = self.network_layers[0](x)  # first layer doesn't have skips
        for skip, network_layer in zip(self._config.skips, self.network_layers[1:]):
            y = torch.cat([y, x], dim=-1) if skip else y
            y = network_layer(y)
        # don't forget to apply out_activation layer before returning the output
        return self.network_layers[-1](y)
