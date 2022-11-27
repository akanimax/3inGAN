from functools import partial
from typing import Any, Dict, Sequence, Tuple

from torch import Tensor
from torch.nn import Module, ModuleList, Sequential, LeakyReLU, Identity, Conv2d

from thre3d_atom.networks.network_interface import Network
from thre3d_atom.networks.shared.equalized_layers import EqualizedConv2d
from thre3d_atom.networks.shared.layers import (
    EncoderBlock,
    MinibatchStdDev,
)
from thre3d_atom.networks.shared.utils import num_feature_maps_depth_decaying


class ConvolutionalEncoder(Network):
    def __init__(
        self,
        depth: int = 8,
        num_input_channels: int = 3,
        latent_size: int = 512,
        fmap_base: int = 2048,
        fmap_decay: float = 1.0,
        fmap_min: int = 32,
        fmap_max: int = 512,
        use_eql: bool = False,
        use_minibatch_stddev: bool = False,
    ) -> None:
        super().__init__()

        # object state
        self._use_minibatch_stddev = use_minibatch_stddev
        self._depth = depth
        self._num_input_channels = num_input_channels
        self._latent_size = latent_size
        self._fmap_base = fmap_base
        self._fmap_decay = fmap_decay
        self._fmap_min = fmap_min
        self._fmap_max = fmap_max
        self._use_eql = use_eql
        self._conv_module = EqualizedConv2d if use_eql else Conv2d

        # construct a shorthand for the nf:
        self._nf = partial(
            num_feature_maps_depth_decaying,
            feature_maps_base=fmap_base,
            feature_maps_decay=fmap_decay,
            feature_maps_max=fmap_max,
            feature_maps_min=fmap_min,
        )

        # create the encoder blocks:
        self._from_rgb = Sequential(
            self._conv_module(
                num_input_channels, self._nf(depth), kernel_size=1, stride=1
            ),
            LeakyReLU(),
        )
        self._blocks = ModuleList([self.block(stage) for stage in range(depth, 1, -1)])

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, self._latent_size, None, None

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, self._num_input_channels, None, None

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "depth": self._depth,
                "num_input_channels": self._num_input_channels,
                "latent_size": self._latent_size,
                "fmap_base": self._fmap_base,
                "fmap_decay": self._fmap_decay,
                "fmap_min": self._fmap_min,
                "fmap_max": self._fmap_max,
                "use_minibatch_stddev": self._use_minibatch_stddev,
                "use_eql": self._use_eql,
            },
            "state_dict": self.state_dict(),
        }

    def block(self, stage: int) -> Module:
        if stage >= 3:  # 8x8 resolution and up
            return EncoderBlock(
                self._nf(stage), self._nf(stage - 1), use_residual=False
            )
        else:  # 4x4 resolution
            # fmt: off
            first_in_channels = self._nf(stage) + (1 if self._use_minibatch_stddev else 0)
            return Sequential(
                MinibatchStdDev() if self._use_minibatch_stddev else Identity(),
                self._conv_module(first_in_channels, self._nf(stage - 1),
                                  kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                LeakyReLU(),
                self._conv_module(self._nf(stage - 1), self._nf(stage - 2),
                                  kernel_size=(4, 4)),
                LeakyReLU(),
                self._conv_module(self._nf(stage - 2), self._latent_size, kernel_size=(1, 1)),
            )
            # fmt: on

    def forward(
        self,
        x: Tensor,
        alpha: float = 1e-7,
        normalize_embeddings: bool = False,
        intermediate_embeddings: bool = False,
    ) -> Tensor:
        y = self._from_rgb(x)
        features_list = []
        for block in self._blocks:
            y = block(y)
            features_list.append(y)
        # remove the last ones, because we are also returning the embeddings
        features_list.pop(-1)
        if normalize_embeddings:
            # normalize y to fall on unit sphere
            y = y / (y.norm(dim=-1, keepdim=True) + alpha)
        if intermediate_embeddings:
            return y, features_list
        return y, []
