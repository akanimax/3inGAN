from functools import partial
from typing import Any, Dict, Sequence, Tuple, Optional, Type

import numpy as np
import torch
from torch import Tensor
from torch.nn import (
    Module,
    Sequential,
    AvgPool2d,
    Upsample,
    Identity,
    BatchNorm2d,
    Conv2d,
    Conv3d,
    Tanh,
    BatchNorm3d,
)
from torch.nn.functional import pad

from thre3d_atom.networks.network_interface import Network
from thre3d_atom.networks.shared.equalized_layers import (
    EqualizedConv3d,
    EqualizedConv2d,
)
from thre3d_atom.utils.constants import (
    NUM_COLOUR_CHANNELS,
    ZERO_PLUS,
    NUM_RGBA_CHANNELS,
)
from thre3d_atom.utils.logging import log

Embedder = Network  # Embedder has the same interface as a network, but renaming this for better readability


# TODO: add a test for this layer
class PositionalEncodingsEmbedder(Embedder):
    """
    Embeds input vectors into periodic encodings
    Args:
        input_dims: number of input dimensions
        emb_dims: number of dimensions in the encoded vectors
    """

    def __init__(self, input_dims: int, emb_dims: int):
        super().__init__()
        self._input_dims = input_dims
        self._emb_dims = emb_dims

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, self._input_dims

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        # 2 for sin and cos
        return None, (self._input_dims + 2 * self._input_dims * self._emb_dims)

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "input_dims": self._input_dims,
            "emb_dims": self._emb_dims,
            "state_dict": self.state_dict(),
        }

    def extra_repr(self) -> str:
        return f"input_dims={self._input_dims}, emb_dims={self._emb_dims}"

    def forward(self, x: Tensor) -> Tensor:
        """
        converts the input vectors into positional encodings
        Args:
            x: batch of input_vectors => [batch_size x self.input_dims]
        Returns: positional encodings =>
                            [batch_size x (self.input_dims + 2 * self.input_dims * emb_dims)]
                            2 for sine and cos
        """
        sin_embedding_dims = (
            torch.arange(0, self._emb_dims, dtype=x.dtype, device=x.device)
            .reshape((1, self._emb_dims))
            .repeat_interleave(repeats=self._input_dims, dim=-1)
        )
        cos_embedding_dims = (
            torch.arange(0, self._emb_dims, dtype=x.dtype, device=x.device)
            .reshape((1, self._emb_dims))
            .repeat_interleave(repeats=self._input_dims, dim=-1)
        )
        for_sines = (2 ** sin_embedding_dims) * x.repeat(1, self._emb_dims)
        for_coses = (2 ** cos_embedding_dims) * x.repeat(1, self._emb_dims)
        sines, coses = torch.sin(for_sines), torch.cos(for_coses)
        return torch.cat((x, sines, coses), dim=-1)


class EncoderBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: torch.nn.Module = torch.nn.LeakyReLU(),
        use_residual: bool = False,
        use_eql: bool = False,
    ) -> None:
        super().__init__()

        conv_module = EqualizedConv2d if use_eql else Conv2d

        # fmt: off
        self._normal_path = Sequential(
            conv_module(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            activation,
            conv_module(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            activation,
            AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self._residual_path = Sequential(
            conv_module(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1)),
            activation,
            AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        ) if use_residual else None
        # fmt: on

    def forward(self, x: Tensor) -> Tensor:
        y = self._normal_path(x)
        if self._residual_path is not None:
            res_y = self._residual_path(x)
            return (y + res_y) * (1 / np.sqrt(2))
        else:
            return y


class Thre3dVolumeMakerBlock(Module):
    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        activation: Type[torch.nn.LeakyReLU] = torch.nn.LeakyReLU(),
    ) -> None:
        super().__init__()

        # fmt: off
        self._block = Sequential(
            PixelwiseNorm(),  # normalizes the latents to hypersphere
            EqualizedConv3d(in_channels, intermediate_channels,
                            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            activation, PixelwiseNorm(),
            EqualizedConv3d(intermediate_channels, intermediate_channels,
                            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            activation, PixelwiseNorm(),
            EqualizedConv3d(intermediate_channels, intermediate_channels,
                            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            activation, PixelwiseNorm(),
            EqualizedConv3d(intermediate_channels, out_channels,
                            kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        # fmt: on

    def forward(self, x: Tensor) -> Tensor:
        return self._block(x)


class Thre3dDecoderBlock(Module):
    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        upsample_factor: Optional[float] = 2.0,
        activation: torch.nn.Module = torch.nn.LeakyReLU(),
        use_residual_connection: bool = True,
        use_trilinear_upsampling: bool = False,
    ) -> None:
        super().__init__()

        self._residual = use_residual_connection
        upsampling_mode = "trilinear" if use_trilinear_upsampling else "nearest"
        self._upsampler = Upsample(scale_factor=upsample_factor, mode=upsampling_mode)

        # fmt: off
        conv = partial(EqualizedConv3d, kernel_size=(3, 3, 3),
                       stride=(1, 1, 1), padding=(1, 1, 1))

        self._block = Sequential(
            self._upsampler,
            conv(in_channels, intermediate_channels),
            activation, PixelwiseNorm(),
            conv(intermediate_channels, intermediate_channels),
            activation, PixelwiseNorm(),
            conv(intermediate_channels, out_channels),
            activation if use_residual_connection else Identity(),
            PixelwiseNorm() if use_residual_connection else Identity(),
            conv(out_channels, in_channels) if use_residual_connection else Identity(),
        )
        # fmt: on

    def forward(self, x: Tensor) -> Tensor:
        if self._residual:
            upsampled_x = self._upsampler(x)
            return (upsampled_x + self._block(x)) * (1 / np.sqrt(2))
        return self._block(x)


class TwodCoarseImageMakerBlock(Module):
    def __init__(
        self,
        in_channels: int = NUM_COLOUR_CHANNELS,
        intermediate_channels: int = 32,
        activation: Type[torch.nn.LeakyReLU] = torch.nn.LeakyReLU(),
        use_eql: bool = True,
    ) -> None:
        super().__init__()

        out_channels: int = in_channels
        out_activation: Type[torch.nn.LeakyReLU] = (
            torch.nn.Tanh() if out_channels == NUM_COLOUR_CHANNELS else Identity()
        )

        conv_module = EqualizedConv2d if use_eql else Conv2d

        # fmt: off
        self._block = Sequential(
            conv_module(in_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(intermediate_channels), activation,
            conv_module(intermediate_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(intermediate_channels), activation,
            conv_module(intermediate_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(intermediate_channels), activation,
            conv_module(intermediate_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(intermediate_channels), activation,
            conv_module(intermediate_channels, out_channels,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            out_activation,
        )
        # fmt: on

    def forward(self, x: Tensor) -> Tensor:
        return self._block(x)


class TwodSinGanResidualDecoderBlock(Module):
    def __init__(
        self,
        in_channels: int = NUM_COLOUR_CHANNELS,
        intermediate_channels: int = 32,
        activation: Type[torch.nn.LeakyReLU] = torch.nn.LeakyReLU(),
        use_eql: bool = True,
    ) -> None:
        super().__init__()

        out_channels = in_channels
        out_activation: Type[torch.nn.LeakyReLU] = (
            torch.nn.Tanh() if out_channels == NUM_COLOUR_CHANNELS else Identity()
        )

        # create a learnable noise_controller
        self._noise_controller = torch.nn.Parameter(
            torch.zeros(size=(in_channels,), dtype=torch.float32, requires_grad=True)
        )

        conv_module = EqualizedConv2d if use_eql else Conv2d

        # fmt: off
        self._block = Sequential(
            conv_module(in_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(intermediate_channels), activation,
            conv_module(intermediate_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(intermediate_channels), activation,
            conv_module(intermediate_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(intermediate_channels), activation,
            conv_module(intermediate_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BatchNorm2d(intermediate_channels), activation,
            conv_module(intermediate_channels, out_channels,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            out_activation,
        )
        # fmt: on

    @property
    def noise_controller(self) -> Tensor:
        return self._noise_controller

    def forward(self, x: Tensor) -> Tensor:
        random_noise = torch.randn_like(x).to(x.device)
        controlled_noise = self._noise_controller[None, :, None, None] * random_noise
        return (x + self._block(x + controlled_noise)) * (1 / np.sqrt(2))


class Thre3dCoarseVolumeMakerBlock(Module):
    def __init__(
        self,
        in_features: int = NUM_RGBA_CHANNELS,
        intermediate_features: int = 32,
        activation: Type[torch.nn.LeakyReLU] = torch.nn.LeakyReLU(negative_slope=0.2),
        use_eql: bool = True,
    ) -> None:
        super().__init__()

        out_features: int = in_features
        out_activation: Type[torch.nn.LeakyReLU] = (
            Tanh() if in_features == NUM_RGBA_CHANNELS else Identity()
        )

        conv_module = EqualizedConv3d if use_eql else Conv3d
        kernel_size, padding_size = (3, 3, 3), (0, 0, 0)
        norm_module = BatchNorm3d

        # note that the following attribute is public
        self.padding_dims = 5  # based on convolutions receptive fields

        # fmt: off
        self._block = Sequential(
            conv_module(in_features, intermediate_features,
                        kernel_size=kernel_size, stride=(1, 1, 1), padding=padding_size),
            norm_module(intermediate_features),
            activation,
            conv_module(intermediate_features, intermediate_features,
                        kernel_size=kernel_size, stride=(1, 1, 1), padding=padding_size),
            norm_module(intermediate_features),
            activation,
            conv_module(intermediate_features, intermediate_features,
                        kernel_size=kernel_size, stride=(1, 1, 1), padding=padding_size),
            norm_module(intermediate_features),
            activation,
            conv_module(intermediate_features, intermediate_features,
                        kernel_size=kernel_size, stride=(1, 1, 1), padding=padding_size),
            norm_module(intermediate_features),
            activation,
            conv_module(intermediate_features, out_features,
                        kernel_size=kernel_size, stride=(1, 1, 1), padding=padding_size),
            out_activation,
        )
        # fmt: on

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self._block.modules():
            if "Conv3d" in layer.__class__.__name__:
                layer.weight.data.normal_(0.0, 0.02)
            elif (
                "Norm" in layer.__class__.__name__
                and hasattr(layer, "weight")
                and hasattr(layer, "bias")
                and layer.weight is not None
                and layer.bias is not None
            ):
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        padded_x = pad(x, [self.padding_dims] * 6)
        return self._block(padded_x)


class Thre3dSinGanResidualDecoderBlock(Module):
    def __init__(
        self,
        in_features: int = NUM_RGBA_CHANNELS,
        intermediate_features: int = 32,
        activation: Module = torch.nn.LeakyReLU(negative_slope=0.2),
        use_eql: bool = True,
    ) -> None:
        super().__init__()

        log.info(
            "Using the new clipping behaviour while adding residual details to previous stage output"
        )

        out_features = in_features
        out_activation = Tanh() if in_features == NUM_RGBA_CHANNELS else Identity()

        conv_module = EqualizedConv3d if use_eql else Conv3d
        kernel_size, padding_size = (3, 3, 3), (0, 0, 0)
        norm_module = BatchNorm3d

        self._padding_dims = 5  # based on convolutions receptive fields

        # variable for caching fixed random noise
        self._fixed_noise = None

        # fmt: off
        self._block = Sequential(
            conv_module(in_features, intermediate_features,
                        kernel_size=kernel_size, stride=(1, 1, 1), padding=padding_size),
            norm_module(intermediate_features), activation,
            conv_module(intermediate_features, intermediate_features,
                        kernel_size=kernel_size, stride=(1, 1, 1), padding=padding_size),
            norm_module(intermediate_features), activation,
            conv_module(intermediate_features, intermediate_features,
                        kernel_size=kernel_size, stride=(1, 1, 1), padding=padding_size),
            norm_module(intermediate_features), activation,
            conv_module(intermediate_features, intermediate_features,
                        kernel_size=kernel_size, stride=(1, 1, 1), padding=padding_size),
            norm_module(intermediate_features), activation,
            conv_module(intermediate_features, out_features,
                        kernel_size=kernel_size, stride=(1, 1, 1), padding=padding_size),
            out_activation,
        )
        # fmt: on

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self._block.modules():
            if "Conv3d" in layer.__class__.__name__:
                layer.weight.data.normal_(0.0, 0.02)
            elif (
                "Norm" in layer.__class__.__name__
                and hasattr(layer, "weight")
                and hasattr(layer, "bias")
                and layer.weight is not None
                and layer.bias is not None
            ):
                layer.weight.data.normal_(1.0, 0.02)
                layer.bias.data.fill_(0)

    def forward(
        self, x: Tensor, noise_multiplier: float = 0.0, use_fixed_noise: bool = False
    ) -> Tensor:
        if use_fixed_noise and self._fixed_noise is not None:
            random_noise = self._fixed_noise
        else:
            random_noise = torch.randn_like(x, device=x.device)
            self._fixed_noise = random_noise

        padded_x = pad(x, [self._padding_dims] * 6)
        controlled_noise = pad(
            noise_multiplier * random_noise, [self._padding_dims] * 6
        )
        return torch.clip(x + (self._block(padded_x + controlled_noise)), -1.0, 1.0)


# TODO: add test for this layer
class MinibatchStdDev(Module):
    """
    Minibatch standard deviation layer for the discriminator
    Args:
        group_size: Size of each group into which the batch is split
    """

    def __init__(self, group_size: int = 4) -> None:
        super(MinibatchStdDev, self).__init__()
        self.group_size = group_size

    def extra_repr(self) -> str:
        return f"group_size={self.group_size}"

    def forward(self, x: Tensor, alpha: float = 1e-8) -> Tensor:
        """
        forward pass of the layer
        Args:
            x: input activation volume
            alpha: small number for numerical stability
        Returns: y => x appended with standard deviation constant map
        """
        batch_size, channels, height, width = x.shape

        assert (
            batch_size % self.group_size == 0 or batch_size < self.group_size
        ), f"batch_size ({batch_size}) must be fully divisible by group_size ({self.group_size}) or less than it"

        # reshape x and create the splits of the input accordingly
        group_size = min(batch_size, self.group_size)
        y = torch.reshape(x, [group_size, -1, channels, height, width])
        y -= y.mean(dim=0, keepdim=True)
        y = torch.mean((y ** 2), dim=0)
        y = torch.sqrt(y + alpha)
        y = torch.mean(y, dim=(1, 2, 3), keepdim=True)
        y = y.repeat(group_size, 1, height, width)

        # [B x (N + 1) x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)

        # return the computed values:
        return y


class PixelwiseNorm(Module):
    """
    ------------------------------------------------------------------------------------
    Pixelwise feature vector normalization.
    reference:
    https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
    ------------------------------------------------------------------------------------
    """

    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    @staticmethod
    def forward(x: Tensor, alpha: float = ZERO_PLUS) -> Tensor:
        y = x.pow(2.0).mean(dim=1, keepdim=True).sqrt()  # [N1...]
        y = x / (y + alpha)  # normalize the input x volume
        return y
