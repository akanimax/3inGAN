from itertools import repeat
from typing import Union, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as torch_func
from torch import Tensor
from torch.nn import Conv2d, ConvTranspose2d, Linear, Conv3d, ConvTranspose3d


# -------------------------------------------
#  Small private helpers
# -------------------------------------------
def _ntuple(n):
    def parse(x):
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


# ----------------------------------------------------------------------------------------------------------------------
#  Equalized layers
# ----------------------------------------------------------------------------------------------------------------------
class EqualizedConv2d(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = np.prod(self.kernel_size) * self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            return torch_func.conv2d(
                torch_func.pad(
                    x, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                self.weight * self.scale,
                self.bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return torch_func.conv2d(
            x,
            self.weight * self.scale,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class EqualizedConv3d(Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = np.prod(self.kernel_size) * self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor) -> Tensor:
        if self.padding_mode != "zeros":
            return torch_func.conv3d(
                torch_func.pad(
                    x, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                self.weight * self.scale,
                self.bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return torch_func.conv3d(
            x,
            self.weight * self.scale,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class EqualizedConvTranspose2d(ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )
        output_padding = self._output_padding(
            x,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )
        return torch_func.conv_transpose2d(
            x,
            self.weight * self.scale,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class EqualizedConvTranspose3d(ConvTranspose3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        output_padding: Union[int, Tuple[int, int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        if bias:
            torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = self.in_channels
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor, output_size: Optional[List[int]] = None) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose3d"
            )
        output_padding = self._output_padding(
            x,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            self.dilation,
        )
        return torch_func.conv_transpose3d(
            x,
            self.weight * self.scale,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class EqualizedLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias=True) -> None:
        super().__init__(in_features, out_features, bias)

        # make sure that the self.weight and self.bias are initialized according to
        # random normal distribution
        torch.nn.init.normal_(self.weight)
        # if bias:
        #   torch.nn.init.zeros_(self.bias)

        # define the scale for the weights:
        fan_in = self.in_features
        self.scale = np.sqrt(2) / np.sqrt(fan_in)

    def forward(self, x: Tensor) -> Tensor:
        return torch_func.linear(x, self.weight * self.scale, self.bias)
