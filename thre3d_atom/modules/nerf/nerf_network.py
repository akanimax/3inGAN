from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import Softplus

from thre3d_atom.networks.dense_nets import SkipMLP, SkipMLPConfig
from thre3d_atom.networks.network_interface import Network
from thre3d_atom.networks.shared.layers import Embedder, PositionalEncodingsEmbedder
from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS, NUM_COLOUR_CHANNELS


class NerfNet(Network):
    def __init__(
        self,
        point_embedder: Embedder,
        dir_embedder: Embedder,
        point_mlp: SkipMLP,
        dir_mlp: SkipMLP,
        num_coord_dimensions: Tuple[int, ...] = NUM_COORD_DIMENSIONS,
    ) -> None:
        super().__init__()
        # self._check_compatibility(
        #     point_embedder, dir_embedder, point_mlp, dir_mlp, num_coord_dimensions
        # )
        self._num_coord_dimensions = num_coord_dimensions
        self._point_embedder = point_embedder
        self._dir_embedder = dir_embedder
        self._point_mlp = point_mlp
        self._dir_mlp = dir_mlp

        # disabled due to the use of the shifted_softplus activation
        # self._point_mlp.network_layers[-2].bias.data[0] = -1.5

    def load_weights(self, weights: Dict[str, Any]) -> None:
        self._point_embedder.load_state_dict(weights["point_embedder"]["state_dict"])
        self._dir_embedder.load_state_dict(weights["dir_embedder"]["state_dict"])
        self._dir_mlp.load_state_dict(weights["dir_mlp"]["state_dict"])
        self._point_mlp.load_state_dict(weights["point_mlp"]["state_dict"])

    def _check_compatibility(
        self,
        point_embedder: Embedder,
        dir_embedder: Embedder,
        point_mlp: SkipMLP,
        dir_mlp: SkipMLP,
        num_coord_dimensions: int,
    ) -> None:
        # check input shapes of embedder networks
        assert (
            num_coord_dimensions
            == point_embedder.input_shape[-1]
            == dir_embedder.input_shape[-1]
        ), (
            f"embedder networks cannot embed "
            f"{self._num_coord_dimensions} dimensional coordinates"
        )

        # check compatibility between point_embedder and point_mlp
        emb_dims = point_embedder.output_shape[-1]
        mlp_dims = point_mlp.input_shape[-1]
        assert emb_dims <= mlp_dims, (
            f"point_embedder and point_mlp are incompatible "
            f"point_embedder output dimensions: {emb_dims} "
            f"point_mlp input dimensions: {mlp_dims}"
        )

        # check compatibility between dir_embedder, point_mlp and dir_mlp
        emb_dims = dir_embedder.output_shape[-1]
        point_mlp_out = point_mlp.output_shape[-1]
        dir_mlp_dims = dir_mlp.input_shape[-1]
        # the -1 below is for density which is not input to the dir_mlp
        assert dir_mlp_dims == (emb_dims + point_mlp_out - 1), (
            f"point_mlp, dir_embedder and dir_mlp are incompatible "
            f"point_mlp output dimensions: {point_mlp_out} "
            f"dir_embedder output dimensions: {emb_dims} "
            f"dir_mlp input dimensions: {dir_mlp_dims}"
        )

        # check the output dimensions of the dir_mlp
        assert (
            dir_mlp.output_shape[-1] == NUM_COLOUR_CHANNELS
        ), f"output of the direction_mlp should be RGB, but got"

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, 2 * self._num_coord_dimensions

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        # plus 1 is for the instantaneous density
        return None, self._dir_mlp.output_shape[-1] + 1

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "point_embedder": self._point_embedder.get_save_info(),
            "dir_embedder": self._dir_embedder.get_save_info(),
            "point_mlp": self._point_mlp.get_save_info(),
            "dir_mlp": self._dir_mlp.get_save_info(),
            "num_coord_dimensions": self._num_coord_dimensions,
        }

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass of the nerf-network. Please refer to the diagram
        in the supplementary material of the NeRF paper"""
        points, directions = x[:, : x.shape[-1] // 2], x[:, x.shape[-1] // 2 :]

        # obtain the embedded input
        embedded_points = self._point_embedder(points)
        embedded_directions = self._dir_embedder(directions)

        # output of the point network
        point_mlp_out = self._point_mlp(embedded_points)
        density = point_mlp_out[:, :1]  # first value in the feature vec is density
        feature_vector = point_mlp_out[:, 1:]

        # output of the direction network
        dir_net_input = torch.cat([feature_vector, embedded_directions], dim=-1)
        colour = self._dir_mlp(dir_net_input)

        # return the colour and density output
        return torch.cat([colour, density], dim=-1)


def get_sota_nerf_net() -> NerfNet:
    """Returns the NerfNetwork with the SOTA configuration
    please refer to the paper (and supplementary material) for more information"""
    point_embedder = PositionalEncodingsEmbedder(
        input_dims=NUM_COORD_DIMENSIONS, emb_dims=10
    )
    directions_embedder = PositionalEncodingsEmbedder(
        input_dims=NUM_COORD_DIMENSIONS, emb_dims=4
    )

    point_mlp_config = SkipMLPConfig(
        input_dims=point_embedder.output_shape[-1],
        output_dims=256 + 1,  # +1 for density
        layer_depths=[256] * 8,
        skips=[False] * 4 + [True] + [False] * 3,
        dropout_prob=0.0,  # dropout is disabled
        use_equalized_learning_rate=False,
        activation_fn=Softplus(),
    )
    point_mlp = SkipMLP(point_mlp_config)

    dir_mlp_config = SkipMLPConfig(
        input_dims=directions_embedder.output_shape[-1]
        + point_mlp.output_shape[-1]
        - 1,
        output_dims=NUM_COLOUR_CHANNELS,
        layer_depths=[128],
        skips=[False],
        dropout_prob=0.0,  # dropout is disabled
        use_equalized_learning_rate=False,
        activation_fn=Softplus(),
    )
    dir_mlp = SkipMLP(dir_mlp_config)

    return NerfNet(
        point_embedder=point_embedder,
        dir_embedder=directions_embedder,
        point_mlp=point_mlp,
        dir_mlp=dir_mlp,
        num_coord_dimensions=NUM_COORD_DIMENSIONS,
    )
