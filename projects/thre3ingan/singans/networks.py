from functools import partial
from typing import Sequence, Tuple, Dict, Any, Optional, Callable, Type, List, Union

import numpy as np
import torch
from torch import Tensor
from torch.nn import (
    LeakyReLU,
    Module,
    ModuleList,
    Softplus,
    Sequential,
    Conv2d,
    Conv3d,
    BatchNorm3d,
    BatchNorm2d,
)
from torch.nn.functional import interpolate

from thre3d_atom.modules.nerf.nerf_network import NerfNet
from thre3d_atom.networks.dense_nets import SkipMLP, SkipMLPConfig
from thre3d_atom.networks.network_interface import Network
from thre3d_atom.networks.shared.equalized_layers import (
    EqualizedConv2d,
    EqualizedConv3d,
)
from thre3d_atom.networks.shared.layers import (
    PositionalEncodingsEmbedder,
    Thre3dVolumeMakerBlock,
    Thre3dDecoderBlock,
    TwodCoarseImageMakerBlock,
    TwodSinGanResidualDecoderBlock,
    Embedder,
    PixelwiseNorm,
    Thre3dCoarseVolumeMakerBlock,
    Thre3dSinGanResidualDecoderBlock,
)
from thre3d_atom.networks.shared.utils import num_feature_maps_depth_decaying
from thre3d_atom.rendering.volumetric.render_interface import Rays, RenderOut
from thre3d_atom.rendering.volumetric.utils import compute_grid_sizes
from thre3d_atom.rendering.volumetric.voxels import (
    VoxelSize,
    FeatureGrid,
    GridLocation,
    render_feature_grid,
)
from thre3d_atom.utils.constants import (
    NUM_COORD_DIMENSIONS,
    NUM_COLOUR_CHANNELS,
    NUM_RGBA_CHANNELS,
)
from thre3d_atom.utils.imaging_utils import SceneBounds
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import toggle_grad


class RenderMLP(NerfNet):
    def __init__(
        self,
        point_embedder: Embedder,
        dir_embedder: Embedder,
        point_mlp: SkipMLP,
        dir_mlp: SkipMLP,
        num_coord_dimensions: Tuple[int, ...] = NUM_COORD_DIMENSIONS,
        normalize_features: bool = False,
    ) -> None:
        super().__init__(
            point_embedder, dir_embedder, point_mlp, dir_mlp, num_coord_dimensions
        )
        self._normalize_features = normalize_features
        self._normalizer = PixelwiseNorm()

    @property
    def feature_dims(self) -> int:
        return self._point_embedder.input_shape[-1]

    def get_save_info(self) -> Dict[str, Any]:
        save_info_dict = super().get_save_info()
        save_info_dict.update({"normalize_features": self._normalize_features})
        return save_info_dict

    def forward(self, x: Tensor) -> Tensor:
        """Performs the forward pass of the nerf-network. Note that point-coordinates are not used
        as an input to the network. The features implicitly encode the point co-ordinates."""
        features, directions = (
            x[:, :-NUM_COORD_DIMENSIONS],
            x[:, -NUM_COORD_DIMENSIONS:],
        )

        # normalize features if requested:
        if self._normalize_features:
            features = self._normalizer(features)

        # obtain the embedded input
        embedded_features = self._point_embedder(features)
        embedded_directions = self._dir_embedder(directions)

        # output of the point network
        point_mlp_out = self._point_mlp(embedded_features)
        density = torch.sigmoid(
            point_mlp_out[:, :1]
        )  # first value in the feature vec is density
        feature_vector = point_mlp_out[:, 1:]

        # output of the direction network
        dir_net_input = torch.cat([feature_vector, embedded_directions], dim=-1)
        colour = torch.sigmoid(self._dir_mlp(dir_net_input))

        # return the colour and density output
        return torch.cat([colour, density], dim=-1)


def get_default_render_mlp(
    feature_size: int,
    feature_embeddings_dims: int = 6,
    dir_embedding_dims: int = 4,
    normalize_features: bool = False,
) -> RenderMLP:
    point_embedder = PositionalEncodingsEmbedder(
        input_dims=feature_size, emb_dims=feature_embeddings_dims
    )
    directions_embedder = PositionalEncodingsEmbedder(
        input_dims=NUM_COORD_DIMENSIONS, emb_dims=dir_embedding_dims
    )

    point_mlp_config = SkipMLPConfig(
        input_dims=point_embedder.output_shape[-1],
        output_dims=256 + 1,  # +1 for density
        layer_depths=[256, 256, 256, 256, 256],
        skips=[False, False, True, False, False],
        dropout_prob=0.0,  # dropout is disabled
        use_equalized_learning_rate=True,
        activation_fn=Softplus(),
    )
    point_mlp = SkipMLP(point_mlp_config)

    dir_mlp_config = SkipMLPConfig(
        # -1 for density
        input_dims=(
            directions_embedder.output_shape[-1] + point_mlp.output_shape[-1] - 1
        ),
        output_dims=NUM_COLOUR_CHANNELS,
        layer_depths=[128],
        skips=[False],
        dropout_prob=0.0,  # dropout is disabled
        use_equalized_learning_rate=True,
        activation_fn=Softplus(),
    )
    dir_mlp = SkipMLP(dir_mlp_config)

    return RenderMLP(
        point_embedder=point_embedder,
        dir_embedder=directions_embedder,
        point_mlp=point_mlp,
        dir_mlp=dir_mlp,
        num_coord_dimensions=NUM_COORD_DIMENSIONS,
        normalize_features=normalize_features,
    )


def get_big_render_mlp(
    feature_size: int,
    feature_embeddings_dims: int = 6,
    dir_embedding_dims: int = 4,
    normalize_features: bool = False,
) -> RenderMLP:
    point_embedder = PositionalEncodingsEmbedder(
        input_dims=feature_size, emb_dims=feature_embeddings_dims
    )
    directions_embedder = PositionalEncodingsEmbedder(
        input_dims=NUM_COORD_DIMENSIONS, emb_dims=dir_embedding_dims
    )

    point_mlp_config = SkipMLPConfig(
        input_dims=point_embedder.output_shape[-1],
        output_dims=256 + 1,  # +1 for density
        layer_depths=[256] * 8,
        skips=[False] * 4 + [True] + [False] * 3,
        dropout_prob=0.0,  # dropout is disabled
        use_equalized_learning_rate=True,
        activation_fn=Softplus(),
    )
    point_mlp = SkipMLP(point_mlp_config)

    dir_mlp_config = SkipMLPConfig(
        # -1 for density
        input_dims=(
            directions_embedder.output_shape[-1] + point_mlp.output_shape[-1] - 1
        ),
        output_dims=NUM_COLOUR_CHANNELS,
        layer_depths=[128],
        skips=[False],
        dropout_prob=0.0,  # dropout is disabled
        use_equalized_learning_rate=True,
        activation_fn=Softplus(),
    )
    dir_mlp = SkipMLP(dir_mlp_config)

    return RenderMLP(
        point_embedder=point_embedder,
        dir_embedder=directions_embedder,
        point_mlp=point_mlp,
        dir_mlp=dir_mlp,
        num_coord_dimensions=NUM_COORD_DIMENSIONS,
        normalize_features=normalize_features,
    )


def get_tiny_render_mlp(
    feature_size: int,
    feature_embeddings_dims: int = 6,
    dir_embedding_dims: int = 4,
    normalize_features: bool = False,
) -> RenderMLP:
    point_embedder = PositionalEncodingsEmbedder(
        input_dims=feature_size, emb_dims=feature_embeddings_dims
    )
    directions_embedder = PositionalEncodingsEmbedder(
        input_dims=NUM_COORD_DIMENSIONS, emb_dims=dir_embedding_dims
    )

    point_mlp_config = SkipMLPConfig(
        input_dims=point_embedder.output_shape[-1],
        output_dims=256 + 1,  # +1 for density
        layer_depths=[256] * 2,
        skips=[False] * 2,
        dropout_prob=0.0,  # dropout is disabled
        use_equalized_learning_rate=True,
        activation_fn=Softplus(),
    )
    point_mlp = SkipMLP(point_mlp_config)

    dir_mlp_config = SkipMLPConfig(
        # -1 for density
        input_dims=(
            directions_embedder.output_shape[-1] + point_mlp.output_shape[-1] - 1
        ),
        output_dims=NUM_COLOUR_CHANNELS,
        layer_depths=[128],
        skips=[False],
        dropout_prob=0.0,  # dropout is disabled
        use_equalized_learning_rate=True,
        activation_fn=Softplus(),
    )
    dir_mlp = SkipMLP(dir_mlp_config)

    return RenderMLP(
        point_embedder=point_embedder,
        dir_embedder=directions_embedder,
        point_mlp=point_mlp,
        dir_mlp=dir_mlp,
        num_coord_dimensions=NUM_COORD_DIMENSIONS,
        normalize_features=normalize_features,
    )


class Thre3dGenerator(Network):
    def __init__(
        self,
        render_mlp_constructor: Callable[[int], RenderMLP] = get_default_render_mlp,
        base_feature_grid_dims: Tuple[int, int, int] = (4, 4, 4),
        upsample_factor: float = 2.0,
        output_feature_size: int = 32,
        num_stages: int = 7,
        fmap_base: int = 2048,
        fmap_max: int = 512,
        fmap_min: int = 32,
        fmap_decay: float = 1.0,
        use_dists_in_rendering: bool = True,
        use_trilinear_upsampling: bool = False,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        """
        Thre3d Generator model. Note that the output is compatible with the RGBA volumetric rendering
        based on alpha-compositing.
        # The [int] input to the render_mlp_constructor callable corresponds to the feature size
        """
        super().__init__()
        assert num_stages >= 1, f"Requested a depth < 1 cannot be produced :("

        # object state
        self._num_stages = num_stages
        self._output_feature_size = output_feature_size
        self._upsample_factor = upsample_factor
        self._device = device
        self._fmap_base = fmap_base
        self._fmap_max = fmap_max
        self._fmap_min = fmap_min
        self._fmap_decay = fmap_decay
        self._use_dists_in_rendering = use_dists_in_rendering
        self._use_trilinear_upsampling = use_trilinear_upsampling
        self._base_feature_grid_dims = base_feature_grid_dims

        # feature maps calculator (short-hand):
        self._nf = partial(
            num_feature_maps_depth_decaying,
            feature_maps_base=fmap_base,
            feature_maps_max=fmap_max,
            feature_maps_min=fmap_min,
            feature_maps_decay=fmap_decay,
        )

        # build the convolutional generator network:
        # fmt: off
        block_list = [Thre3dVolumeMakerBlock(output_feature_size, self._nf(1),
                                             output_feature_size, activation=LeakyReLU())]
        if num_stages >= 2:
            block_list += [self._block(stage) for stage in range(2, num_stages + 1)]
        self._generator_network = ModuleList(block_list).to(device)
        # fmt: on

        # build the render_mlp for the rendering the generated
        # volumes at different stages (grid-resolutions)
        self._render_mlp = render_mlp_constructor(self._output_feature_size).to(device)

    def _block(self, stage: int) -> Module:
        intermediate_channels = self._nf(stage)
        return Thre3dDecoderBlock(
            self._output_feature_size,
            intermediate_channels,
            self._output_feature_size,
            activation=LeakyReLU(),
            upsample_factor=self._upsample_factor,
            use_residual_connection=True,
            use_trilinear_upsampling=self._use_trilinear_upsampling,
        )

    def _get_validated_or_default_stage(self, stage: Optional[int] = None) -> int:
        if stage is not None:
            assert 1 <= stage <= self._num_stages, (
                f"requested query at stage ({stage}) "
                f"is incompatible with the network's stage range ({1, self._num_stages})"
            )
            return stage
        return self._num_stages

    @property
    def num_stages(self) -> int:
        return self._num_stages

    @property
    def scale_factor(self) -> float:
        return self._upsample_factor

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, self._output_feature_size, *self._base_feature_grid_dims

    def feature_grid_shape_at_stage(self, stage: int) -> Tuple[int, int, int]:
        w, d, h = self._base_feature_grid_dims
        for _ in range(stage - 1):
            w *= self._upsample_factor
            d *= self._upsample_factor
            h *= self._upsample_factor
            w, d, h = int(w), int(d), int(h)
        return w, d, h

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        w, d, h = self.feature_grid_shape_at_stage(self._num_stages)
        return None, self._output_feature_size, w, d, h

    def thre3d_generator(self, stage: Optional[int] = None) -> Network:
        if stage is None:
            return self._generator_network
        else:
            stage = self._get_validated_or_default_stage(stage)
            return self._generator_network[stage - 1]

    @property
    def renderer(self) -> RenderMLP:
        return self._render_mlp

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "render_mlp": self._render_mlp,
            "upsample_factor": self._upsample_factor,
            "output_feature_size": self._output_feature_size,
            "base_feature_grid_dims": self._base_feature_grid_dims,
            "num_stages": self._num_stages,
            "fmap_base": self._fmap_base,
            "fmap_max": self._fmap_max,
            "fmap_min": self._fmap_min,
            "fmap_decay": self._fmap_decay,
            "use_dists_in_rendering": self._use_dists_in_rendering,
            "use_trilinear_upsampling": self._use_trilinear_upsampling,
            "state_dict": self.state_dict(),
        }

    def _generate_feature_grid_tensor(self, input_noise: Tensor, stage: int) -> Tensor:
        assert input_noise.shape[2:] == torch.Size(self._base_feature_grid_dims), (
            f"input noise shape {input_noise.shape[2:]} doesnt' match "
            f"base feature grid dims {torch.Size(self._base_feature_grid_dims)}"
        )

        output = input_noise
        for layer_num, block in enumerate(self._generator_network[:stage]):
            if layer_num < stage - 1:
                toggle_grad(block, False)  # only the last layer is kept trainable
            output = block(output)
        return output

    def get_feature_grid(
        self,
        voxel_size: VoxelSize,
        grid_location: Optional[GridLocation] = GridLocation(),
        input_noise: Optional[Tensor] = None,
        stage: Optional[int] = None,
    ) -> FeatureGrid:
        stage = self._get_validated_or_default_stage(stage)
        if input_noise is None:
            input_noise = torch.randn(
                1, self._output_feature_size, *self._base_feature_grid_dims
            ).to(self._device)

        # obtain the feature grid using the thre3d_unet:
        feature_grid = self._generate_feature_grid_tensor(input_noise, stage).squeeze(0)

        # return a created feature-grid object
        # note that it's not tunable because the features are generated by a generator
        return FeatureGrid(
            feature_grid, voxel_size, grid_location=grid_location, tunable=False
        )

    def _are_close_enough(
        self,
        shape_1: Tuple[int, int, int],
        shape_2: Tuple[int, int, int],
    ) -> bool:
        (a, b, c), (d, e, f) = shape_1, shape_2
        abs_int = lambda x: int(np.abs(x))
        return (
            (abs_int(a - d) <= self._num_stages)
            and (abs_int(b - e) <= self._num_stages)
            and (abs_int(c - f) <= self._num_stages)
        )

    def forward(
        self,
        rays: Rays,
        voxel_size: VoxelSize,
        scene_bounds: SceneBounds,
        num_samples_per_ray: int = 64,
        input_noise: Optional[Tensor] = None,
        grid_location: Optional[GridLocation] = GridLocation(),
        density_noise_std: Optional[float] = 0.0,
        stage: Optional[int] = None,
        cached_feature_grid: Optional[FeatureGrid] = None,
    ) -> RenderOut:
        stage = self._get_validated_or_default_stage(stage)
        feature_grid = (
            cached_feature_grid
            if cached_feature_grid is not None
            else self.get_feature_grid(voxel_size, grid_location, input_noise, stage)
        )

        feature_grid_shape = (
            feature_grid.width_x,
            feature_grid.depth_z,
            feature_grid.height_y,
        )
        required_shape = self.feature_grid_shape_at_stage(stage)
        assert feature_grid_shape == torch.Size(required_shape), (
            f"feature grid shape {feature_grid_shape} and requested "
            f"stage {self.feature_grid_shape_at_stage(stage)} are incompatible :("
        )

        # render the feature_grid given the rays and num_samples_per_ray
        return render_feature_grid(
            rays=rays,
            num_samples=num_samples_per_ray,
            feature_grid=feature_grid,
            point_processor_network=self._render_mlp,
            scene_bounds=scene_bounds,
            density_noise_std=density_noise_std,
            perturb_sampled_points=True,
            use_dists_in_rendering=self._use_dists_in_rendering,
        )


# ---------------------------------------------------------------------------------------------------------------------
# Blocks for 2D SINGAN:
# ---------------------------------------------------------------------------------------------------------------------


class TwodSinGanDiscriminator(Network):
    def __init__(
        self,
        in_channels: int = NUM_COLOUR_CHANNELS,
        intermediate_channels: int = 32,
        use_eql: bool = True,
        activation: Type[torch.nn.LeakyReLU] = torch.nn.LeakyReLU(negative_slope=0.2),
    ) -> None:
        super().__init__()

        self._intermediate_channels = intermediate_channels
        self._in_channels = in_channels
        self._activation = activation

        conv_module = EqualizedConv2d if use_eql else Conv2d

        # fmt: off
        self._block = Sequential(
            conv_module(in_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1)),
            BatchNorm2d(intermediate_channels),
            activation,
            conv_module(intermediate_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1)),
            BatchNorm2d(intermediate_channels),
            activation,
            conv_module(intermediate_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1)),
            BatchNorm2d(intermediate_channels),
            activation,
            conv_module(intermediate_channels, intermediate_channels,
                        kernel_size=(3, 3), stride=(1, 1)),
            BatchNorm2d(intermediate_channels),
            activation,
            conv_module(intermediate_channels, 1,
                        kernel_size=(3, 3), stride=(1, 1)),
        )
        # fmt: on
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for layer in self._block.modules():
            if "Conv2d" in layer.__class__.__name__:
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

    @property
    def total_receptive_field(self) -> int:
        return 32

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, NUM_COLOUR_CHANNELS, None, None

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, 1, None, None

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "intermediate_channels": self._intermediate_channels,
                "in_channels": self._in_channels,
                "activation": self._activation,
            },
            "state_dict": self.state_dict(),
        }

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        # outputs the raw logits aka. scores
        return self._block(x), None


class TwodSinGanGenerator(Network):
    def __init__(
        self,
        output_resolution: Tuple[int, int] = (256, 256),
        num_channels: int = NUM_COLOUR_CHANNELS,
        num_intermediate_channels: int = 32,
        num_stages: int = 8,
        scale_factor: float = (1 / 0.75),
        use_eql: bool = True,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        assert (
            num_stages >= 1
        ), f"Num-stages for the 2d-Singan Generator cannot be <= 1. requested: {num_stages}"

        super().__init__()
        self._output_resolution = output_resolution
        self._num_channels = num_channels
        self._num_intermediate_channels = num_intermediate_channels
        self._num_stages = num_stages
        self._scale_factor = scale_factor
        self._use_eql = use_eql
        self._device = device

        # create a list of the required output_sizes for the given scale_factor
        self._image_sizes = self.compute_image_sizes(
            self._output_resolution, self._num_stages, self._scale_factor
        )

        # a short-hand for the base noise_size:
        self._starting_noise_shape = (1, self._num_channels, *self._image_sizes[0])

        # make a fixed_noise_tensor for performing reconstruction:
        self._reconstruction_noise = torch.randn(
            self._starting_noise_shape, device=device
        )

        # the blocks needed for generation:
        # first stage:
        block_list = [
            TwodCoarseImageMakerBlock(
                in_channels=self._num_channels,
                intermediate_channels=self._num_intermediate_channels,
                use_eql=self._use_eql,
            )
        ]
        # stage two onwards:
        block_list += [
            TwodSinGanResidualDecoderBlock(
                in_channels=self._num_channels,
                intermediate_channels=self._num_intermediate_channels,
                use_eql=self._use_eql,
            )
            for _ in range(2, num_stages + 1)
        ]
        self._generator_blocks = ModuleList(block_list).to(device)

    @staticmethod
    def compute_image_sizes(
        output_resolution: Tuple[int, int], num_stages: int, scale_factor: float
    ) -> List[Tuple[int, int]]:
        h, w = output_resolution
        image_sizes = [(h, w)]
        for _ in range(num_stages - 1):
            h = int(np.ceil((1 / scale_factor) * h))
            w = int(np.ceil((1 / scale_factor) * w))
            image_sizes.insert(0, (h, w))
        return image_sizes

    def get_block_at_stage(self, stage: int) -> Module:
        stage = self._get_validated_or_default_stage(stage)
        return self._generator_blocks[stage - 1]

    def load_block_at_stage(self, block: Module, stage) -> None:
        stage = self._get_validated_or_default_stage(stage)
        try:
            self._generator_blocks[stage - 1].load_state_dict(block.state_dict())
        except RuntimeError:
            log.info(
                "Tried loading the block's weights into the requested stage, but some keys were missing"
            )
            self._generator_blocks[stage - 1].load_state_dict(
                block.state_dict(), strict=False
            )
        # reset the noise_controllers at the requested stage to zero again:
        if hasattr(self._generator_blocks[stage - 1], "noise_controller"):
            with torch.no_grad():
                self._generator_blocks[
                    stage - 1
                ].noise_controller.data = torch.zeros_like(
                    self._generator_blocks[stage - 1].noise_controller
                )
            log.info(
                f"Zeroed out the noise_controller: {self._generator_blocks[stage - 1].noise_controller}"
            )

    @property
    def image_sizes(self) -> List[Tuple[int, int]]:
        return self._image_sizes

    @property
    def reconstruction_noise(self) -> Tensor:
        return self._reconstruction_noise

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, self._num_channels, None, None

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, self._num_channels, None, None

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "num_stages": self._num_stages,
                "output_resolution": self._output_resolution,
                "num_channels": self._num_channels,
                "num_intermediate_channels": self._num_intermediate_channels,
                "scale_factor": self._scale_factor,
                "use_eql": self._use_eql,
            },
            "state_dict": self.state_dict(),
        }

    def _get_validated_or_default_stage(self, stage: Optional[int] = None) -> int:
        if stage is not None:
            assert 1 <= stage <= self._num_stages, (
                f"requested query at stage ({stage}) "
                f"is incompatible with the network's stage range ({1, self._num_stages})"
            )
            return stage
        return self._num_stages

    def forward(
        self, input_noise: Optional[Tensor] = None, stage: Optional[int] = None
    ) -> Tensor:
        stage = self._get_validated_or_default_stage(stage)
        if input_noise is None:
            input_noise = torch.randn(self._starting_noise_shape, device=self._device)

        # only the last block is kept trainable. Rest other blocks are frozen
        for block_stage, block in enumerate(self._generator_blocks, start=1):
            if block_stage == stage:
                toggle_grad(block, True)
            else:
                toggle_grad(block, False)

        # apply the blocks till the required stage is reached
        output_image = self._generator_blocks[0](input_noise)
        for block, image_size in zip(
            self._generator_blocks[1:stage], self._image_sizes[1:stage]
        ):
            output_image = block(
                interpolate(
                    output_image, size=image_size, mode="bilinear", align_corners=False
                )
            )

        return output_image


# ---------------------------------------------------------------------------------------------------------------------
# Blocks for Direct Supervision 3D SINGAN:
# ---------------------------------------------------------------------------------------------------------------------


class Thre3dSinGanDiscriminatorDS(Network):
    def __init__(
        self,
        required_receptive_field: Tuple[int, int, int],
        num_layers: int = 5,
        in_features: int = NUM_RGBA_CHANNELS,
        intermediate_features: int = 32,
        use_eql: bool = True,
        activation: Type[torch.nn.LeakyReLU] = torch.nn.LeakyReLU(negative_slope=0.2),
    ) -> None:
        super().__init__()

        assert num_layers >= 2, "No point in creating a discriminator with < 2 layers"

        self.intermediate_features = intermediate_features
        self._num_layers = num_layers
        self._in_features = in_features
        self._activation = activation
        self._use_eql = use_eql

        conv_module = EqualizedConv3d if use_eql else Conv3d

        # compute kernel_size based on the required_receptive_field
        # and the num_layers. Also notify the user if the requested receptive field
        # is not possible
        self._actual_receptive_field, self._kernel_size = [], []
        modified = False
        for dim in required_receptive_field:
            if dim % num_layers != 0:
                modified = True
            kernel_size = int(np.floor(dim / num_layers)) + 1
            self._kernel_size.append(kernel_size)
            self._actual_receptive_field.append(
                kernel_size + ((kernel_size - 1) * (num_layers - 1))
            )
        if modified:
            log.info(
                f"The requested receptive field ({required_receptive_field}) has been "
                f"modified to ({self._actual_receptive_field}) as an approximation"
            )

        # create the blocks list:
        block_list = [  # first block(s)
            conv_module(
                in_features,
                intermediate_features,
                kernel_size=self._kernel_size,
                stride=(1, 1, 1),
            ),
            BatchNorm3d(intermediate_features),
            activation,
        ]

        # intermediate blocks:
        for _ in range(self._num_layers - 2):
            block_list.append(
                conv_module(
                    intermediate_features,
                    intermediate_features,
                    kernel_size=self._kernel_size,
                    stride=(1, 1, 1),
                )
            )
            block_list.append(BatchNorm3d(intermediate_features))
            block_list.append(activation)

        # score giver block:
        block_list.append(
            conv_module(
                intermediate_features,
                1,
                kernel_size=self._kernel_size,
                stride=(1, 1, 1),
            )
        )

        self._block = Sequential(*block_list)

        # initialize the weights according to the stable strategy
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

    @property
    def receptive_field(self) -> Tuple[int, int, int]:
        return self._actual_receptive_field

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, self._in_features, None, None, None

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, 1, None, None, None

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "required_receptive_field": self._actual_receptive_field,
                "num_layers": self._num_layers,
                "in_features": self._in_features,
                "intermediate_features": self.intermediate_features,
                "use_eql": self._use_eql,
                "activation": self._activation,
            },
            "state_dict": self.state_dict(),
        }

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        # outputs the raw logits aka. scores
        return self._block(x), None


class Thre3dSinGanGeneratorDS(Network):
    def __init__(
        self,
        output_resolution: Tuple[int, int, int] = (128, 128, 128),
        num_features: int = NUM_RGBA_CHANNELS,
        num_intermediate_features: Union[int, List[int]] = 32,
        num_stages: int = 8,
        scale_factor: float = (1 / 0.75),
        use_eql: bool = True,
        noise_scale: float = 0.1,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        assert (
            num_stages >= 1
        ), f"Num-stages for the 2d-Singan Generator cannot be <= 1. requested: {num_stages}"

        if isinstance(num_intermediate_features, list):
            assert len(num_intermediate_features) == num_stages, (
                f"Provided num_intermediate_features list is incompatible with requested num_stages"
                f"num_intermediate_features: {num_intermediate_features} \n num_stages: {num_stages}"
            )
        else:
            # build a list using same number of features for all stages
            num_intermediate_features = [num_intermediate_features] * num_stages

        super().__init__()
        self._output_resolution = output_resolution
        self._num_features = num_features
        self._num_intermediate_features = num_intermediate_features
        self._num_stages = num_stages
        self._scale_factor = scale_factor
        self._use_eql = use_eql
        self._noise_scale = noise_scale
        self._device = device

        log.info(
            f"!!!Juggernaut!!! Using value of for the noise-scale: {self._noise_scale}"
        )

        # create a list of the required output_sizes for the given scale_factor
        self._grid_sizes = compute_grid_sizes(
            self._output_resolution, self._num_stages, self._scale_factor
        )

        # the blocks needed for generation:
        # first stage:
        block_list = [
            Thre3dCoarseVolumeMakerBlock(
                in_features=self._num_features,
                intermediate_features=self._num_intermediate_features[0],
                use_eql=self._use_eql,
            )
        ]
        # stage two onwards:
        block_list += [
            Thre3dSinGanResidualDecoderBlock(
                in_features=self._num_features,
                intermediate_features=intermediate_features,
                use_eql=self._use_eql,
            )
            for intermediate_features in self._num_intermediate_features[1:]
        ]
        self._generator_blocks = ModuleList(block_list).to(device)
        self._noise_amps = [0.0] * (len(self._generator_blocks))

        # a short-hand for the base noise_size:
        # starting_noise_padding = 2 * self._generator_blocks[0].padding_dims
        self._starting_noise_shape = (1, 1, *self._grid_sizes[0])

        # make a fixed_noise_tensor for performing reconstruction:
        self._reconstruction_noise = torch.randn(
            self._starting_noise_shape, device=device
        )

    @property
    def noise_amps(self) -> List[float]:
        return self._noise_amps

    @noise_amps.setter
    def noise_amps(self, noise_amps: List[float]) -> None:
        assert len(noise_amps) == len(
            self._noise_amps
        ), f"provided noise_amps are incompatible with the Generator's Noise_amps"
        self._noise_amps = noise_amps

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, self._num_features, None, None, None

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, self._num_features, None, None, None

    @property
    def grid_sizes(self) -> List[Tuple[int, int, int]]:
        return self._grid_sizes

    @property
    def reconstruction_noise(self) -> Tensor:
        return self._reconstruction_noise

    def get_block_at_stage(self, stage: int) -> Module:
        stage = self._get_validated_or_default_stage(stage)
        return self._generator_blocks[stage - 1]

    def load_block_at_stage(self, block: Module, stage) -> None:
        stage = self._get_validated_or_default_stage(stage)
        if stage > 1:
            self._generator_blocks[stage - 1].load_state_dict(block.state_dict())
        else:
            log.info(f"Requested module is at stage {stage}. So, skipped loading")

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "output_resolution": self._output_resolution,
                "num_features": self._num_features,
                "num_intermediate_features": self._num_intermediate_features,
                "num_stages": self._num_stages,
                "scale_factor": self._scale_factor,
                "use_eql": self._use_eql,
                "noise_scale": self._noise_scale,
            },
            "noise_amps": self._noise_amps,
            "state_dict": self.state_dict(),
        }

    def _get_validated_or_default_stage(self, stage: Optional[int] = None) -> int:
        if stage is not None:
            assert 1 <= stage <= self._num_stages, (
                f"requested query at stage ({stage}) "
                f"is incompatible with the network's stage range ({1, self._num_stages})"
            )
            return stage
        return self._num_stages

    def forward(
        self,
        input_noise: Optional[Tensor] = None,
        stage: Optional[int] = None,
        apply_noise: bool = True,
        use_intermediate_fixed_noise: bool = False,
        noise_multiplier: Optional[float] = None,
    ) -> Tensor:
        stage = self._get_validated_or_default_stage(stage)
        if input_noise is None:
            input_noise = torch.randn(self._starting_noise_shape, device=self._device)

        # tile the input noise channelwise
        input_noise = input_noise.repeat(1, self._num_features, 1, 1, 1)

        # only the last block is kept trainable. Rest other blocks are frozen
        for block_stage, block in enumerate(self._generator_blocks, start=1):
            if block_stage == stage:
                toggle_grad(block, True)
            else:
                toggle_grad(block, False)

        if stage > 1 and noise_multiplier is not None:
            self._noise_amps[stage - 1] = self._noise_scale * noise_multiplier

        grid_sizes = self._grid_sizes
        if input_noise.shape[2:] != self._reconstruction_noise.shape[2:]:
            # we need to recompute the grid_sizes:
            output_resolution = [
                int(out_res_dim / recon_noise_dim * noise_dim)
                for out_res_dim, recon_noise_dim, noise_dim in zip(
                    self._output_resolution,
                    self._reconstruction_noise.shape[2:],
                    input_noise.shape[2:],
                )
            ]
            grid_sizes = compute_grid_sizes(
                output_resolution, self._num_stages, self._scale_factor
            )

        # apply the blocks till the required stage is reached
        output_volume = self._generator_blocks[0](input_noise)
        for block, grid_size, noise_amp in zip(
            self._generator_blocks[1:stage],
            grid_sizes[1:stage],
            self._noise_amps[1:stage],
        ):
            output_volume = block(
                interpolate(
                    output_volume, size=grid_size, mode="trilinear", align_corners=False
                ),
                noise_multiplier=noise_amp if apply_noise else 0.0,
                use_fixed_noise=use_intermediate_fixed_noise,
            )

        return output_volume
