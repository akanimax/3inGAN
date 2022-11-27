""" manually written implementation for voxel-based volumetric models """
from functools import partial
from typing import Tuple, NamedTuple, Optional, Callable, Union

import numpy as np
import torch
from torch import Tensor, conv3d
from torch.nn import Module
from torch.nn.functional import interpolate, grid_sample

from thre3d_atom.data.loaders import PosedImagesDataset
from thre3d_atom.networks.network_interface import Network
from thre3d_atom.rendering.volumetric.implicit import (
    accumulate_processed_points_on_rays,
    raw2alpha_base,
    accumulate_processed_points_on_rays_with_msfg_bg,
)
from thre3d_atom.rendering.volumetric.render_interface import (
    Rays,
    RenderOut,
    render,
    SampledPointsOnRays,
    ProcessedPointsOnRays,
)
from thre3d_atom.utils.types import AxisAlignedBoundingBox
from thre3d_atom.rendering.volumetric.sample import (
    sample_cdf_weighted_points_on_rays,
    sample_uniform_points_on_regular_feature_grid,
    sample_uniform_points_on_rays,
    sample_uniform_points_with_inverted_sphere_background,
    sample_uniform_points_on_regular_feature_grid_with_background,
)
from thre3d_atom.rendering.volumetric.spherical_harmonics import eval_sh
from thre3d_atom.utils.constants import (
    EXTRA_POINT_WEIGHTS,
    EXTRA_POINT_DEPTHS,
    NUM_COORD_DIMENSIONS,
    NUM_RGBA_CHANNELS,
    NUM_COLOUR_CHANNELS,
)
from thre3d_atom.utils.imaging_utils import SceneBounds, adjust_dynamic_range

# ----------------------------------------------------------------------------------------------------------------------
# Definition of a FeatureGrid
# ----------------------------------------------------------------------------------------------------------------------
from thre3d_atom.utils.misc import batchify


class VoxelSize(NamedTuple):
    # lengths of a single voxel's edges in the x, y and z dimensions
    x_size: float = 1.0
    y_size: float = 1.0
    z_size: float = 1.0


class GridLocation(NamedTuple):
    # indicates where the Grid is located in World Coordinate System
    # i.e. indicates where the centre of the grid is located in the World
    # The Grid is always assumed to be axis aligned.
    x_coord: float = 0.0
    y_coord: float = 0.0
    z_coord: float = 0.0


class FeatureGrid(Module):
    def __init__(
        self,
        features: Tensor,
        voxel_size: VoxelSize,
        grid_location: Optional[GridLocation] = GridLocation(),
        tunable: bool = False,
        preactivation: Optional[Callable[[Tensor], Tensor]] = torch.nn.Tanh(),
        colour_preactivation: Optional[Callable[[Tensor], Tensor]] = torch.nn.Sigmoid(),
        use_sh: bool = False,
    ):
        """
        Defines a Feature Grid denoting a 3D-volume. To obtain features of a particular point inside
        the volume, we find the enclosing vertices and obtain the features by doing trilinear interpolation
        Args:
            features: Tensor of shape [F x W x H x D] corresponds to the features on the grid-vertices
                                          [X x Y x Z]
            voxel_size: Size of each voxel. (could be different in different axis (x, y, z))
            grid_location: Location of the center of the grid
            tunable: whether to treat the features Tensor as a tunable (trainable) parameter
            preactivation: the activation to be applied to the raw features before interpolating.
                           This is a means of constraining the values of the interpolated features.
            colour_preactivation: the activation to be applied to the raw colour features before interpolating.
            use_sh: whether to use the as coefficients for spherical harmonics.
        """
        assert (
            len(features.shape) == 4
        ), f"features should be of shape [F x W x H x D] as opposed to ({features.shape})"

        super().__init__()
        self._preactivation = preactivation
        self._colour_preactivation = colour_preactivation
        self._use_sh = use_sh

        # We permute the feature dimension to the last
        self._features = features.permute(1, 2, 3, 0)
        # The three spatial dimensions correspond to X, Y, and Z
        self.tunable = tunable
        if tunable:
            self._features = torch.nn.Parameter(self._features)

        self.grid_location = grid_location
        self.tunable = tunable
        self.voxel_size = voxel_size
        self._device = features.device

        # note the x, y and z conventions for the width, height and depth
        self.width_x, self.height_y, self.depth_z = (
            self._features.shape[0],
            self._features.shape[1],
            self._features.shape[2],
        )

        # setup the bounding box planes
        self._setup_bounding_box_planes()

    @property
    def features(self) -> Tensor:
        return self._features  # shape is W x H x D x F

    @features.setter
    def features(self, features: Tensor) -> None:
        assert self._features.shape == features.shape
        if not self.tunable:
            self._features = features
        else:
            if not isinstance(features, torch.nn.Parameter):
                self._features = torch.nn.Parameter(features)

    @property
    def grid_dims(self) -> Tuple[int, int, int]:
        return self._features.shape[:-1]

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        return AxisAlignedBoundingBox(
            x_range=self._width_x_range,
            y_range=self._height_y_range,
            z_range=self._depth_z_range,
        )

    def extra_repr(self) -> str:
        return (
            f"grid_dims: {self._features.shape[:-1]}, "
            f"feature_dims: {self._features.shape[-1]}, "
            f"voxel_size: {self.voxel_size}, "
            f"grid_location: {self.grid_location}"
        )

    def _setup_bounding_box_planes(self):
        half_width = ((self.width_x - 1) * self.voxel_size.x_size) / 2
        half_height = ((self.height_y - 1) * self.voxel_size.y_size) / 2
        half_depth = ((self.depth_z - 1) * self.voxel_size.z_size) / 2
        self._width_x_range = (
            self.grid_location.x_coord - half_width,
            self.grid_location.x_coord + half_width,
        )
        self._height_y_range = (
            self.grid_location.y_coord - half_height,
            self.grid_location.y_coord + half_height,
        )
        self._depth_z_range = (
            self.grid_location.z_coord - half_depth,
            self.grid_location.z_coord + half_depth,
        )

    def get_bounding_vertices(self) -> Tensor:
        x_min, x_max = self._width_x_range
        y_min, y_max = self._depth_z_range
        z_min, z_max = self._height_y_range
        return torch.tensor(
            [
                [x_min, y_min, z_min],
                [x_min, y_min, z_max],
                [x_min, y_max, z_min],
                [x_min, y_max, z_max],
                [x_max, y_min, z_min],
                [x_max, y_min, z_max],
                [x_max, y_max, z_min],
                [x_max, y_max, z_max],
            ],
            dtype=torch.float32,
        )

    def get_all_feature_grid_coordinates(self) -> np.array:
        volume_origin = torch.tensor(
            [
                self._width_x_range[0],
                self._depth_z_range[0],
                self._height_y_range[0],
            ],
            dtype=torch.float32,
        )
        voxel_scale = torch.tensor(
            [self.voxel_size.x_size, self.voxel_size.y_size, self.voxel_size.z_size],
            dtype=torch.float32,
        )

        feature_indices = torch.stack(
            torch.meshgrid(
                torch.arange(self.width_x),
                torch.arange(self.depth_z),
                torch.arange(self.height_y),
            )
        ).permute(1, 2, 3, 0)

        feature_rel_coords = feature_indices * voxel_scale
        feature_abs_coords = feature_rel_coords + volume_origin
        return feature_abs_coords.reshape(-1, feature_abs_coords.shape[-1]).numpy()

    def is_inside_volume(self, points: Tensor) -> Tensor:
        """ AABB test on the entire scene volume """
        return torch.logical_and(
            torch.logical_and(
                torch.logical_and(
                    points[..., 0:1] > self._width_x_range[0],
                    points[..., 0:1] < self._width_x_range[1],
                ),
                torch.logical_and(
                    points[..., 1:2] > self._height_y_range[0],
                    points[..., 1:2] < self._height_y_range[1],
                ),
            ),
            torch.logical_and(
                points[..., 2:] > self._depth_z_range[0],
                points[..., 2:] < self._depth_z_range[1],
            ),
        )

    def _normalize_points(self, points: Tensor) -> Tensor:
        normalized_points = torch.empty_like(points, device=points.device)
        normalized_points[:, 0] = adjust_dynamic_range(
            points[:, 0],
            drange_in=self._width_x_range,
            drange_out=(-1 + (1 / self.width_x), 1 - (1 / self.width_x)),
            slack=True,
        )
        normalized_points[:, 1] = adjust_dynamic_range(
            points[:, 1],
            drange_in=self._height_y_range,
            drange_out=(-1 + (1 / self.height_y), 1 - (1 / self.height_y)),
            slack=True,
        )
        normalized_points[:, 2] = adjust_dynamic_range(
            points[:, 2],
            drange_in=self._depth_z_range,
            drange_out=(-1 + (1 / self.depth_z), 1 - (1 / self.depth_z)),
            slack=True,
        )
        return normalized_points

    def forward(
        self,
        points: Tensor,
        opacity_masking: bool = True,
    ) -> Tensor:
        normalized_points = self._normalize_points(points)
        preactivated_features = self._features
        if self._features.shape[-1] == NUM_RGBA_CHANNELS or self._use_sh:
            # apply preactivation to the density:
            colour, density = self._features[..., :-1], self._features[..., -1:]
            colour = (
                self._colour_preactivation(colour)
                if self._colour_preactivation is not None
                else colour
            )
            density = (
                self._preactivation(density)
                if self._preactivation is not None
                else density
            )
            preactivated_features = torch.cat([colour, density], dim=-1)

        if opacity_masking:
            # we use the grid density to mask colours/features in the free space for better (correct) interpolation
            colour, density = (
                preactivated_features[..., :-1],
                preactivated_features[..., -1:],
            )
            opacity_mask = (density > 0.0).type(torch.float32)

            # apply 1 voxel dilation to the opacity mask:
            dilated_opacity_mask = torch.nn.functional.conv3d(
                opacity_mask.permute(3, 0, 1, 2)[None, ...],
                torch.ones(1, 1, 3, 3, 3, device=self._device, requires_grad=False),
                padding=1,
            )[0].permute(1, 2, 3, 0)
            dilated_opacity_mask = torch.clip(dilated_opacity_mask, 0.0, 1.0)

            colour = colour * dilated_opacity_mask
            preactivated_features = torch.cat([colour, density], dim=-1)

        # ===========================================================================================
        # OMG!!! Note all these operations that are required because of PyTorch's
        # grid_sample convention!!!
        # ===========================================================================================
        # permute the grid so that it corresponds to our X-Y-Z axes convention
        features_for_interpolation = preactivated_features.permute(2, 1, 0, 3)
        # note to flip the Z-axis and the Y-axis
        features_for_interpolation = torch.flip(features_for_interpolation, dims=(0,))
        # ===========================================================================================

        features = (
            grid_sample(
                features_for_interpolation[None, ...].permute(0, 4, 1, 2, 3),
                normalized_points[None, None, None, ...],
                align_corners=False,
            )
            .permute(0, 2, 3, 4, 1)
            .squeeze()
        )

        return features


# noinspection PyUnusedLocal
class MultiSphereFeatureGrid(FeatureGrid):
    def __init__(
        self,
        features: Tensor,
        _1: VoxelSize = None,
        _2: GridLocation = None,
        tunable: bool = False,
        preactivation: Callable[[Tensor], Tensor] = torch.nn.Tanh(),
        colour_preactivation: Callable[[Tensor], Tensor] = torch.nn.Sigmoid(),
        _3: bool = False,
    ):
        assert (
            features.shape[0] == NUM_RGBA_CHANNELS
        ), f"MSIFeatureGrid only supports RGBA features at the moment"

        super().__init__(
            features,
            voxel_size=VoxelSize(),
            grid_location=GridLocation(),
            tunable=tunable,
            preactivation=preactivation,
            colour_preactivation=colour_preactivation,
            use_sh=False,
        )

    def forward(self, points: Tensor, _: bool = False) -> Tensor:
        """performs appropriate warping of points and applies interpolation
        assume points are [N_rays x num_samples_per_ray x NUM_COORD_DIMENSIONS]
        please note that this is different from the other FeatureGrids here
        """
        num_rays, num_samples_per_ray, num_coords = points.shape
        point_radii = points.norm(dim=-1)

        # assert torch.all(
        #     point_radii >= 1.0 - ZERO_PLUS
        # ), f"points inside the unit_sphere are being interpolated on MultiSphereFeatureGrid"

        # perform equirectangular warping of the points
        x, y, z, r = points[..., 0], points[..., 1], points[..., 2], point_radii
        latitude = adjust_dynamic_range(
            torch.asin(z / r), drange_in=(-np.pi / 2, np.pi / 2), drange_out=(-1, 1)
        )
        longitude = adjust_dynamic_range(
            np.pi + torch.atan2(y, x), drange_in=(0, 2 * np.pi), drange_out=(-1, 1)
        )
        depth = torch.linspace(
            -1.0, 1.0, num_samples_per_ray, device=longitude.device, dtype=torch.float32
        )[None, :].tile(num_rays, 1)
        warped_points = torch.stack([latitude, longitude, depth], dim=-1)

        # use the warped points to interpolate the values on the grid:
        preactivated_features = self._features
        if self._features.shape[-1] == NUM_RGBA_CHANNELS or self._use_sh:
            # apply preactivation to the density:
            colour, density = self._features[..., :-1], self._features[..., -1:]
            colour = self._colour_preactivation(colour)
            density = self._preactivation(density)
            preactivated_features = torch.cat([colour, density], dim=-1)

        flat_warped_points = warped_points.reshape(-1, num_coords)
        features = (
            grid_sample(
                preactivated_features[None, ...].permute(0, 4, 3, 2, 1),
                flat_warped_points[None, None, None, ...],
                align_corners=True,
            )
            .permute(0, 2, 3, 4, 1)
            .squeeze()
        )

        return features


class HybridRGBAFeatureGrid(FeatureGrid):
    def forward(self, points: Tensor, rgba_only: bool = False) -> Tensor:
        """obtain the features for the given points.
        Procedure =>
        Note that, points need to be a flat tensor of shape: [Num_points x 3]
        1. Find the eight features in the _features tensor whose voxel tightly bounds the point(s)
        2. Tri-linearly interpolate between the eight features
        """
        # mask the points which lie outside of the bounding volume
        inside_points_mask = self.is_inside_volume(points)
        inside_points = points * inside_points_mask.float()
        normalized_inside_points = self._normalize_points(inside_points)

        if rgba_only:
            grid_features = self._features[..., :NUM_RGBA_CHANNELS]
        else:
            # please note to detach the direct alpha values before masking features
            with torch.no_grad():
                # pick the direct alpha values for masking the features
                a_features = torch.clip(
                    self._features[..., NUM_RGBA_CHANNELS - 1 : NUM_RGBA_CHANNELS],
                    0.0,
                    1.0,
                )
                # apply a single dilation pass:
                a_features = torch.clip(
                    conv3d(
                        a_features[None, ...].permute(0, 4, 1, 2, 3),
                        torch.ones(1, 1, 3, 3, 3, device=a_features.device),
                        padding=1,
                    ),
                    0.0,
                    1.0,
                )[0].permute(1, 2, 3, 0)
            grid_features = a_features * self._features[..., NUM_RGBA_CHANNELS:]

        features = (
            grid_sample(
                grid_features[None, ...].permute(0, 4, 3, 2, 1),
                normalized_inside_points[None, None, None, ...],
                align_corners=False,
            )
            .permute(0, 2, 3, 4, 1)
            .squeeze()
        )

        # note to use -ve infinity instead of zeros because of the activation functions on top
        minus_infinity_features = torch.full(
            features.shape, -100.0, device=self._device
        )
        return torch.where(inside_points_mask, features, minus_infinity_features)


# ----------------------------------------------------------------------------------------------------------------------
# Rendering and general utilities for the FeatureGrid
# ----------------------------------------------------------------------------------------------------------------------


def scale_feature_grid_with_output_size(
    feature_grid: FeatureGrid,
    output_size: Tuple[int, int, int],
    # mode has to be trilinear if output_size is used
) -> FeatureGrid:
    # obtain the features, voxel_size and grid_location from the original feature grid
    og_features = feature_grid.features.permute(3, 0, 1, 2)[None, ...]
    og_voxel_size = feature_grid.voxel_size
    og_grid_location = feature_grid.grid_location

    # new features are upsampled/downsampled by a factor of scale_factor
    # noinspection PyArgumentList
    # from scipy.ndimage import zoom
    #
    # x_size, y_size, z_size = og_features.shape[2:]
    #
    # new_features = torch.from_numpy(
    #     zoom(
    #         og_features.detach().cpu().numpy(),
    #         (
    #             1,
    #             1,
    #             output_size[0] / x_size,
    #             output_size[1] / y_size,
    #             output_size[2] / z_size,
    #         ),
    #     ),
    # )[0].to(og_features.device)

    new_features = interpolate(
        og_features,
        size=output_size,
        mode="trilinear",
        align_corners=False,
        recompute_scale_factor=False,
    )[0]
    # no need to permute this back again, because the FeatureGrid expects an F x W x D x H tensor

    assert new_features.shape[1:] == output_size

    # new voxel size is also similarly scaled
    new_voxel_size = VoxelSize(
        og_voxel_size.x_size * (og_features.shape[2] / output_size[0]),
        og_voxel_size.y_size * (og_features.shape[3] / output_size[1]),
        og_voxel_size.z_size * (og_features.shape[4] / output_size[2]),
    )

    # make sure that the same class is output
    # noinspection PyProtectedMember
    return feature_grid.__class__(
        new_features,
        new_voxel_size,
        og_grid_location,
        feature_grid.tunable,
        feature_grid._preactivation,
        feature_grid._colour_preactivation,
        feature_grid._use_sh,
    )


def scale_feature_grid_with_scale_factor(
    feature_grid: FeatureGrid, scale_factor: float, mode: str = "trilinear"
) -> FeatureGrid:
    # obtain the features, voxel_size and grid_location from the original feature grid
    og_features = feature_grid.features
    og_voxel_size = feature_grid.voxel_size
    og_grid_location = feature_grid.grid_location

    # new features are unsampled/downsampled by a factor of scale_factor
    # noinspection PyArgumentList
    new_features = interpolate(
        og_features.permute(3, 0, 1, 2)[None, ...],
        scale_factor=scale_factor,
        mode=mode,
        align_corners=False if mode.lower() != "nearest" else None,
        recompute_scale_factor=False,
    )[0]
    # no need to permute this back again, because the FeatureGrid expects a F x W x H x D tensor

    # new voxel size is also similarly scaled
    og_x_dim, og_y_dim, og_z_dim, _ = og_features.shape
    _, new_x_dim, new_y_dim, new_z_dim = new_features.shape
    new_voxel_size = VoxelSize(
        (og_voxel_size.x_size * (og_x_dim - 1)) / (new_x_dim - 1),
        (og_voxel_size.y_size * (og_y_dim - 1)) / (new_y_dim - 1),
        (og_voxel_size.z_size * (og_z_dim - 1)) / (new_z_dim - 1),
    )

    # make sure that the same class is output
    # noinspection PyProtectedMember
    return feature_grid.__class__(
        new_features,
        new_voxel_size,
        og_grid_location,
        feature_grid.tunable,
        feature_grid._preactivation,
        feature_grid._colour_preactivation,
        feature_grid._use_sh,
    )


def get_voxel_size_from_scene_bounds_and_hem_rad(
    hem_rad: float,
    grid_dim: int,
    scene_bounds: SceneBounds,
) -> VoxelSize:
    # note below that we leave some room around the tight scene geometry
    # we use (0.8 * scene_bounds.near) instead of scene_bounds.near
    cube_diagonal = hem_rad - (0.8 * scene_bounds.near)
    cube_dim = 2 * np.sqrt((cube_diagonal ** 2) / 3)
    return VoxelSize(
        *(
            cube_dim
            / torch.tensor(
                [(grid_dim - 1), (grid_dim - 1), (grid_dim - 1)], dtype=torch.float32
            )
        )
    )


def get_voxel_size_from_scene_bounds_and_dataset(
    dataset: PosedImagesDataset,
    grid_dim: int,
    scene_bounds: Optional[SceneBounds] = None,
) -> VoxelSize:
    scene_bounds = dataset.scene_bounds if scene_bounds is None else scene_bounds
    # note below that we leave some room around the tight scene geometry
    # we use (0.8 * scene_bounds.near) instead of scene_bounds.near
    cube_diagonal = dataset.get_hemispherical_radius_estimate() - (
        0.8 * scene_bounds.near
    )
    cube_dim = 2 * np.sqrt((cube_diagonal ** 2) / 3)
    return VoxelSize(
        *(
            cube_dim
            / torch.tensor(
                [(grid_dim - 1), (grid_dim - 1), (grid_dim - 1)], dtype=torch.float32
            )
        )
    )


def _rgba_extraction(
    flat_sampled_points: Tensor,
    feature_grid: FeatureGrid,
    background_mlp: Optional[Network] = None,
) -> Tensor:
    if background_mlp is not None:
        return torch.where(
            feature_grid.is_inside_volume(flat_sampled_points),
            feature_grid(
                flat_sampled_points[..., :NUM_COORD_DIMENSIONS], rgba_only=True
            )
            if isinstance(feature_grid, HybridRGBAFeatureGrid)
            else feature_grid(flat_sampled_points[..., :NUM_COORD_DIMENSIONS]),
            background_mlp(flat_sampled_points),
        )
    else:
        return (
            feature_grid(
                flat_sampled_points[..., :NUM_COORD_DIMENSIONS], rgba_only=True
            )
            if isinstance(feature_grid, HybridRGBAFeatureGrid)
            else feature_grid(flat_sampled_points[..., :NUM_COORD_DIMENSIONS])
        )


def _point_processing(
    network_input: Tensor,
    feature_grid: FeatureGrid,
    processor_network: Network,
    background_processor_network: Optional[Network] = None,
) -> Tensor:
    # obtain the trilinearly sampled features from the feature grid
    flat_sampled_points = network_input[:, :NUM_COORD_DIMENSIONS]
    features = feature_grid(flat_sampled_points)

    # note that for the points outside the bounding_volume, we
    # output -INFINITY (RGBA) raw values which after activation correspond to empty space
    minus_infinity_features = torch.full(
        size=(flat_sampled_points.shape[0], processor_network.output_shape[-1]),
        fill_value=-100.0,
        device=flat_sampled_points.device,
    )

    # note that the point-coordinates are not input to the network
    # we only use the points for obtaining their associated features from
    # the FeatureGrid
    return torch.where(
        torch.logical_and(
            feature_grid.is_inside_volume(flat_sampled_points),
            features.abs().sum(dim=-1, keepdim=True) > 0.0,
        ),
        processor_network(torch.cat([features, network_input[:, 3:]], dim=-1)),
        background_processor_network(network_input)
        if background_processor_network is not None
        else minus_infinity_features,
    )


def implicit_feature_grid_point_processor(
    sampled_points: SampledPointsOnRays,
    rays: Rays,
    feature_grid: FeatureGrid,
    processor_network: Network,
    background_processor_network: Optional[Network],
    chunk_size: int = 1024 * 64,
) -> ProcessedPointsOnRays:
    num_rays, num_samples_per_ray, num_coords = sampled_points.points.shape
    flat_sampled_points = sampled_points.points.reshape(-1, num_coords)

    # obtain the processed_points
    viewdirs = rays.directions / rays.directions.norm(dim=-1, keepdim=True)
    viewdirs_tiled = (
        viewdirs[:, None, :].repeat(1, num_samples_per_ray, 1).reshape(-1, num_coords)
    )
    network_input = torch.cat([flat_sampled_points, viewdirs_tiled], dim=-1)

    # create a batchified processing pipeline
    batchified_point_processing = batchify(
        partial(
            _point_processing,
            feature_grid=feature_grid,
            processor_network=processor_network,
            background_processor_network=background_processor_network,
        ),
        collate_fn=partial(torch.cat, dim=0),
        chunk_size=chunk_size,
    )

    # obtain the processed points through the batchified processing pipeline
    processed_points = batchified_point_processing(network_input)

    # noinspection PyUnresolvedReferences
    return ProcessedPointsOnRays(
        processed_points.reshape(num_rays, num_samples_per_ray, -1),
        sampled_points.depths,
    )


def direct_feature_grid_processor(
    sampled_points: SampledPointsOnRays,
    _: Rays,
    feature_grid: FeatureGrid,
    background_mlp: Optional[Network] = None,
    chunk_size: int = 1024 * 64,
) -> ProcessedPointsOnRays:
    """ Denotes the Feature Grid as an RGBA volume directly"""
    num_rays, num_samples_per_ray, num_coords = sampled_points.points.shape
    flat_sampled_points = sampled_points.points.reshape(-1, num_coords)
    # we use dummy zero vectors for directions
    flat_sampled_directions = torch.zeros_like(
        flat_sampled_points, device=flat_sampled_points.device
    )
    flat_sampled_input = torch.cat(
        [flat_sampled_points, flat_sampled_directions], dim=-1
    )

    # create a batchified processing pipeline
    batchified_point_processing = batchify(
        partial(
            _rgba_extraction,
            feature_grid=feature_grid,
            background_mlp=background_mlp,
        ),
        partial(torch.cat, dim=0),
        chunk_size=chunk_size,
    )

    # obtain the processed points through the batchified processing pipeline
    processed_points = batchified_point_processing(flat_sampled_input)

    # noinspection PyUnresolvedReferences
    return ProcessedPointsOnRays(
        processed_points.reshape(num_rays, num_samples_per_ray, -1),
        sampled_points.depths,
    )


def direct_feature_grid_processor_with_sh(
    sampled_points: SampledPointsOnRays,
    rays: Rays,
    feature_grid: FeatureGrid,
    background_feature_grid: Optional[MultiSphereFeatureGrid] = None,
    render_diffuse: bool = False,
) -> ProcessedPointsOnRays:
    """ Denotes the Feature Grid as a volume containing the SH coefficients and the density """
    num_rays, num_samples_per_ray, num_coords = sampled_points.points.shape
    viewdirs = rays.directions / rays.directions.norm(dim=-1, keepdim=True)

    # either divide the points into inside and outside points if a background_feature_grid is available
    # or just use all points inside
    if background_feature_grid is not None:
        bg_num_samples_per_ray = background_feature_grid.features.shape[-2]
        inside_points, outside_points = (
            sampled_points.points[:, :-bg_num_samples_per_ray, :],
            sampled_points.points[:, -bg_num_samples_per_ray:, :],
        )
        flat_sampled_inside_points = inside_points.reshape(-1, num_coords)

        # handle the view-directions as well:
        viewdirs_tiled = (
            viewdirs[:, None, :]
            .repeat(1, num_samples_per_ray - bg_num_samples_per_ray, 1)
            .reshape(-1, num_coords)
        )
    else:
        flat_sampled_inside_points = sampled_points.points.reshape(-1, num_coords)
        outside_points = None

        viewdirs_tiled = (
            viewdirs[:, None, :]
            .repeat(1, num_samples_per_ray, 1)
            .reshape(-1, num_coords)
        )

    # process the points and directions to obtain the colour and the density:
    grid_interpolated_features = feature_grid(flat_sampled_inside_points)

    sh_coeffs, density = (
        grid_interpolated_features[..., :-1],
        grid_interpolated_features[..., -1:],
    )

    # evaluate the sh_coeffs at the view_dirs:
    if render_diffuse:
        # if rendering the diffuse variant, then we only use the degree 0 features
        sh_coeffs = sh_coeffs.reshape(sh_coeffs.shape[0], NUM_COLOUR_CHANNELS, -1)
        sh_coeffs = sh_coeffs[..., :1]
        sh_degree = 0
    else:
        sh_coeffs = sh_coeffs.reshape(sh_coeffs.shape[0], NUM_COLOUR_CHANNELS, -1)
        sh_degree = int(np.sqrt(sh_coeffs.shape[-1])) - 1
    colours = eval_sh(
        deg=sh_degree,
        sh=sh_coeffs,
        dirs=viewdirs_tiled,
    )
    processed_points = torch.cat([colours, density], dim=-1)
    processed_points = processed_points.reshape(num_rays, -1, NUM_RGBA_CHANNELS)

    if background_feature_grid is not None:
        background_processed_points = background_feature_grid(outside_points)
        background_processed_points = background_processed_points.reshape(
            num_rays, -1, NUM_RGBA_CHANNELS
        )
        processed_points = torch.cat(
            [processed_points, background_processed_points], dim=1
        )

    # noinspection PyUnresolvedReferences
    return ProcessedPointsOnRays(
        processed_points,
        sampled_points.depths,
    )


def render_feature_grid(
    rays: Rays,
    num_samples: int,
    feature_grid: FeatureGrid,
    scene_bounds: SceneBounds,
    point_processor_network: Optional[Network] = None,
    secondary_point_processor_network: Optional[Network] = None,
    background_processor_network: Optional[Network] = None,
    background_feature_grid: Optional[MultiSphereFeatureGrid] = None,
    num_samples_fine: Optional[int] = None,
    chunk_size: int = 1024 * 64,
    density_noise_std: float = 0.0,
    raw2alpha: Callable[[Tensor, Tensor], Tensor] = raw2alpha_base,
    colour_producer: Callable[[Tensor], Tensor] = torch.sigmoid,
    perturb_sampled_points: bool = True,
    use_dists_in_rendering: bool = True,
    hybrid_mode_only_rgba: bool = False,
    use_sh_based_rendering: bool = False,
    render_diffuse: bool = False,
    optimized_sampling_mode: bool = False,
    white_bkgd: bool = False,
) -> Union[RenderOut, Tuple[RenderOut, RenderOut]]:
    """
    Renders a 2D view from a given volumetric feature grid and point processor network
    Args:
        rays: Rays to be rendered using the volumetric feature grid
        num_samples: number of points to be sampled per ray
        feature_grid: regular feature grid. shape: [F x D x H x W], Object of FeatureGrid
        point_processor_network: point processor network
        secondary_point_processor_network: second point processor network used for hierarchical generation
        background_processor_network: a (potential) third point processor network for modelling the scene background
        background_feature_grid: a (potential) model for the background using a discrete feature grid
                                 in the form of multi-sphere images
        num_samples_fine: number of fine samples used for rendering
        scene_bounds: scene_bounds of the scene used for scaling the feature grid
        chunk_size: size of chunks created while processing points
        density_noise_std: whether to add small noise to the predicted density values
        raw2alpha: activation function applied to the predicted raw densities
        colour_producer: activation function applied to the predicted raw colour values
        perturb_sampled_points: whether to perturb the sampled points along rays during rendering
        use_dists_in_rendering: whether to use the delta-dists in rendering
        hybrid_mode_only_rgba: whether to only render the RGBA part of a hybrid FG+MLP model
        use_sh_based_rendering: whether to use Spherical Harmonics based rendering for view-dep-fx
        render_diffuse: whether to render degree 0 (diffuse) model
        optimized_sampling_mode: whether to sample ray-intervals using Ray-Voxel intersection
                                 used mainly for good rendering. Messes up training however :(
        white_bkgd: whether to use a white background in rendering (for un-accumulated rays)
    Returns: rendered 2D view. (Rendered Out) either returns single rendered out or two for primary and secondary
    """
    assert not (
        background_feature_grid is not None and background_processor_network is not None
    ), f"using both MLP and MSFeatureGrid to model the background :("

    # set up the rendering mechanism using the implicit render_interface
    if point_processor_network is not None and not hybrid_mode_only_rgba:
        point_processor = partial(
            implicit_feature_grid_point_processor,
            feature_grid=feature_grid,
            processor_network=point_processor_network,
            background_processor_network=background_processor_network,
            chunk_size=chunk_size,
        )
    else:
        if use_sh_based_rendering:
            point_processor = partial(
                direct_feature_grid_processor_with_sh,
                feature_grid=feature_grid,
                background_feature_grid=background_feature_grid,
                render_diffuse=render_diffuse,
            )
        else:
            point_processor = partial(
                direct_feature_grid_processor,
                feature_grid=feature_grid,
                background_mlp=background_processor_network,
                chunk_size=chunk_size,
            )
    if background_processor_network is not None:
        sampler_function = partial(
            sample_uniform_points_on_regular_feature_grid_with_background,
            aabb=feature_grid.aabb,
            perturb=perturb_sampled_points,
        )
    elif background_feature_grid is not None:
        sampler_function = partial(
            sample_uniform_points_with_inverted_sphere_background,
            num_bg_samples=background_feature_grid.features.shape[-2],
            aabb=feature_grid.aabb if optimized_sampling_mode else None,
            perturb=perturb_sampled_points,
        )
    else:
        if optimized_sampling_mode:
            sampler_function = partial(
                sample_uniform_points_on_regular_feature_grid,
                aabb=feature_grid.aabb,
                perturb=perturb_sampled_points,
            )
        else:
            sampler_function = partial(
                sample_uniform_points_on_rays,
                perturb=perturb_sampled_points,
            )
    if background_feature_grid is not None:
        accumulator_function = partial(
            accumulate_processed_points_on_rays_with_msfg_bg,
            num_bg_samples=background_feature_grid.features.shape[-2],
            density_noise_std=density_noise_std,
            raw2_alpha=raw2alpha,
            colour_producer=colour_producer,
            use_dists=use_dists_in_rendering,
        )
    else:
        accumulator_function = partial(
            accumulate_processed_points_on_rays,
            density_noise_std=density_noise_std,
            raw2_alpha=raw2alpha,
            use_dists=use_dists_in_rendering,
            colour_producer=colour_producer,
            use_infinite_far_dist=background_processor_network is not None
            or background_feature_grid is not None,
            white_bkgd=white_bkgd,
        )

    if secondary_point_processor_network is None:
        return render(
            rays,
            scene_bounds,
            num_samples,
            sampler_function,
            point_processor,
            accumulator_function,
        )
    else:
        assert (
            point_processor_network is not None and not hybrid_mode_only_rgba
        ), f"Point processor cannot be None when secondary point processor network is provided"
        assert (
            num_samples_fine is not None
        ), f"Provided a secondary point processor network, but `num_fine_samples` are None :("

        # we need to perform hierarchical Sampling
        secondary_point_processor = partial(
            implicit_feature_grid_point_processor,
            feature_grid=feature_grid,
            processor_network=secondary_point_processor_network,
            background_processor_network=background_processor_network,
            chunk_size=chunk_size,
        )

        # obtain coarse_rendered_output:
        coarse_rendered_output = render(
            rays,
            scene_bounds,
            num_samples,
            sampler_fn=sampler_function,
            point_processor_fn=point_processor,
            accumulator_fn=accumulator_function,
        )

        # then use the coarse_rendered_output to sample more points for fine
        # network
        point_weights = coarse_rendered_output.extra[EXTRA_POINT_WEIGHTS].detach()
        point_depths = coarse_rendered_output.extra[EXTRA_POINT_DEPTHS].detach()
        # note the `detach` above, so no backpropagation should happen from here

        bins = 0.5 * (point_depths[..., 1:] + point_depths[..., :-1])
        fine_stratified_depths = sample_cdf_weighted_points_on_rays(
            bins,
            point_weights[..., 1:-1],  # first and last points are ignored
            num_samples=num_samples_fine,
        )

        # use all the points (uniform and hierarchically sampled)
        all_depths, _ = torch.sort(
            torch.cat([point_depths, fine_stratified_depths], dim=-1), dim=-1
        )
        all_points = (
            rays.origins[..., None, :]
            + rays.directions[..., None, :] * all_depths[..., :, None]
        )
        hierarchically_sampled_points = SampledPointsOnRays(
            points=all_points, depths=all_depths
        )

        # Obtain the fine rendered output:
        processed_points = secondary_point_processor(
            hierarchically_sampled_points, rays
        )
        fine_rendered_output = accumulator_function(processed_points, rays)

        # return the coarse and fine output both:
        return coarse_rendered_output, fine_rendered_output
