from functools import partial
from typing import Callable

import torch
from torch import Tensor
from torch.nn.functional import softplus

from thre3d_atom.networks.network_interface import Network
from thre3d_atom.rendering.volumetric.render_interface import (
    Rays,
    SampledPointsOnRays,
    ProcessedPointsOnRays,
    RenderOut,
)
from thre3d_atom.utils.constants import (
    ZERO_PLUS,
    INFINITY,
    EXTRA_ACCUMULATED_WEIGHTS,
    EXTRA_DEPTHS,
    EXTRA_POINT_WEIGHTS,
    EXTRA_POINT_DEPTHS,
    EXTRA_POINT_DENSITIES,
    EXTRA_MEDIAN_DISPARITY,
    EXTRA_RAW_DENSITIES,
    EXTRA_DISTS,
    EXTRA_MEAN_DISPARITY,
)
from thre3d_atom.utils.imaging_utils import CameraIntrinsics, CameraPose
from thre3d_atom.utils.misc import batchify


def shifted_sigmoid(x: Tensor, epsilon: float = 0.001) -> Tensor:
    return (1 + (2 * epsilon)) / (1 + torch.exp(-x)) - epsilon


def _shifted_softplus(x: Tensor) -> Tensor:
    return torch.nn.functional.softplus(x - 1)


def raw2alpha_base(
    raw: Tensor,
    dists: Tensor,
    act_fn: Callable[[Tensor], Tensor] = _shifted_softplus,
    raw_scale: float = 1.0,
) -> Tensor:
    """Function for computing density from model prediction.
    This value is strictly between [0, 1]."""
    return 1.0 - torch.exp(-act_fn(raw * raw_scale) * dists)


def raw2alpha_new(raw: Tensor, dists: Tensor) -> Tensor:
    """Function for computing density from model prediction.
    This value is strictly between [0, 1]."""
    x = softplus(raw * dists - 1, beta=2)
    return (2 * x) / (1 + (x ** 2))


def cast_rays(
    camera_intrinsics: CameraIntrinsics,
    pose: CameraPose,
    device: torch.device = torch.device("cpu"),
) -> Rays:
    # convert the camera pose into tensors if they are numpy arrays
    if not (isinstance(pose.rotation, Tensor) and isinstance(pose.translation, Tensor)):
        rot = torch.from_numpy(pose.rotation)
        trans = torch.from_numpy(pose.translation)
        pose = CameraPose(rot, trans)
    if not (pose.rotation.device == device and pose.translation.device == device):
        pose = CameraPose(pose.rotation.to(device), pose.translation.to(device))

    # cast the rays for the given CameraPose
    height, width, focal = camera_intrinsics
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(0.5, height - 0.5, height, dtype=torch.float32, device=device),
        torch.linspace(0.5, width - 0.5, width, dtype=torch.float32, device=device),
    )

    # note the flip on the y_coords below, because of the image-indexing
    x_coords, y_coords = x_coords, torch.flip(y_coords, dims=(0,))

    dirs = torch.stack(
        [
            (x_coords - width * 0.5) / focal,
            (y_coords - height * 0.5) / focal,
            -torch.ones_like(x_coords, device=device),
        ],
        -1,
    )

    rays_d = (pose.rotation @ dirs[..., None])[..., 0]
    rays_o = torch.broadcast_to(pose.translation.squeeze(), rays_d.shape)
    flat_rays_o, flat_rays_d = rays_o.reshape(-1, rays_o.shape[-1]), rays_d.reshape(
        -1, rays_d.shape[-1]
    )
    return Rays(flat_rays_o, flat_rays_d)


def simple_process_points_on_rays_with_network(
    sampled_points: SampledPointsOnRays,
    _: Rays,
    network: Network,
    chunk_size: int,
) -> ProcessedPointsOnRays:
    # Process the sampled points with the NeuralNetwork in small chunks to avoid memory Overflow
    tensor_collate_function = partial(torch.cat, dim=0)
    batchified_network_function = batchify(
        network, tensor_collate_function, chunk_size=chunk_size, verbose=False
    )

    num_rays, num_points_per_ray, coord_dims = sampled_points.points.shape
    network_input = sampled_points.points.reshape(-1, coord_dims)
    processed_points = batchified_network_function(network_input)

    # noinspection PyUnresolvedReferences
    return ProcessedPointsOnRays(
        processed_points.reshape(num_rays, num_points_per_ray, -1),
        sampled_points.depths,
    )


def process_points_on_rays_with_network(
    sampled_points: SampledPointsOnRays,
    rays: Rays,
    network: Network,
    chunk_size: int,
    use_viewdirs: bool = True,
) -> ProcessedPointsOnRays:
    if use_viewdirs:
        num_points = sampled_points.points.shape[1]
        viewdirs = rays.directions / rays.directions.norm(dim=-1, keepdim=True)
        viewdirs_tiled = viewdirs[:, None, :].repeat(1, num_points, 1)
        network_input = torch.cat([sampled_points.points, viewdirs_tiled], dim=-1)
    else:
        viewdirs = torch.zeros_like(
            sampled_points.points, device=sampled_points.points.device
        )
        network_input = torch.cat([sampled_points.points, viewdirs], dim=-1)

    # Process the sampled points with the NeuralNetwork in small chunks to avoid memory Overflow
    tensor_collate_function = partial(torch.cat, dim=0)
    batchified_network_function = batchify(
        network, tensor_collate_function, chunk_size=chunk_size, verbose=False
    )
    processed_points = batchified_network_function(
        network_input.view(-1, network_input.shape[-1])
    )

    # noinspection PyUnresolvedReferences
    return ProcessedPointsOnRays(
        processed_points.view(*network_input.shape[:-1], -1), sampled_points.depths
    )


def accumulate_processed_points_on_rays(
    processed_points: ProcessedPointsOnRays,
    rays: Rays,
    density_noise_std: float = 0.0,
    raw2_alpha: Callable[[Tensor, Tensor], Tensor] = raw2alpha_base,
    colour_producer: Callable[[Tensor], Tensor] = torch.sigmoid,
    use_dists: bool = True,
    use_infinite_far_dist: bool = True,
    white_bkgd: bool = False,
    extra_debug_info: bool = False,
) -> RenderOut:
    raw_colour, raw_density = (
        processed_points.points[..., :-1],
        processed_points.points[..., -1],
    )
    # compute point distances for ray-time integration
    dists = processed_points.depths[..., 1:] - processed_points.depths[..., :-1]

    if use_infinite_far_dist:
        last_dist = torch.full((*dists.shape[:-1], 1), INFINITY, device=dists.device)
    else:
        last_dist = torch.zeros((*dists.shape[:-1], 1), device=dists.device)
    dists = torch.cat([dists, last_dist], dim=-1)  # [N_rays, N_samples]
    dists = dists * rays.directions[..., None, :].norm(dim=-1)

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    density_noise = (
        torch.randn(raw_density.shape, device=raw_density.device) * density_noise_std
    )
    if use_dists:
        alpha = raw2_alpha(raw_density + density_noise, dists)  # [N, num_samples]
    else:
        alpha = raw2_alpha(raw_density + density_noise, torch.ones_like(dists))

    # compute the colour weights for accumulation along the ray
    ones = torch.ones((alpha.shape[0], 1), device=alpha.device)
    one_minus_alpha = (1.0 - alpha) + ZERO_PLUS
    one_minus_alpha_with_one = torch.cat([ones, one_minus_alpha], -1)
    weights = alpha * torch.cumprod(one_minus_alpha_with_one, -1)[:, :-1]

    # accumulate the predicted colour values of the points
    # using the computed density weights
    colour = colour_producer(raw_colour)
    colour_map = torch.sum(
        weights[..., None] * colour, dim=-2
    )  # [N, NUM_COLOUR_CHANNELS]

    # Estimated depth map is average of depths weighted by density weights.
    actual_depths = processed_points.depths * rays.directions[..., None, :].norm(dim=-1)

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    if white_bkgd:
        # invert the background of the rendered colour_maps to basically have
        # white colour instead of black when nothing is accumulated on the ray.
        colour_map = colour_map + (1 - acc_map[..., None])

    # compute NerFIE style disparity_map
    high_weights = (weights >= 0.5).float()
    median_indices = high_weights.argmax(dim=-1, keepdim=True)
    median_depth_values = actual_depths.gather(-1, median_indices)
    median_depth_map = torch.where(
        acc_map == 0,
        torch.full(
            (median_depth_values.shape[0],), INFINITY, device=median_depth_values.device
        ),  # missed rays (infinite rays)
        median_depth_values.reshape(-1),
    )
    median_disparity_map = 1.0 / torch.maximum(
        torch.full(acc_map.shape, ZERO_PLUS, device=median_depth_map.device),
        median_depth_map / acc_map,
    )

    # compute normal NeRF style disparity_map
    # mean_depth_map = torch.where(
    #     acc_map == 0,
    #     torch.full(
    #         (actual_depths.shape[0],), INFINITY, device=actual_depths.device
    #     ),  # missed rays (infinite rays)
    #     (actual_depths * weights).sum(dim=-1),
    # )
    mean_depth_map = (actual_depths * weights).sum(dim=-1)
    mean_disparity_map = 1.0 / torch.maximum(
        torch.full(acc_map.shape, ZERO_PLUS, device=mean_depth_map.device),
        mean_depth_map / acc_map,
    )

    extra_dict = {
        EXTRA_MEAN_DISPARITY: mean_disparity_map,
        EXTRA_MEDIAN_DISPARITY: median_disparity_map,
        EXTRA_ACCUMULATED_WEIGHTS: acc_map,
        EXTRA_DEPTHS: median_depth_map,
    }

    if extra_debug_info:
        # adds very big buffers per ray (or ray-chunk). Please be careful while using
        extra_dict.update(
            {
                EXTRA_RAW_DENSITIES: raw_density,
                EXTRA_DISTS: dists,
                EXTRA_POINT_DENSITIES: alpha,
                EXTRA_POINT_WEIGHTS: weights,
                EXTRA_POINT_DEPTHS: processed_points.depths,
            }
        )

    return RenderOut(
        colour=colour_map,
        disparity=mean_depth_map,
        extra=extra_dict,
    )


def accumulate_processed_points_on_rays_with_msfg_bg(
    processed_points: ProcessedPointsOnRays,
    rays: Rays,
    num_bg_samples: int,
    density_noise_std: float = 0.0,
    raw2_alpha: Callable[[Tensor, Tensor], Tensor] = raw2alpha_base,
    colour_producer: Callable[[Tensor], Tensor] = torch.sigmoid,
    use_dists: bool = True,
) -> RenderOut:
    # accumulate the processed points inside and outside the sphere separately
    inside_points, outside_points = (
        processed_points.points[:, :-num_bg_samples, :],
        processed_points.points[:, -num_bg_samples:, :],
    )
    inside_depths, outside_depths = (
        processed_points.depths[:, :-num_bg_samples],
        processed_points.depths[:, -num_bg_samples:],
    )

    # render the inside points normally:
    inside_render = accumulate_processed_points_on_rays(
        ProcessedPointsOnRays(inside_points, inside_depths),
        rays,
        density_noise_std,
        raw2_alpha,
        colour_producer,
        use_dists,
        use_infinite_far_dist=False,
    )

    # render the outside points in a discrete manner. We use basic over alpha-compositing
    # No volumes involved.
    outside_colours, outside_alphas = (
        outside_points[..., :-1],
        outside_points[..., -1],
    )

    ones = torch.ones((outside_alphas.shape[0], 1), device=outside_alphas.device)
    outside_weights = (
        outside_alphas
        * torch.cumprod(torch.cat([ones, 1.0 - outside_alphas + ZERO_PLUS], -1), -1)[
            :, :-1
        ]
    )

    # accumulate the predicted colour values of the points
    # using the computed density weights
    # [N, NUM_COLOUR_CHANNELS]
    outside_acc_map = torch.sum(outside_weights, dim=-1)
    outside_colour_map = torch.sum(outside_weights[..., None] * outside_colours, dim=-2)

    # composite the inside and outside together:
    inside_colour_map, inside_acc_map = (
        inside_render.colour,
        inside_render.extra[EXTRA_ACCUMULATED_WEIGHTS],
    )

    colour_map = inside_colour_map + (
        (1 - inside_acc_map[..., None]) * outside_colour_map
    )
    acc_map = inside_acc_map + ((1 - inside_acc_map) * outside_acc_map)

    inside_render.extra.update({EXTRA_ACCUMULATED_WEIGHTS: acc_map})
    return RenderOut(colour_map, inside_render.disparity, inside_render.extra)
    # return inside_render
