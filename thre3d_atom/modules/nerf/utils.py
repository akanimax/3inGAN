import torch
from torch import Tensor

from thre3d_atom.rendering.volumetric.sample import sample_cdf_weighted_points_on_rays
from thre3d_atom.rendering.volumetric.render_interface import (
    Rays,
    SampledPointsOnRays,
)
from thre3d_atom.utils.imaging_utils import SceneBounds, CameraIntrinsics


def ndcize_rays(rays: Rays, camera_intrinsics: CameraIntrinsics) -> Rays:
    """Normalized device coordinate rays.
    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    """
    # unpack everything
    height, width, focal = camera_intrinsics
    near = 1.0
    rays_o, rays_d = rays

    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1.0 / (width / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (height / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = (
        -1.0
        / (width / (2.0 * focal))
        * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    )
    d1 = (
        -1.0
        / (height / (2.0 * focal))
        * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    )
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return Rays(rays_o, rays_d)


def hierarchical_sampler(
    rays: Rays,
    _: SceneBounds,
    num_samples: int,
    coarse_point_weights: Tensor,
    coarse_point_depths: Tensor,
) -> SampledPointsOnRays:
    """custom sampler function for the hierarchical case. We use the point_weights and
    the point_depths which are output by the coarse network to sample better
    points for the fine network. Please note that the sampling is not differentiable
    and the two networks coarse and fine are two different networks.
    Note that we don't need scene_bounds for hierarchical sampling because we use the
    weighted depths of the coarse points"""

    bins = 0.5 * (coarse_point_depths[..., 1:] + coarse_point_depths[..., :-1])
    fine_stratified_depths = sample_cdf_weighted_points_on_rays(
        bins,
        coarse_point_weights[..., 1:-1],  # first and last points are ignored
        num_samples=num_samples,
    )

    # use all the points (uniform and hierarchically sampled)
    all_depths, _ = torch.sort(
        torch.cat([coarse_point_depths, fine_stratified_depths], dim=-1), dim=-1
    )
    all_points = (
        rays.origins[..., None, :]
        + rays.directions[..., None, :] * all_depths[..., :, None]
    )
    return SampledPointsOnRays(points=all_points, depths=all_depths)
