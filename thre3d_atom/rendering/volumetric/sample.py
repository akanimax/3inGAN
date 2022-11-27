from typing import Union, Tuple, Optional

import torch
from torch import Tensor

from thre3d_atom.rendering.volumetric.render_interface import (
    Rays,
    SampledPointsOnRays,
)
from thre3d_atom.utils.types import AxisAlignedBoundingBox
from thre3d_atom.utils.constants import ZERO_PLUS, INFINITY
from thre3d_atom.utils.imaging_utils import SceneBounds


def sample_uniform_points_on_rays(
    rays: Rays,
    bounds: Union[SceneBounds, Tensor],
    num_samples: int,
    perturb: bool = True,
    linear_disparity_sampling: bool = False,
) -> SampledPointsOnRays:
    rays_o, rays_d = rays.origins, rays.directions
    num_rays = rays_o.shape[0]

    if isinstance(bounds, SceneBounds):
        near, far = bounds.near, bounds.far
    else:
        near, far = bounds[:, :1], bounds[:, 1:]

    # ray sampling logic
    t_vals = torch.linspace(0.0, 1.0, num_samples, device=rays_o.device)[None, ...]
    if linear_disparity_sampling:
        z_vals = 1.0 / (1.0 / (near + ZERO_PLUS) * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        z_vals = near * (1.0 - t_vals) + far * t_vals

    if z_vals.shape[0] != num_rays:
        z_vals = z_vals.repeat([num_rays, 1])

    # Perturb sampled points along each ray.
    if perturb:
        # get intervals between samples
        mid_points = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper_points = torch.cat([mid_points, z_vals[..., -1:]], -1)
        lower_points = torch.cat([z_vals[..., :1], mid_points], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(*z_vals.shape, device=rays_o.device)
        z_vals = lower_points + (upper_points - lower_points) * t_rand

    # Points in space to evaluate model at.
    sampled_points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return SampledPointsOnRays(sampled_points, z_vals)


# TODO: refactor this function and test
def sample_cdf_weighted_points_on_rays(
    bins: Tensor, weights: Tensor, num_samples: int, det: bool = False
) -> Tensor:
    # Get pdf
    weights = weights + ZERO_PLUS  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0.0, 1.0, steps=num_samples).to(cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [num_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples]).to(cdf.device)

    # Invert CDF
    u = u.contiguous()
    indices = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(indices - 1), indices - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(indices), indices)
    indices_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [indices_g.shape[0], indices_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, indices_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, indices_g)

    denominator = cdf_g[..., 1] - cdf_g[..., 0]
    denominator = torch.where(
        denominator < 1e-5, torch.ones_like(denominator), denominator
    )
    t = (u - cdf_g[..., 0]) / denominator
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def _ray_aabb_intersection(
    rays: Rays, bounds: Union[SceneBounds, Tensor], aabb: AxisAlignedBoundingBox
) -> Tuple[Tensor, Tensor]:
    # Please refer this blog for the implementation ->
    # https://www.scratchapixel.com/lessons/3d-basic-rendering/->
    # ->minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
    # ======================================================================================
    # compute the near and far bounds for each ray based on it's intersection with the aabb
    # ======================================================================================
    origins, directions = rays
    num_rays = origins.shape[0]
    device = rays.origins.device
    orig_ray_bounds = (
        torch.cat(
            [
                torch.full(
                    (num_rays, 1), bounds.near, dtype=torch.float32, device=device
                ),
                torch.full(
                    (num_rays, 1), bounds.far, dtype=torch.float32, device=device
                ),
            ],
            dim=-1,
        )
        if isinstance(bounds, SceneBounds)
        else bounds
    )
    intersecting = torch.full((num_rays, 1), True, dtype=torch.bool, device=device)
    non_intersecting = torch.full((num_rays, 1), False, dtype=torch.bool, device=device)

    # compute intersections with the X-planes:
    x_min = (aabb.x_range[0] - origins[:, 0]) / (directions[:, 0] + ZERO_PLUS)
    x_max = (aabb.x_range[1] - origins[:, 0]) / (directions[:, 0] + ZERO_PLUS)
    x_ray_bounds = torch.stack([x_min, x_max], dim=-1)
    x_ray_bounds = torch.where(
        x_ray_bounds[:, :1] > x_ray_bounds[:, 1:], x_ray_bounds[:, [1, 0]], x_ray_bounds
    )
    final_ray_bounds = x_ray_bounds

    # compute intersections with the Y-planes:
    y_min = (aabb.y_range[0] - origins[:, 1]) / (directions[:, 1] + ZERO_PLUS)
    y_max = (aabb.y_range[1] - origins[:, 1]) / (directions[:, 1] + ZERO_PLUS)
    y_ray_bounds = torch.stack([y_min, y_max], dim=-1)

    y_ray_bounds = torch.where(
        y_ray_bounds[:, :1] > y_ray_bounds[:, 1:], y_ray_bounds[:, [1, 0]], y_ray_bounds
    )

    intersecting = torch.where(
        torch.logical_or(
            final_ray_bounds[:, :1] > y_ray_bounds[:, 1:],
            y_ray_bounds[:, :1] > final_ray_bounds[:, 1:],
        ),
        non_intersecting,
        intersecting,
    )

    final_ray_bounds[:, 0] = torch.where(
        y_ray_bounds[:, 0] > final_ray_bounds[:, 0],
        y_ray_bounds[:, 0],
        final_ray_bounds[:, 0],
    )

    final_ray_bounds[:, 1] = torch.where(
        y_ray_bounds[:, 1] < final_ray_bounds[:, 1],
        y_ray_bounds[:, 1],
        final_ray_bounds[:, 1],
    )

    # compute intersections with the Z-planes:
    z_min = (aabb.z_range[0] - origins[:, 2]) / (directions[:, 2] + ZERO_PLUS)
    z_max = (aabb.z_range[1] - origins[:, 2]) / (directions[:, 2] + ZERO_PLUS)
    z_ray_bounds = torch.stack([z_min, z_max], dim=-1)
    z_ray_bounds = torch.where(
        z_ray_bounds[:, :1] > z_ray_bounds[:, 1:], z_ray_bounds[:, [1, 0]], z_ray_bounds
    )

    intersecting = torch.where(
        torch.logical_or(
            final_ray_bounds[:, :1] > z_ray_bounds[:, 1:],
            z_ray_bounds[:, :1] > final_ray_bounds[:, 1:],
        ),
        non_intersecting,
        intersecting,
    )

    final_ray_bounds[:, 0] = torch.where(
        z_ray_bounds[:, 0] > final_ray_bounds[:, 0],
        z_ray_bounds[:, 0],
        final_ray_bounds[:, 0],
    )

    final_ray_bounds[:, 1] = torch.where(
        z_ray_bounds[:, 1] < final_ray_bounds[:, 1],
        z_ray_bounds[:, 1],
        final_ray_bounds[:, 1],
    )

    # finally, revert the non_intersecting rays to the original scene_bounds:
    final_ray_bounds = torch.where(
        torch.logical_not(intersecting), orig_ray_bounds, final_ray_bounds
    )

    # We don't consider the intersections behind the camera
    final_ray_bounds = torch.clip(final_ray_bounds, min=0.0)

    # return the computed intersections (final_ray_bounds) and the boolean Tensor intersecting
    # denoting whether the ray intersected the aabb or not.
    return final_ray_bounds, intersecting


def sample_uniform_points_on_regular_feature_grid(
    rays: Rays,
    bounds: Union[SceneBounds, Tensor],
    num_samples: int,
    aabb: AxisAlignedBoundingBox,
    perturb: bool = True,
) -> SampledPointsOnRays:

    final_ray_bounds, _ = _ray_aabb_intersection(rays, bounds, aabb)

    # return uniform points sampled on the rays using the new ray_bounds:
    return sample_uniform_points_on_rays(
        rays,
        bounds=final_ray_bounds,
        num_samples=num_samples,
        perturb=perturb,
    )


def sample_uniform_points_on_regular_feature_grid_with_background(
    rays: Rays,
    bounds: SceneBounds,
    num_samples: int,
    aabb: AxisAlignedBoundingBox,
    perturb: bool = True,
) -> SampledPointsOnRays:
    # obtain num_samples / 2 points for rays intersecting the aabb
    final_ray_bounds, intersecting = _ray_aabb_intersection(rays, bounds, aabb)
    uniform_points_inside = sample_uniform_points_on_rays(
        rays,
        bounds=final_ray_bounds,
        num_samples=int(num_samples / 2),
        perturb=perturb,
        linear_disparity_sampling=False,
    )

    post_ray_bounds = torch.cat(
        [
            final_ray_bounds[:, 1:],
            torch.tensor(
                [[bounds.far]], device=final_ray_bounds.device, dtype=torch.float32
            ).repeat(final_ray_bounds.shape[0], 1),
        ],
        dim=-1,
    )

    # sample (num_samples / 2) points outside the box for the intersecting rays
    uniform_points_outside = sample_uniform_points_on_rays(
        rays,
        bounds=post_ray_bounds,
        num_samples=int(num_samples / 2),
        perturb=perturb,
        linear_disparity_sampling=False,
    )

    all_intersecting_ray_points_tensor = torch.cat(
        [uniform_points_inside.points, uniform_points_outside.points], dim=1
    )
    all_intersecting_ray_depths_tensor = torch.cat(
        [uniform_points_inside.depths, uniform_points_outside.depths], dim=1
    )

    # sample uniform points for all the rays that don't intersect the aabb:
    simple_uniform_sampled_points = sample_uniform_points_on_rays(
        rays,
        bounds,
        num_samples=num_samples,
        perturb=perturb,
        linear_disparity_sampling=False,
    )

    # combine all the points:
    all_sampled_points = torch.where(
        intersecting[:, None],
        all_intersecting_ray_points_tensor,
        simple_uniform_sampled_points.points,
    )
    all_sampled_depths = torch.where(
        intersecting,
        all_intersecting_ray_depths_tensor,
        simple_uniform_sampled_points.depths,
    )

    return SampledPointsOnRays(all_sampled_points, all_sampled_depths)


def _ray_sphere_intersection(rays: Rays, radius: Union[float, Tensor] = 1.0) -> Tensor:
    """
    Note that the radius can be a float value or a scalar torch.Tensor
    Assumptions made by this function:
    We assume that rays are always shot from inside the unit sphere.
        1. Which means that the rays will always intersect the sphere, so no checking is done
        2. Which also means that we only consider the +ve solution to the quadratic
           (since the -ve is in the opposite side of the camera)
        3. Note that the sphere can have arbitrary radius, but it is always centered at the origin.
    """
    a = (rays.directions * rays.directions).sum(dim=-1)
    b = 2 * ((rays.directions * rays.origins).sum(dim=-1))
    c = ((rays.origins * rays.origins).sum(dim=-1)) - (radius ** 2)
    poi = (-b + torch.sqrt((b ** 2) - (4 * a * c))) / (2 * a)
    return poi


def _batched_ray_sphere_intersection(rays: Rays, radii: Tensor) -> Tensor:
    """same as above, but obtains all the intersections with all radii in parallel"""
    num_rays, num_spheres = rays.origins.shape[0], radii.shape[0]
    tiled_rays_origins = rays.origins[:, None, :].tile(1, num_spheres, 1)
    tiled_rays_dirs = rays.directions[:, None, :].tile(1, num_spheres, 1)
    tiled_radii = radii[None, :].tile(num_rays, 1)

    a = (tiled_rays_dirs * tiled_rays_dirs).sum(dim=-1)
    b = 2 * ((tiled_rays_dirs * tiled_rays_origins).sum(dim=-1))
    c = ((tiled_rays_origins * tiled_rays_origins).sum(dim=-1)) - (tiled_radii ** 2)
    pois = (-b + torch.sqrt((b ** 2) - (4 * a * c))) / (2 * a)
    return pois


def sample_uniform_points_with_inverted_sphere_background(
    rays: Rays,
    bounds: SceneBounds,
    num_samples: int,
    num_bg_samples: int,
    aabb: Optional[AxisAlignedBoundingBox] = None,
    perturb: bool = True,
) -> SampledPointsOnRays:
    # Find the unit-sphere intersection points for all the rays
    unit_sphere_intersections = _ray_sphere_intersection(rays, 1.0)

    # sample the inside points
    device, num_rays = rays.origins.device, len(rays.origins)
    inside_bounds = torch.tensor(
        [bounds.near, bounds.far], dtype=torch.float32, device=device
    )[None, :].tile(num_rays, 1)
    inside_bounds[:, -1] = unit_sphere_intersections
    inside_points = (
        sample_uniform_points_on_regular_feature_grid(
            rays, inside_bounds, num_samples, aabb, perturb
        )
        if aabb is not None
        else sample_uniform_points_on_rays(rays, inside_bounds, num_samples, perturb)
    )

    # sample the outside points
    outside_spheres_inv_radii = torch.linspace(
        1.0 / INFINITY, 1.0, num_bg_samples, dtype=torch.float32, device=device
    )
    outside_spheres_radii = torch.flip(1.0 / outside_spheres_inv_radii, dims=(0,))

    all_intersections = _batched_ray_sphere_intersection(rays, outside_spheres_radii)
    outside_points = rays.origins[:, None, :] + (
        rays.directions[:, None, :] * all_intersections[:, :, None]
    )
    outside_depths = all_intersections

    return SampledPointsOnRays(
        points=torch.cat([inside_points.points, outside_points], dim=1),
        depths=torch.cat([inside_points.depths, outside_depths], dim=1),
    )
