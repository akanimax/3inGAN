from typing import Sequence, Tuple, Union, List

import numpy as np
import torch
from torch import Tensor

from thre3d_atom.rendering.volumetric.render_interface import RenderOut, Rays
from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS
from thre3d_atom.utils.imaging_utils import CameraIntrinsics


def collate_rendered_output(rendered_chunks: Sequence[RenderOut]) -> RenderOut:
    """Defines how a sequence of rendered_chunks can be
    collated into a render_out"""
    # collect all the rendered_chunks into lists
    colour, disparity, extra = [], [], {}
    for rendered_chunk in rendered_chunks:
        colour.append(rendered_chunk.colour)
        disparity.append(rendered_chunk.disparity)
        for key, value in rendered_chunk.extra.items():
            extra[key] = extra.get(key, []) + [value]

    # combine all the tensor information
    colour = torch.cat(colour, dim=0)
    disparity = torch.cat(disparity, dim=0)
    extra = {key: torch.cat(extra[key], dim=0) for key in extra}

    # return the collated rendered_output
    return RenderOut(colour=colour, disparity=disparity, extra=extra)


def reshape_rendered_output(
    rendered_output: RenderOut, camera_intrinsics: CameraIntrinsics
) -> RenderOut:
    new_shape = (camera_intrinsics.height, camera_intrinsics.width, -1)
    return RenderOut(
        colour=rendered_output.colour.reshape(*new_shape),
        disparity=rendered_output.disparity.reshape(*new_shape),
        extra={
            key: value.reshape(*new_shape)
            for key, value in rendered_output.extra.items()
        },
    )


def collate_coarse_and_fine_rendered_output(
    rendered_chunks: Sequence[Tuple[RenderOut, RenderOut]]
) -> Tuple[RenderOut, RenderOut]:
    """collate function for collating tuples of coarse and fine rendered_outputs"""
    coarse_rendered_chunks, fine_rendered_chunks = list(zip(*rendered_chunks))
    coarse_rendered_output = collate_rendered_output(coarse_rendered_chunks)
    fine_rendered_output = (
        collate_rendered_output(fine_rendered_chunks)
        if all(
            fine_rendered_chunk is not None
            for fine_rendered_chunk in fine_rendered_chunks
        )
        else None
    )
    return coarse_rendered_output, fine_rendered_output


def shuffle_rays_and_pixels_synchronously(
    rays: Rays, pixels: Tensor
) -> Tuple[Rays, Tensor]:
    permutation = torch.randperm(pixels.shape[0])
    rays_origins, rays_directions = rays.origins, rays.directions
    shuffled_rays_origins = rays_origins[permutation, :]
    shuffled_rays_directions = rays_directions[permutation, :]
    shuffled_pixels = pixels[permutation, :]
    return Rays(shuffled_rays_origins, shuffled_rays_directions), shuffled_pixels


def select_shuffled_rays_and_pixels_synchronously(
    rays: Union[Rays, Tensor],
    pixels: Tensor,
    select_batch: int,
) -> Tuple[Rays, Tensor]:
    permutation = torch.randperm(pixels.shape[0], device=pixels.device)
    selected_subset = permutation[:select_batch]
    selected_pixels = pixels[selected_subset, :]
    rays_tensor = (
        torch.cat([rays.origins, rays.directions], dim=-1)
        if isinstance(rays, Rays)
        else rays
    )
    selected_rays_tensor = rays_tensor[selected_subset, :]
    return (
        Rays(
            selected_rays_tensor[:, :NUM_COORD_DIMENSIONS],
            selected_rays_tensor[:, NUM_COORD_DIMENSIONS:],
        ),
        selected_pixels,
    )


def collate_rays(rays_list: Sequence[Rays]) -> Rays:
    """utility method for collating rays"""
    return Rays(
        origins=torch.cat([rays.origins for rays in rays_list], dim=0),
        directions=torch.cat([rays.directions for rays in rays_list], dim=0),
    )


def flatten_rays(rays: Rays) -> Rays:
    return Rays(
        origins=rays.origins.reshape(-1, NUM_COORD_DIMENSIONS),
        directions=rays.directions.reshape(-1, NUM_COORD_DIMENSIONS),
    )


def reshape_and_rebuild_flat_rays(
    rays_list: Sequence[Rays], camera_intrinsics: CameraIntrinsics
) -> Rays:
    """utility method for reshaping and rebuilding rays"""
    height, width = camera_intrinsics.height, camera_intrinsics.width
    return Rays(
        origins=torch.stack(
            [rays.origins.reshape(height, width, -1) for rays in rays_list], dim=0
        ),
        directions=torch.stack(
            [rays.directions.reshape(height, width, -1) for rays in rays_list],
            dim=0,
        ),
    )


def compute_grid_sizes(
    output_resolution: Tuple[int, int, int], num_stages: int, scale_factor: float
) -> List[Tuple[int, int, int]]:
    x, y, z = output_resolution
    grid_sizes = [(x, y, z)]
    for _ in range(num_stages - 1):
        x = int(np.ceil((1 / scale_factor) * x))
        y = int(np.ceil((1 / scale_factor) * y))
        z = int(np.ceil((1 / scale_factor) * z))
        grid_sizes.insert(0, (x, y, z))
    return grid_sizes
