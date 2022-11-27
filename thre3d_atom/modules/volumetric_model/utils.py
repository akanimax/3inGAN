import time
from typing import Optional, Union, Tuple, Callable

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from thre3d_atom.networks.network_interface import Network
from thre3d_atom.rendering.volumetric.implicit import (
    cast_rays,
    raw2alpha_base,
)
from thre3d_atom.rendering.volumetric.render_interface import (
    RenderOut,
)
from thre3d_atom.rendering.volumetric.utils import (
    reshape_rendered_output,
    collate_rendered_output,
)
from thre3d_atom.rendering.volumetric.voxels import (
    FeatureGrid,
    render_feature_grid,
    MultiSphereFeatureGrid,
)
from thre3d_atom.utils.imaging_utils import SceneBounds, CameraIntrinsics, CameraPose
from thre3d_atom.utils.logging import log


def render_image_in_chunks(
    cam_intrinsics: CameraIntrinsics,
    camera_pose: CameraPose,
    num_rays_chunk: int,
    num_samples_per_ray: int,
    feature_grid: FeatureGrid,
    scene_bounds: SceneBounds,
    num_points_chunk: int = 65536,
    processor_mlp: Optional[Network] = None,
    secondary_processor_mlp: Optional[Network] = None,
    background_processor_mlp: Optional[Network] = None,
    background_feature_grid: Optional[MultiSphereFeatureGrid] = None,
    num_samples_fine: Optional[int] = None,
    density_noise_std: float = 1.0,
    perturb_sampled_points: bool = False,
    use_dists_in_rendering: bool = True,
    raw2alpha: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    colour_producer: Optional[Callable[[Tensor], Tensor]] = None,
    hybrid_mode_only_rgba: bool = False,
    use_sh_based_rendering: bool = False,
    diffuse_render: bool = False,
    optimized_sampling_mode: bool = False,
    white_bkgd: bool = False,
    gpu_render: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose: bool = False,
    profile: bool = False,
) -> Union[RenderOut, Tuple[RenderOut, RenderOut]]:
    progress_bar = tqdm if verbose else lambda x: x
    num_points_chunk = (
        num_points_chunk
        if processor_mlp is not None
        else (cam_intrinsics.width * cam_intrinsics.height * num_samples_per_ray)
    )

    # free up cuda memory so that rendering can be made faster
    # torch.cuda.empty_cache()

    start_time = time.time()
    last_time = start_time

    # rays are created directly on the GPU and
    casted_rays = cast_rays(
        camera_intrinsics=cam_intrinsics, pose=camera_pose, device=device
    )

    if profile:
        time_taken = time.time() - last_time
        last_time = time.time()
        log.info(
            f"time taken for constructing rays on {device}: {time_taken * 1000} ms"
        )

    rendered_chunks_list, fine_rendered_chunks_list = [], []
    if verbose:
        log.info("Rendering the output chunk by chunk")
    profiling_info_list = []

    for chunk_index in progress_bar(range(0, len(casted_rays.origins), num_rays_chunk)):
        with torch.no_grad():
            raw2alpha = raw2alpha_base if raw2alpha is None else raw2alpha
            colour_producer = (
                torch.sigmoid if colour_producer is None else colour_producer
            )
            rendered_chunk, profiling_info = render_feature_grid(
                rays=casted_rays[chunk_index : chunk_index + num_rays_chunk],
                num_samples=num_samples_per_ray,
                feature_grid=feature_grid,
                point_processor_network=processor_mlp,
                secondary_point_processor_network=secondary_processor_mlp,
                background_processor_network=background_processor_mlp,
                background_feature_grid=background_feature_grid,
                num_samples_fine=num_samples_fine,
                chunk_size=num_points_chunk,
                scene_bounds=scene_bounds,
                density_noise_std=density_noise_std,
                perturb_sampled_points=perturb_sampled_points,
                use_dists_in_rendering=use_dists_in_rendering,
                raw2alpha=raw2alpha,
                colour_producer=colour_producer,
                hybrid_mode_only_rgba=hybrid_mode_only_rgba,
                use_sh_based_rendering=use_sh_based_rendering,
                render_diffuse=diffuse_render,
                optimized_sampling_mode=optimized_sampling_mode,
                white_bkgd=white_bkgd,
            )

            profiling_info_list.append(profiling_info)
            if isinstance(rendered_chunk, RenderOut):
                rendered_chunk = (rendered_chunk,)

        if not gpu_render:
            # detach is not needed, but we do it just in case
            rendered_chunks_list.append(
                rendered_chunk[0].detach().to(torch.device("cpu"))
            )
            if secondary_processor_mlp is not None:
                fine_rendered_chunks_list.append(
                    rendered_chunk[1].detach().to(torch.device("cpu"))
                )
        else:
            # detach is not needed, but we do it just in case
            rendered_chunks_list.append(rendered_chunk[0].detach())
            if secondary_processor_mlp is not None:
                fine_rendered_chunks_list.append(rendered_chunk[1].detach())

    if profile:
        time_taken = time.time() - last_time
        log.info(
            f"time taken for rendering rays chunk by chunk on {device} (The 3 steps together): {time_taken * 1000} ms"
        )
        separation = {}
        for info_chunk in profiling_info_list:
            for key, value in info_chunk.items():
                separation[key] = separation.get(key, []) + [value]
        log.info(f"sampling time: {np.sum(separation['sampling']).item()} ms")
        log.info(f"processing time: {np.sum(separation['processing']).item()} ms")
        log.info(f"accumulation time: {np.sum(separation['accumulation']).item()} ms")
        last_time = time.time()

    rendered_output = reshape_rendered_output(
        collate_rendered_output(rendered_chunks_list), camera_intrinsics=cam_intrinsics
    )

    if profile:
        time_taken = time.time() - last_time
        log.info(
            f"time taken for collating and reshaping the rendered chunks on {device}: {time_taken * 1000} ms"
        )

    if profile:
        time_taken = time.time() - start_time
        log.info(f"Total time taken for rendering the image: {time_taken * 1000} ms")

    # free up the cuda memory containing elements from the rendering
    # torch.cuda.empty_cache()

    if secondary_processor_mlp is not None:
        fine_rendered_output = reshape_rendered_output(
            collate_rendered_output(fine_rendered_chunks_list),
            camera_intrinsics=cam_intrinsics,
        )
        return rendered_output, fine_rendered_output

    return rendered_output
