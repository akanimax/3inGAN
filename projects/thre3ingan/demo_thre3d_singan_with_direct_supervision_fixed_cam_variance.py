import argparse
import sys
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

from projects.thre3ingan.singans import (
    create_thre3d_singan_with_direct_supervision_from_saved_model,
    Thre3dSinGanWithDirectSupervision,
)
from thre3d_atom.utils.imaging_utils import (
    to8b,
    pose_spherical,
    CameraPose,
    postprocess_disparity_map,
    SceneBounds,
    scale_camera_intrinsics,
)
from thre3d_atom.utils.logging import log


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Thre3d SinGAN reconstruction and varied-gan 3D samples demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("-i", "-m", "--model_path",
                        action="store", type=Path, required=True, help="path to the trained 3dSGDS model")
    parser.add_argument("-o", "--output_dir",
                        action="store", type=Path, required=True, help="path to the output directory")

    # non required arguments:
    parser.add_argument("--camera_path", action="store",
                        type=str, required=False, default="360",
                        help="What type of camera path to use. Should be one of 'rotating' or '360'")
    parser.add_argument("--num_frames", action="store",
                        type=int, required=False, default=21,
                        help="number of frames in the rendered gif (each frame has a different seed)")
    parser.add_argument("--num_camera_views", action="store",
                        type=int, required=False, default=4,
                        help="number of different camera views visualized in render")
    parser.add_argument("--render_stage", action="store",
                        type=int, required=False, default=None, help="Stage from which output is to be rendered")
    parser.add_argument("--camera_intrinsics_scale_factor", action="store",
                        type=float, required=False, default=1.0,
                        help="scale the camera-intrinsics for "
                             "higher (> 1.0) or lower (< 1.0) rendered resolution")
    parser.add_argument("--camera_dolly", action="store",
                        type=float, required=False, default=0.0,
                        help="value of camera dolly. +ve is out and -ve is in")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def render_variance_visualization(
    tds: Thre3dSinGanWithDirectSupervision,
    hem_rad: float,
    output_dir: Path,
    num_seeds: int,
    render_poses: List[CameraPose],
    stage: Optional[int] = None,
    camera_intrinsics_scale_factor: Optional[float] = None,
) -> None:
    # make sure output dir exists:
    output_dir.mkdir(parents=True, exist_ok=True)

    # setup camera intrinsics for rendering
    camera_intrinsics = scale_camera_intrinsics(
        tds.render_params.camera_intrinsics, camera_intrinsics_scale_factor
    )

    # setup noise seeds for rendering different versions
    sample_noises = [
        torch.randn(
            1,
            *tds.reconstruction_noise.shape[1:],
            device=tds.reconstruction_noise.device,
        )
        for _ in range(num_seeds)
    ]

    # setup the view grid:
    num_views = len(render_poses)
    tile_nrows = int(np.ceil(np.sqrt(num_views)))

    # render the output seed-by-seed for the required camera poses:
    log.info(f"Rendering output seed by seed ...")
    for seed_num, sample_noise in enumerate(sample_noises, 1):
        log.info(f"rendering samples for seed ... {seed_num}")
        rendered_colours, rendered_disparities = [], []
        for pose in tqdm(render_poses):
            with torch.no_grad():
                render_out = tds.render_random_sample(
                    hem_rad=hem_rad,
                    camera_pose=pose,
                    camera_intrinsics=camera_intrinsics,
                    scene_bounds=tds.render_params.scene_bounds,
                    random_noise=sample_noise,
                    stage=stage,
                )
            rendered_colours.append(render_out.colour)
            rendered_disparities.append(render_out.disparity)

        # make tiles out of rendered colours and combine them side-by-side
        colour_tile = make_grid(
            torch.stack(rendered_colours, dim=0).permute(0, 3, 1, 2), nrow=tile_nrows
        ).permute(1, 2, 0)
        disparity_tile = make_grid(
            torch.stack(rendered_disparities, dim=0).permute(0, 3, 1, 2),
            nrow=tile_nrows,
        ).permute(1, 2, 0)
        disparity_tile = postprocess_disparity_map(disparity_tile.numpy()[..., 0])

        # make a frame by attaching colours and disparities:
        colour_tile = to8b(colour_tile.numpy())
        seed_frame = np.concatenate([colour_tile, disparity_tile], axis=1)

        # save the frame to disk:
        imageio.imwrite(output_dir / f"{seed_num}.png", seed_frame)


def demo_render(args: argparse.Namespace) -> None:
    tsds, extra_info = create_thre3d_singan_with_direct_supervision_from_saved_model(
        args.model_path
    )

    tsds.render_params.num_rays_chunk = 1024  # reduced to fit on local GPUs
    # and most other GPUs at the cost of a bit of time

    # compute the rendering parameters:
    hem_rad = extra_info["camera_hemisphere_radius"] + args.camera_dolly
    orig_sb = tsds.render_params.scene_bounds
    tsds.render_params.scene_bounds = SceneBounds(
        orig_sb.near + args.camera_dolly, orig_sb.far + args.camera_dolly
    )

    # this is done so that the last one is ignored:
    num_cameras = args.num_camera_views + 1
    if args.camera_path.lower() == "rotating":
        render_poses = [
            pose_spherical(yaw, pitch, hem_rad)
            for (pitch, yaw) in zip(
                list(np.linspace(0, -90, num_cameras // 2).tolist())
                + list(np.linspace(-90, 0, num_cameras // 2).tolist()),
                np.linspace(-180, 180, num_cameras)[:-1],
            )
        ]
    elif args.camera_path.lower() == "360":
        fixed_pitch = -20
        render_poses = [
            pose_spherical(yaw, pitch, hem_rad)
            for (pitch, yaw) in zip(
                [fixed_pitch] * num_cameras,
                np.linspace(-180, 180, num_cameras)[:-1],
            )
        ]
    else:
        raise ValueError(f"Unknown camera_path requested: {args.camera_path}")

    # make sure that the output directory exists:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # render the camera path for the reconstruction case:
    log.info("Rendering the variance visualization ...")
    render_variance_visualization(
        hem_rad=hem_rad,
        output_dir=args.output_dir,
        tds=tsds,
        render_poses=render_poses,
        num_seeds=args.num_frames,
        stage=args.render_stage,
        camera_intrinsics_scale_factor=args.camera_intrinsics_scale_factor,
    )


def main() -> None:
    demo_render(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
