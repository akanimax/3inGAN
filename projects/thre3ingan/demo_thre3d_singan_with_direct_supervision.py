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
from thre3d_atom.utils.config_utils import str2bool
from thre3d_atom.utils.constants import EXTRA_ACCUMULATED_WEIGHTS
from thre3d_atom.utils.imaging_utils import (
    to8b,
    pose_spherical,
    CameraPose,
    postprocess_disparity_map,
    SceneBounds,
    scale_camera_intrinsics,
)
from thre3d_atom.utils.logging import log


def parse_arguments(args: str) -> argparse.Namespace:
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
    parser.add_argument("--reconstruction_mode", action="store",
                        type=str2bool, required=False, default=False, help="Whether to render the reconstruction video")
    parser.add_argument("--camera_path", action="store",
                        type=str, required=False, default="360",
                        help="What type of camera path to use. Should be one of 'rotating' or '360'")
    parser.add_argument("--num_frames", action="store",
                        type=int, required=False, default=42, help="number of frames in the rendered video")
    parser.add_argument("--num_gan_samples", action="store",
                        type=int, required=False, default=6, help="number of gan samples needed for demo")
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


def render_path(
    hem_rad: float,
    output_dir: Path,
    tds: Thre3dSinGanWithDirectSupervision,
    render_poses: List[CameraPose],
    num_gan_samples: int,
    reconstruction_mode: bool,
    stage: Optional[int] = None,
    camera_intrinsics_scale_factor: Optional[float] = None,
) -> None:
    # make sure output dir exists:
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_intrinsics = scale_camera_intrinsics(
        tds.render_params.camera_intrinsics, camera_intrinsics_scale_factor
    )

    # use either the reconstruction noise or randomly sampled noises:
    if reconstruction_mode:
        sample_noises = [tds.reconstruction_noise]
    else:
        sample_noises = [
            torch.randn(
                1,
                *tds.reconstruction_noise.shape[1:],
                device=tds.reconstruction_noise.device,
            )
            for _ in range(num_gan_samples)
        ]

    tile_nrows = int(np.ceil(np.sqrt(num_gan_samples)))

    # render the output frame-by-frame for the required camera poses:
    log.info(f"Rendering output frame by frame ...")
    for frame_num, pose in enumerate(render_poses, 1):
        log.info("rendering sample ...")
        rendered_colours, rendered_disparities, rendered_accs = [], [], []
        for sample_noise in tqdm(sample_noises):
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
            rendered_accs.append(render_out.extra[EXTRA_ACCUMULATED_WEIGHTS])

        # make tiles out of rendered colours and combine them side-by-side
        colour_tile = make_grid(
            torch.stack(rendered_colours, dim=0).permute(0, 3, 1, 2), nrow=tile_nrows
        ).permute(1, 2, 0)
        disparity_tile = make_grid(
            torch.stack(rendered_disparities, dim=0).permute(0, 3, 1, 2),
            nrow=tile_nrows,
        ).permute(1, 2, 0)
        disparity_tile = postprocess_disparity_map(disparity_tile.numpy()[..., 0])
        acc_tile = to8b(
            make_grid(
                torch.stack(rendered_accs, dim=0).permute(0, 3, 1, 2), nrow=tile_nrows
            )
            .permute(1, 2, 0)
            .numpy()
        )[..., 0:1]
        disparity_tile = np.concatenate([disparity_tile, acc_tile], axis=-1)

        # make a frame by attaching colours and disparities:
        colour_tile = to8b(colour_tile.numpy())
        colour_tile = np.concatenate([colour_tile, acc_tile], axis=-1)
        frame = np.concatenate([colour_tile, disparity_tile], axis=1)

        # save the frame to disk:
        imageio.imwrite(output_dir / f"{frame_num}.png", frame)


def demo_render(args: argparse.Namespace) -> None:
    tsds, extra_info = create_thre3d_singan_with_direct_supervision_from_saved_model(
        args.model_path
    )

    tsds.render_params.num_rays_chunk = 1024  # reduced to fit on local GPUs

    # we use a dummy hemispherical radius value for now:
    hem_rad = extra_info["camera_hemisphere_radius"] + args.camera_dolly
    init = tsds.render_params.scene_bounds
    tsds.render_params.scene_bounds = SceneBounds(
        init.near + args.camera_dolly, init.far + args.camera_dolly
    )

    if args.camera_path.lower() == "rotating":
        render_poses = [
            pose_spherical(yaw, pitch, hem_rad)
            for (pitch, yaw) in zip(
                list(np.linspace(0, -90, args.num_frames // 2).tolist())
                + list(np.linspace(-90, 0, args.num_frames // 2).tolist()),
                np.linspace(-180, 180, args.num_frames)[:-1],
            )
        ]
    elif args.camera_path.lower() == "360":
        render_poses = [
            pose_spherical(yaw, pitch, hem_rad)
            for (pitch, yaw) in zip(
                [-20] * args.num_frames,
                np.linspace(-180, 180, args.num_frames)[:-1],
            )
        ]
    else:
        raise ValueError(f"Unknown camera_path requested: {args.camera_path}")

    # make sure that the output directory exists:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # render the camera path for the reconstruction case:
    log.info("Rendering reconstruction output ...")
    render_path(
        hem_rad=hem_rad,
        output_dir=args.output_dir / "recon",
        tds=tsds,
        render_poses=render_poses,
        num_gan_samples=args.num_gan_samples,
        reconstruction_mode=True,
        stage=args.render_stage,
        camera_intrinsics_scale_factor=args.camera_intrinsics_scale_factor,
    )

    # render the paths for gan_samples:
    log.info("Rendering gan based output ... Please be patient :D")
    render_path(
        hem_rad=hem_rad,
        output_dir=args.output_dir / "gan",
        tds=tsds,
        render_poses=render_poses,
        num_gan_samples=args.num_gan_samples,
        reconstruction_mode=False,
        stage=args.render_stage,
        camera_intrinsics_scale_factor=args.camera_intrinsics_scale_factor,
    )


def main() -> None:
    demo_render(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
