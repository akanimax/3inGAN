import argparse
import sys
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
import torch
from projects.thre3ingan.singans import (
    create_thre3d_singan_with_direct_supervision_from_saved_model,
    Thre3dSinGanWithDirectSupervision,
)
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
from tqdm import tqdm


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Thre3d SinGAN variations 360 demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("-i", "-m", "--model_path",
                        action="store", type=Path, required=True, help="path to the trained 3dSGDS model")
    parser.add_argument("-o", "--output_dir",
                        action="store", type=Path, required=True, help="path to the output directory")

    # non required arguments:
    parser.add_argument("--num_frames", action="store",
                        type=int, required=False, default=360, help="number of frames in the rendered video")
    parser.add_argument("--switch_factor", action="store",
                        type=int, required=False, default=40, help="number of frames after which noise is resampled")
    parser.add_argument("--camera_pitch", action="store",
                        type=float, required=False, default=-30.0, help="camera pitch in the spherical render path")
    parser.add_argument("--camera_dolly", action="store",
                        type=float, required=False, default=0.0,
                        help="value of camera dolly. +ve is out and -ve is in")
    parser.add_argument("--camera_intrinsics_scale_factor", action="store",
                        type=float, required=False, default=1.0,
                        help="scale the camera-intrinsics for "
                             "higher (> 1.0) or lower (< 1.0) rendered resolution")
    parser.add_argument("--render_stage", action="store",
                        type=int, required=False, default=None, help="Stage from which output is to be rendered")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def render_path(
    hem_rad: float,
    output_dir: Path,
    tds: Thre3dSinGanWithDirectSupervision,
    render_poses: List[CameraPose],
    stage: Optional[int] = None,
    switch_factor: int = 10,
) -> None:
    # make sure output dir exists:
    output_dir.mkdir(parents=True, exist_ok=True)

    # render the output frame-by-frame for the required camera poses:
    log.info(f"Rendering output frame by frame ...")
    random_noise = torch.randn_like(tds.reconstruction_noise).to(
        tds.reconstruction_noise.device
    )
    tds.render_params.num_rays_chunk = 1024  # reduced to fit on local GPUs
    for frame_num, pose in tqdm(enumerate(render_poses, 1)):
        with torch.no_grad():
            if frame_num <= switch_factor:
                render_out = tds.render_reconstruction(
                    stage=stage,
                    hem_rad=hem_rad,
                    camera_pose=pose,
                    camera_intrinsics=tds.render_params.camera_intrinsics,
                    scene_bounds=tds.render_params.scene_bounds,
                )
            else:
                # we only resample noise on the switched main noise frame
                # then onwards (otherwise) keep using the same noise
                use_fixed_noise = True

                if frame_num % switch_factor == 0:
                    use_fixed_noise = False
                    random_noise = torch.randn_like(tds.reconstruction_noise).to(
                        tds.reconstruction_noise.device
                    )
                render_out = tds.render_random_sample(
                    hem_rad=hem_rad,
                    camera_pose=pose,
                    camera_intrinsics=tds.render_params.camera_intrinsics,
                    scene_bounds=tds.render_params.scene_bounds,
                    random_noise=random_noise,
                    stage=stage,
                    use_fixed_noise=use_fixed_noise,
                )

        colour_frame = render_out.colour.numpy()
        colour_frame = np.concatenate(
            [colour_frame, render_out.extra[EXTRA_ACCUMULATED_WEIGHTS].numpy()], axis=-1
        )
        disparity_frame = postprocess_disparity_map(
            render_out.disparity.numpy()[..., 0]
        )
        disparity_frame = np.concatenate(
            [
                disparity_frame,
                to8b(render_out.extra[EXTRA_ACCUMULATED_WEIGHTS].numpy()),
            ],
            axis=-1,
        )

        # make a frame by attaching colours and disparities:
        frame = np.concatenate([to8b(colour_frame), disparity_frame], axis=1)

        # save the frame to disk:
        imageio.imwrite(output_dir / f"{frame_num}.png", frame)


def demo_render(args: argparse.Namespace) -> None:
    tsds, extra_info = create_thre3d_singan_with_direct_supervision_from_saved_model(
        args.model_path
    )

    hem_rad = extra_info["camera_hemisphere_radius"]
    original_scene_bounds = tsds.render_params.scene_bounds
    tsds.render_params.scene_bounds = SceneBounds(
        original_scene_bounds.near + args.camera_dolly,
        original_scene_bounds.far + args.camera_dolly,
    )
    render_poses = [
        pose_spherical(yaw, args.camera_pitch, hem_rad + args.camera_dolly)
        for yaw in np.linspace(90.0, 450, args.num_frames)[:-1]
    ]

    # make sure that the output directory exists:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # render the camera path for the reconstruction case:
    log.info("Rendering demo ...")
    tsds.render_params.camera_intrinsics = scale_camera_intrinsics(
        tsds.render_params.camera_intrinsics,
        scale_factor=args.camera_intrinsics_scale_factor,
    )
    render_path(
        hem_rad=hem_rad + args.camera_dolly,
        output_dir=args.output_dir,
        tds=tsds,
        render_poses=render_poses,
        stage=args.render_stage,
        switch_factor=args.switch_factor,
    )


def main() -> None:
    demo_render(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
