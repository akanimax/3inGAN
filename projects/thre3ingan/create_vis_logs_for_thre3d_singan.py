import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
import torch
from tqdm import tqdm

from projects.thre3ingan.singans import (
    create_thre3d_singan_with_direct_supervision_from_saved_model,
    Thre3dSinGanWithDirectSupervision,
)
from thre3d_atom.modules.volumetric_model.volumetric_model import (
    create_vol_mod_from_saved_model,
    VolumetricModel,
)
from thre3d_atom.utils.config_utils import int_or_none
from thre3d_atom.utils.constants import EXTRA_ACCUMULATED_WEIGHTS
from thre3d_atom.utils.imaging_utils import (
    to8b,
    pose_spherical,
    CameraPose,
    postprocess_disparity_map,
    SceneBounds,
    scale_camera_intrinsics,
    CameraIntrinsics,
)
from thre3d_atom.utils.logging import log


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Thre3d SinGAN reconstruction and varied-gan 3D sample creation script. "
        "Writes the RGB and depth maps as transparent png images to disk. "
        "Note that we use the accumulated alpha for the transparency channel of the written images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("-i", "-m", "--model_path",
                        action="store", type=Path, required=True, help="path to the trained 3dSGDS model")
    parser.add_argument("-i_g", "-m_g", "--gt_model_path",
                        action="store", type=Path, required=True, help="path to the ground truth ReluField model")
    parser.add_argument("-o", "--output_dir",
                        action="store", type=Path, required=True, help="path to the output directory")

    # non required arguments:
    parser.add_argument("--num_frames_per_seed", action="store",
                        type=int, required=False, default=42, help="number of frames rendered for each seed")
    parser.add_argument("--num_gan_samples", action="store",
                        type=int, required=False, default=6, help="number of gan samples needed for demo")
    parser.add_argument("--stage", action="store",
                        type=int, required=False, default=None, help="Stage from which output is to be rendered")
    parser.add_argument("--camera_path", action="store",
                        type=str, required=False, default="360",
                        help="What type of camera path to use. Should be one of 'rotating' or '360'")
    parser.add_argument("--starting_random_noise_shape", action="store",
                        type=int_or_none, required=False, default=None, nargs=3,
                        help="size of the starting noise (when different gives retargeted output)")
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


def render_path_vol_mod(
    model: VolumetricModel,
    output_dir: Path,
    camera_intrinsics: CameraIntrinsics,
    scene_bounds: SceneBounds,
    render_poses: List[CameraPose],
) -> None:
    # create the output directory for the GT model render visualization
    output_path = output_dir / "gt"
    output_path.mkdir(parents=True, exist_ok=True)

    # make sure to remove the background from the vol-mod before rendering
    model.background_render_mlp = None

    # render the output frame-by-frame for the required camera poses:
    log.info(f"Rendering output frame by frame ...")
    for frame_num, pose in tqdm(enumerate(render_poses, 1)):
        with torch.no_grad():
            render_out = model.render(
                camera_intrinsics=camera_intrinsics,
                camera_pose=pose,
                scene_bounds=scene_bounds,
            )
        rendered_colour = render_out.colour
        rendered_disparity = render_out.disparity
        rendered_acc = render_out.extra[EXTRA_ACCUMULATED_WEIGHTS]

        # make tiles out of rendered colours and combine them side-by-side
        colour_frame = to8b(rendered_colour.numpy())
        disparity_frame = postprocess_disparity_map(rendered_disparity.numpy()[..., 0])
        acc_frame = to8b(rendered_acc.numpy())

        # attach the acc_frame to the disparity and the colour frames as the alpha channel
        disparity_frame = np.concatenate([disparity_frame, acc_frame], axis=-1)
        colour_frame = np.concatenate([colour_frame, acc_frame], axis=-1)

        # save the frames to disk:
        imageio.imwrite(output_path / f"{frame_num}_col.png", colour_frame)
        imageio.imwrite(output_path / f"{frame_num}_dis.png", disparity_frame)


def render_path(
    model: Thre3dSinGanWithDirectSupervision,
    output_dir: Path,
    hem_rad: float,
    render_poses: List[CameraPose],
    camera_intrinsics: CameraIntrinsics,
    scene_bounds: SceneBounds,
    num_gan_samples: int,
    reconstruction_mode: bool,
    stage: Optional[int] = None,
    starting_random_noise_shape: Tuple[int, int, int] = None,
) -> None:
    # use either the reconstruction noise or randomly sampled noises:
    if reconstruction_mode:
        sample_noises = [model.reconstruction_noise]
    else:
        if starting_random_noise_shape is not None:
            noise_shape = starting_random_noise_shape
        else:
            noise_shape = model.reconstruction_noise.shape[2:]
        sample_noises = [
            torch.randn(
                1,
                1,
                *noise_shape,
                device=model.reconstruction_noise.device,
            )
            for _ in range(num_gan_samples)
        ]

    # render the output frame-by-frame for the required camera poses:
    log.info(f"Rendering output frame by frame ...")
    for sample_num, sample_noise in enumerate(sample_noises, 1):
        if not reconstruction_mode:
            log.info(f"Currently rendering seed number: {sample_num}")

        for frame_num, pose in tqdm(enumerate(render_poses, 1)):
            # setup the output path for the visualization png images
            if reconstruction_mode:
                output_path = output_dir / "recon"
            else:
                output_path = output_dir / f"{sample_num}"
            output_path.mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                render_out = model.render_random_sample(
                    hem_rad=hem_rad,
                    camera_pose=pose,
                    camera_intrinsics=camera_intrinsics,
                    scene_bounds=scene_bounds,
                    random_noise=sample_noise,
                    stage=stage,
                    use_fixed_noise=True,
                )
            rendered_colour = render_out.colour
            rendered_disparity = render_out.disparity
            rendered_acc = render_out.extra[EXTRA_ACCUMULATED_WEIGHTS]

            # make tiles out of rendered colours and combine them side-by-side
            colour_frame = to8b(rendered_colour.numpy())
            disparity_frame = postprocess_disparity_map(
                rendered_disparity.numpy()[..., 0]
            )
            acc_frame = to8b(rendered_acc.numpy())

            # attach the acc_frame to the disparity and the colour frames as the alpha channel
            disparity_frame = np.concatenate([disparity_frame, acc_frame], axis=-1)
            colour_frame = np.concatenate([colour_frame, acc_frame], axis=-1)

            # save the frames to disk:
            imageio.imwrite(output_path / f"{frame_num}_col.png", colour_frame)
            imageio.imwrite(output_path / f"{frame_num}_dis.png", disparity_frame)


def render_visualization(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the volumetric model and the thre3ingan models
    vol_mod, _ = create_vol_mod_from_saved_model(args.gt_model_path, device=device)
    tsds, extra_info = create_thre3d_singan_with_direct_supervision_from_saved_model(
        args.model_path
    )
    tsds.render_params.num_rays_chunk = 1024  # reduced to fit on local GPUs

    # setup the rendering camera information
    hem_rad = extra_info["camera_hemisphere_radius"] + args.camera_dolly
    init = tsds.render_params.scene_bounds
    scene_bounds = SceneBounds(
        init.near + args.camera_dolly, init.far + args.camera_dolly
    )
    camera_intrinsics = scale_camera_intrinsics(
        tsds.render_params.camera_intrinsics, args.camera_intrinsics_scale_factor
    )
    num_frames = args.num_frames_per_seed
    if args.camera_path.lower() == "rotating":
        render_poses = [
            pose_spherical(yaw, pitch, hem_rad)
            for (pitch, yaw) in zip(
                list(np.linspace(0, -90, num_frames // 2).tolist())
                + list(np.linspace(-90, 0, num_frames // 2).tolist()),
                np.linspace(-180, 180, num_frames)[:-1],
            )
        ]
    elif args.camera_path.lower() == "360":
        render_poses = [
            pose_spherical(yaw, pitch, hem_rad)
            for (pitch, yaw) in zip(
                [-20] * num_frames,
                np.linspace(-180, 180, num_frames)[:-1],
            )
        ]
    else:
        raise ValueError(f"Unknown camera_path requested: {args.camera_path}")

    # make sure that the output directory exists:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # render the camera path for the GT volumetric model (ReluField)
    log.info("Rendering the GT volmod output ...")
    render_path_vol_mod(
        model=vol_mod,
        output_dir=args.output_dir,
        camera_intrinsics=camera_intrinsics,
        scene_bounds=scene_bounds,
        render_poses=render_poses,
    )

    # render the camera path for the fixed reconstruction seed:
    log.info("Rendering reconstruction output from the trained tsds model ...")
    render_path(
        model=tsds,
        output_dir=args.output_dir,
        hem_rad=hem_rad,
        render_poses=render_poses,
        camera_intrinsics=camera_intrinsics,
        scene_bounds=scene_bounds,
        num_gan_samples=args.num_gan_samples,
        reconstruction_mode=True,
        stage=args.stage,
    )

    # render the camera paths for various remix samples:
    log.info("Rendering remixed output ... Please be patient :D")
    render_path(
        model=tsds,
        output_dir=args.output_dir,
        hem_rad=hem_rad,
        render_poses=render_poses,
        camera_intrinsics=camera_intrinsics,
        scene_bounds=scene_bounds,
        num_gan_samples=args.num_gan_samples,
        reconstruction_mode=False,
        stage=args.stage,
        starting_random_noise_shape=args.starting_random_noise_shape,
    )


def main() -> None:
    render_visualization(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
