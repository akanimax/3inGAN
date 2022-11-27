import argparse
import sys
from pathlib import Path
from typing import List, Optional

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
from thre3d_atom.utils.imaging_utils import (
    to8b,
    pose_spherical,
    CameraPose,
    SceneBounds,
    scale_camera_intrinsics,
    CameraIntrinsics,
)
from thre3d_atom.utils.logging import log


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Sets up rendered views (in a particular folder structure) for the "
        "Quality and Diversity metrics to be computed",
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
    parser.add_argument("--num_views", action="store",
                        type=int, required=False, default=5, help="number of views used for generation")
    parser.add_argument("--num_gan_samples", action="store",
                        type=int, required=False, default=5, help="number of gan samples used for generation")
    parser.add_argument("--stage", action="store",
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


def render_synchronized_samples(
    model_rf: VolumetricModel,
    model_tsds: Thre3dSinGanWithDirectSupervision,
    output_dir: Path,
    hem_rad: float,
    render_poses: List[CameraPose],
    camera_intrinsics: CameraIntrinsics,
    scene_bounds: SceneBounds,
    num_gan_samples: int,
    stage: Optional[int] = None,
) -> None:
    # ensure that the background for the model_rf is disabled if one exists:
    model_rf.background_render_mlp = None

    # setup the output directories where the rendered images are going to be written
    gt_render_dir = output_dir / "gt_views"
    random_sample_dir = output_dir / "random_samples"
    for directory in (gt_render_dir, random_sample_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # setup random noise seeds for the thre3ingan model
    sample_noises = [
        torch.randn(
            1,
            *model_tsds.reconstruction_noise.shape[1:],
            device=model_tsds.reconstruction_noise.device,
        )
        for _ in range(num_gan_samples)
    ]

    # render the output frame-by-frame for the required camera poses:
    log.info(f"Rendering output pose by pose ...")
    for pose_num, pose in tqdm(enumerate(render_poses, 1)):

        log.info(f"Rendering gt_view for pose: {pose_num}")
        # render the GT model for the current pose
        with torch.no_grad():
            render_out = model_rf.render(
                camera_intrinsics=camera_intrinsics,
                camera_pose=pose,
                scene_bounds=scene_bounds,
            )
        rendered_colour = render_out.colour
        imageio.imwrite(
            gt_render_dir / f"{pose_num}.png", to8b(rendered_colour.numpy())
        )

        # setup the directory for the random seed samples for the current pose:
        random_sample_dir_for_current_pose = random_sample_dir / f"{pose_num}"
        random_sample_dir_for_current_pose.mkdir(parents=True, exist_ok=True)

        # render and write the GAN samples for the same pose
        log.info(f"Rendering gan samples for pose: {pose_num}")
        for sample_num, sample_noise in tqdm(enumerate(sample_noises, 1)):
            with torch.no_grad():
                render_out = model_tsds.render_random_sample(
                    hem_rad=hem_rad,
                    camera_pose=pose,
                    camera_intrinsics=camera_intrinsics,
                    scene_bounds=scene_bounds,
                    random_noise=sample_noise,
                    stage=stage,
                    use_fixed_noise=False,
                )
            rendered_colour = render_out.colour
            imageio.imwrite(
                random_sample_dir_for_current_pose / f"{sample_num}.png",
                to8b(rendered_colour.numpy()),
            )


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

    num_poses = args.num_views
    render_poses = [
        pose_spherical(yaw, pitch, hem_rad)
        for (pitch, yaw) in zip(
            np.random.uniform(0, 0, num_poses), np.random.uniform(0, 360, num_poses)
        )
    ]

    # render the camera path for the GT volumetric model (ReluField)
    log.info("Rendering the view synchronized real and generated samples ...")
    render_synchronized_samples(
        model_rf=vol_mod,
        model_tsds=tsds,
        output_dir=args.output_dir,
        hem_rad=hem_rad,
        render_poses=render_poses,
        camera_intrinsics=camera_intrinsics,
        scene_bounds=scene_bounds,
        num_gan_samples=args.num_gan_samples,
    )


def main() -> None:
    render_visualization(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
