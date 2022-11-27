import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
import torch
from torch import Tensor

from thre3d_atom.rendering.volumetric.voxels import GridLocation
from torch.backends import cudnn
from torchvision.utils import make_grid

from projects.thre3ingan.singans.networks import (
    Thre3dGenerator,
)
from projects.thre3ingan.singans.thre3d_singan import (
    Thre3dSinGan,
    Thre3dSinGanRenderingParameters,
)
from thre3d_atom.utils.config_utils import str2bool
from thre3d_atom.utils.imaging_utils import (
    CameraIntrinsics,
    CameraPose,
    SceneBounds,
    to8b,
    pose_spherical,
)
from thre3d_atom.utils.logging import log

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# noinspection PyProtectedMember
def render_path(
    thre3d_singan: Thre3dSinGan,
    render_poses: List[CameraPose],
    camera_intrinsics: CameraIntrinsics,
    out_dir: Path,
    render_grid_size: Tuple[int, int],
    latent_size: int,
    render_downsample_factor: Optional[int] = None,
    save_frames: bool = True,
    noise_vectors: Optional[Tensor] = None,
    render_stage: Optional[int] = None,
) -> None:
    if render_downsample_factor is not None:
        # Render downsampled for speed
        height, width, focal = camera_intrinsics
        height = height // render_downsample_factor
        width = width // render_downsample_factor
        focal = focal / render_downsample_factor
        camera_intrinsics = CameraIntrinsics(height, width, focal)

    rgb_images, disparity_maps = [], []

    colour_frames_path, disparity_frames_path = None, None
    if save_frames:
        colour_frames_path = out_dir / f"colour_frames"
        disparity_frames_path = out_dir / f"disparity_frames"
        colour_frames_path.mkdir(parents=True, exist_ok=True)
        disparity_frames_path.mkdir(parents=True, exist_ok=True)

    # cache the feature grid to speed up rendering:
    voxel_size = thre3d_singan._render_params.voxel_size
    grid_location = GridLocation()

    # noise_vectors:
    if noise_vectors is None:
        noise_vectors = torch.randn(
            np.prod(render_grid_size),
            latent_size,
            *thre3d_singan._generator.feature_grid_shape_at_stage(1),
        ).to(device)

    # render all the poses along the path
    total_frames = len(render_poses) + 1
    for render_pose_num, render_pose in enumerate(render_poses):
        log.info(f"rendering frame number: ({render_pose_num + 1}/{total_frames})")
        colour_frame_images_list, disparity_frame_images_list = [], []
        for noise_num, noise_vector in enumerate(noise_vectors, 1):
            log.info(f"rendering sample: {noise_num}")
            rendered_output = thre3d_singan.render(
                camera_intrinsics,
                render_pose,
                thre3d_singan._render_params.scene_bounds,
                voxel_size,
                grid_location,
                input_noise=noise_vector[None, ...],
                verbose=True,
                stage=render_stage,
            )
            colour_frame_images_list.append(rendered_output.colour)
            disparity_frame_images_list.append(rendered_output.disparity)

        colour_frame = torch.stack(colour_frame_images_list, dim=0).permute(0, 3, 1, 2)
        colour_frame = make_grid(
            colour_frame, nrow=render_grid_size[-1], padding=0
        ).permute(1, 2, 0)

        disparity_frame = torch.stack(disparity_frame_images_list, dim=0).permute(
            0, 3, 1, 2
        )
        disparity_frame = make_grid(
            disparity_frame, nrow=render_grid_size[-1], padding=0
        ).permute(1, 2, 0)

        if save_frames:
            imageio.imwrite(
                colour_frames_path / f"{render_pose_num}.png",
                to8b(colour_frame.numpy()),
            )
            imageio.imwrite(
                disparity_frames_path / f"{render_pose_num}.png",
                to8b(disparity_frame.numpy()),
            )
        else:
            rgb_images.append(colour_frame.numpy())
            disparity_maps.append(disparity_frame.numpy())

        if render_pose_num == 0:
            log.info(
                f"Shapes of rendered images -> "
                f"Colour: {colour_frame.shape}, Disparity: {disparity_frame.shape}"
            )

    if not save_frames:
        rgb_images = np.stack(rgb_images, 0)
        disparity_maps = np.stack(disparity_maps, 0)

        imageio.mimwrite(out_dir / "colour_video.mp4", to8b(rgb_images), fps=24)
        imageio.mimwrite(out_dir / "disparity_video.mp4", to8b(disparity_maps), fps=24)


def parse_arguments(args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Thre3d SinGAN Reconstruction 360 render demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("-i", "--model_path", action="store", type=Path, required=False,
                        help="path to nerf snapshot")
    parser.add_argument("-o", "--output_dir", action="store", type=Path, required=False,
                        help="path to the output directory")

    # optional arguments
    # rendering related arguments
    parser.add_argument("--render_stage", action="store", type=int, required=False, default=None,
                        help="Stage from which output is supposed to be rendered")
    parser.add_argument("--num_rays_chunk", action="store", type=int, required=False, default=512,
                        help="number of rays to be rendered simultaneously on a GPU")
    parser.add_argument("--num_samples_per_ray", action="store", type=int, required=False, default=None,
                        help="number of points to be sampled per ray during rendering. "
                             "Uses the training time count by default. Increasing this gives cleaner renders"
                             "but takes longer.")
    parser.add_argument("--radius", action="store", type=float, required=False, default=4.0,
                        help="radius of the circle traced by the camera")
    parser.add_argument("--num_frames", action="store", type=int, required=False, default=42,
                        help="number of frames to be rendered")
    parser.add_argument("--height", action="store", type=int, required=False, default=189,
                        help="height of the rendered video")
    parser.add_argument("--width", action="store", type=int, required=False, default=252,
                        help="width of the rendered video")
    parser.add_argument("--focal", action="store", type=float, required=False, default=208.369491577,
                        help="focal length of the pinhole camera")
    parser.add_argument("--grid_size", action="store", type=int, required=False, nargs=2, default=(2, 3),
                        help="Number of random samples to be shown in the visualization")
    parser.add_argument("--scene_bounds_dolly_adder", action="store", type=float, required=False, default=0.0,
                        help="additive factor to camera_scene_bounds for camera dolly_in or dolly_out")
    parser.add_argument("--save_frames", action="store", type=str2bool, required=False, default=False,
                        help="whether to write individual frames or directly write the video. "
                             "Use this when the video is too big")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


# noinspection PyProtectedMember
def render_360(args: argparse.Namespace) -> None:
    # Load the pretrained model:
    loaded_model = torch.load(args.model_path)
    thre3d_gen = Thre3dGenerator(
        num_stages=loaded_model["thre3d_gen"]["num_stages"],
        base_feature_grid_dims=loaded_model["thre3d_gen"]["base_feature_grid_dims"],
        output_feature_size=loaded_model["thre3d_gen"]["output_feature_size"],
        upsample_factor=loaded_model["thre3d_gen"]["upsample_factor"],
        fmap_base=loaded_model["thre3d_gen"]["fmap_base"],
        fmap_max=loaded_model["thre3d_gen"]["fmap_max"],
        fmap_min=loaded_model["thre3d_gen"]["fmap_min"],
        fmap_decay=loaded_model["thre3d_gen"]["fmap_decay"],
        use_trilinear_upsampling=loaded_model["thre3d_gen"]["use_trilinear_upsampling"],
        use_dists_in_rendering=loaded_model["thre3d_gen"]["use_dists_in_rendering"],
        device=device,
    )
    thre3d_gen.load_state_dict(loaded_model["thre3d_gen"]["state_dict"])

    # construct the 3D singan object from the restored state:
    # override scene_bounds for camera dolly:
    render_params = loaded_model["render_params"]
    scene_bounds = render_params["scene_bounds"]
    scene_bounds = SceneBounds(
        scene_bounds.near + args.scene_bounds_dolly_adder,
        scene_bounds.far + args.scene_bounds_dolly_adder,
    )
    render_params["scene_bounds"] = scene_bounds

    thre3d_singan = Thre3dSinGan(
        thre3d_gen=thre3d_gen,
        render_params=Thre3dSinGanRenderingParameters(**render_params),
    )

    # override the num_rays_chunk since it could have been trained on a bigger GPU
    thre3d_singan._render_params.num_rays_chunk = args.num_rays_chunk
    if args.num_samples_per_ray is not None:
        thre3d_singan._render_params.num_samples_per_ray = args.num_samples_per_ray

    # obtain the poses for rendering this demo
    render_poses = [
        pose_spherical(yaw, pitch, args.radius)
        for (pitch, yaw) in zip(
            list(np.linspace(0, -90, args.num_frames // 2).tolist())
            + list(np.linspace(-90, 0, args.num_frames // 2).tolist()),
            np.linspace(-180, 180, args.num_frames)[:-1],
        )
    ]

    # make sure the output_dir exists:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    camera_intrinsics = CameraIntrinsics(args.height, args.width, args.focal)

    # render the reconstruction video
    render_path(
        thre3d_singan,
        render_poses,
        camera_intrinsics,
        out_dir=args.output_dir / "reconstruction",
        render_grid_size=args.grid_size,
        latent_size=loaded_model["thre3d_gen"]["output_feature_size"],
        save_frames=args.save_frames,
        noise_vectors=loaded_model["fixed_recon_noise"],
        render_stage=args.render_stage,
    )

    # render the gan based video
    render_path(
        thre3d_singan,
        render_poses,
        camera_intrinsics,
        out_dir=args.output_dir / "gan",
        render_grid_size=args.grid_size,
        latent_size=loaded_model["thre3d_gen"]["output_feature_size"],
        save_frames=args.save_frames,
        render_stage=args.render_stage,
    )


def main() -> None:
    render_360(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
