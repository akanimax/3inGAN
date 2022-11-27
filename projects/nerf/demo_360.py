import argparse
import sys
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
import torch

from thre3d_atom.modules.nerf import Nerf, NerfRenderingParameters
from thre3d_atom.modules.nerf import get_sota_nerf_net
from thre3d_atom.utils.config_utils import str2bool
from thre3d_atom.utils.constants import EXTRA_ACCUMULATED_WEIGHTS
from thre3d_atom.utils.imaging_utils import (
    CameraIntrinsics,
    CameraPose,
    SceneBounds,
    to8b,
    pose_spherical,
    postprocess_disparity_map,
)
from thre3d_atom.utils.logging import log


def render_path(
    nerf: Nerf,
    render_poses: List[CameraPose],
    camera_intrinsics: CameraIntrinsics,
    scene_bounds: SceneBounds,
    out_dir: Path,
    render_downsample_factor: Optional[int] = None,
    save_frames: bool = True,
) -> None:
    if render_downsample_factor is not None:
        # Render downsampled for speed
        height, width, focal = camera_intrinsics
        height = height // render_downsample_factor
        width = width // render_downsample_factor
        focal = focal / render_downsample_factor
        camera_intrinsics = CameraIntrinsics(height, width, focal)

    rgb_images, disparity_maps = [], []

    acc_map_path, colour_frames_path, disparity_frames_path = None, None, None
    if save_frames:
        acc_map_path = out_dir / f"acc_map_path"
        colour_frames_path = out_dir / f"colour_frames"
        disparity_frames_path = out_dir / f"disparity_frames"
        acc_map_path.mkdir(parents=True, exist_ok=True)
        colour_frames_path.mkdir(parents=True, exist_ok=True)
        disparity_frames_path.mkdir(parents=True, exist_ok=True)

    total_frames = len(render_poses) + 1
    for render_pose_num, render_pose in enumerate(render_poses):
        log.info(f"rendering frame number: ({render_pose_num + 1}/{total_frames})")
        # use only the fine rendered_output for saving
        _, rendered_output = nerf.render(camera_intrinsics, render_pose, scene_bounds)
        if save_frames:
            imageio.imwrite(
                acc_map_path / f"{render_pose_num}.png",
                to8b(rendered_output.extra[EXTRA_ACCUMULATED_WEIGHTS].numpy()),
            )
            imageio.imwrite(
                colour_frames_path / f"{render_pose_num}.png",
                to8b(rendered_output.colour.numpy()),
            )
            imageio.imwrite(
                disparity_frames_path / f"{render_pose_num}.png",
                postprocess_disparity_map(rendered_output.disparity.squeeze().numpy()),
            )
        else:
            rgb_images.append(rendered_output.colour.numpy())
            disparity_maps.append(
                postprocess_disparity_map(rendered_output.disparity.squeeze().numpy())
            )

        if render_pose_num == 0:
            log.info(
                f"Shapes of rendered images -> "
                f"Colour: {rendered_output.colour.shape}, Disparity: {rendered_output.disparity.shape}"
            )

    if not save_frames:
        rgb_images = np.stack(rgb_images, 0)
        disparity_maps = np.stack(disparity_maps, 0)
        imageio.mimwrite(out_dir / "colour_video.mp4", to8b(rgb_images), fps=24)
        imageio.mimwrite(out_dir / "disparity_video.mp4", disparity_maps, fps=24)


def parse_arguments(args: str) -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("NeRF 360 render demo",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument("-i", "--model_path", action="store", type=Path, required=False,
                        help="path to nerf snapshot")
    parser.add_argument("-o", "--output_dir", action="store", type=Path, required=False,
                        help="path to the output directory")

    # optional arguments
    # rendering related arguments
    parser.add_argument("--num_rays_chunk", action="store", type=int, required=False, default=512,
                        help="number of rays to be rendered simultaneously on a GPU")
    parser.add_argument("--num_frames", action="store", type=int, required=False, default=42,
                        help="number of frames to be rendered")
    parser.add_argument("--save_frames", action="store", type=str2bool, required=False, default=False,
                        help="whether to write individual frames or directly write the video. "
                             "Use this when the video is too big")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def render_360(args: argparse.Namespace) -> None:
    loaded_model = torch.load(args.model_path)

    nerf_net_coarse = get_sota_nerf_net()
    nerf_net_fine = get_sota_nerf_net()
    nerf_net_coarse.load_weights(loaded_model["nerf_net_coarse"])
    nerf_net_fine.load_weights(loaded_model["nerf_net_fine"])

    render_params = NerfRenderingParameters(**loaded_model["render_params"])
    # override the num_rays_chunk since you might be running the demo on a different machine than the one
    # trained on.
    render_params.num_rays_chunk = args.num_rays_chunk

    #  extract the extra rendering info from the loaded_model:
    radius = loaded_model["extra_info"]["hemispherical_radius"]
    scene_bounds = loaded_model["extra_info"]["scene_bounds"]
    camera_intrinsics = loaded_model["extra_info"]["camera_intrinsics"]

    # build a nerf object from the loaded state
    nerf = Nerf(nerf_net_coarse, render_params, nerf_net_fine)

    # obtain the poses for rendering this demo
    render_poses = [
        pose_spherical(angle, -30.0, radius)
        for angle in np.linspace(-180, 180, args.num_frames)[:-1]
    ]

    # make sure the output_dir exists:
    args.output_dir.mkdir(parents=True, exist_ok=True)

    nerf.toggle_all_networks("eval")
    render_path(
        nerf,
        render_poses,
        camera_intrinsics,
        scene_bounds,
        out_dir=args.output_dir,
        save_frames=args.save_frames,
    )


def main() -> None:
    render_360(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
