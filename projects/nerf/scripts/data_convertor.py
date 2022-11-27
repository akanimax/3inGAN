import argparse

import imageio
import numpy as np
from pathlib import Path

from load_blender import load_blender_data
from load_deepvoxels import load_dv_data
from load_llff import load_llff_data
from thre3d_atom.utils.imaging_utils import to8b
from thre3d_atom.utils.config_utils import str2bool
import json


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Script for converting the data from one of NeRF's format into thre3d_atom's format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "-i",
        "--data_path",
        action="store",
        type=Path,
        required=True,
        help="path to the nerf's data directory",
    )
    parser.add_argument(
        "-d",
        "--dataset_type",
        action="store",
        type=str,
        required=True,
        help="type of source dataset",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        action="store",
        type=Path,
        required=True,
        help="path to the output directory",
    )

    # optional arguments
    parser.add_argument(
        "--factor",
        action="store",
        type=int,
        required=False,
        default=8,
        help="image downsample factor",
    )
    parser.add_argument(
        "--spherify",
        action="store",
        type=str2bool,
        required=False,
        default=False,
        help="whether to spherify poses",
    )
    parser.add_argument(
        "--dv_shape",
        action="store",
        type=str,
        required=False,
        default="greek",
        help="shape for loading deepvoxels data. one of greek / armchair/ cube / vase",
    )
    parsed_args = parser.parse_args()
    return parsed_args


def main(args: argparse.Namespace) -> None:
    if args.dataset_type == "llff":
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.data_path,
            args.factor,
            recenter=True,
            bd_factor=0.75,
            spherify=args.spherify,
        )
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print("Loaded llff", images.shape, render_poses.shape, hwf, args.data_path)

        print("DEFINING BOUNDS")

        near = bds.min() * 0.9
        far = bds.max() * 1.0
        print("NEAR FAR", near, far)

    elif args.dataset_type == "blender":
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.data_path, False
        )
        print("Loaded blender", images.shape, render_poses.shape, hwf, args.datadir)

        near = 2.0
        far = 6.0

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1.0 - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == "deepvoxels":

        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.dv_shape, basedir=args.data_path
        )

        print("Loaded deepvoxels", images.shape, render_poses.shape, hwf, args.datadir)

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.0
        far = hemi_R + 1.0

    else:
        print("Unknown dataset type", args.dataset_type, "exiting")
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)

    save_file = args.output_dir / "camera_params.json"

    camera_params = {}
    for image_num, (bd, pose) in enumerate(zip(bds, poses), 1):
        camera_params[f"{image_num}.png"] = {
            "extrinsic": {
                "rotation": pose[:3, :3].astype(np.str).tolist(),
                "translation": pose[:3, 3:].astype(np.str).tolist(),
            },
            "intrinsic": {
                "height": str(H),
                "width": str(W),
                "focal": str(focal),
                "bounds": bd.astype(np.str).tolist(),
            },
        }

    image_dir = save_file.parent / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for image_num, image in enumerate(images, 1):
        imageio.imwrite(image_dir / f"{image_num}.png", to8b(image))

    with open(str(save_file), "w") as json_dumper:
        json.dump(camera_params, json_dumper, indent=4)


if __name__ == "__main__":
    main(parse_arguments())
