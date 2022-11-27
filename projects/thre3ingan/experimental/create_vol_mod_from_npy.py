import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

from thre3d_atom.modules.volumetric_model.volumetric_model import (
    VolumetricModel,
    VolumetricModelRenderingParameters,
)
from thre3d_atom.rendering.volumetric.voxels import (
    GridLocation,
    FeatureGrid,
    VoxelSize,
)
from thre3d_atom.utils.constants import (
    NUM_RGBA_CHANNELS,
)
from thre3d_atom.utils.imaging_utils import SceneBounds, CameraIntrinsics


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Converts Feature-Grid (+ mlp model) into an RGBA grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("-i", "-m", "--model_path",
                        action="store", type=Path, required=True, help="path to the trained 3dSGDS model")
    parser.add_argument("-o", "--output_dir",
                        action="store", type=Path, required=True, help="path to the output directory")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


## noinspection PyUnresolvedReferences
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments(sys.argv[1:])

    # load the numpy model:
    np_model = np.load(args.model_path, allow_pickle=True)
    features = torch.from_numpy(np_model["grid"]).to(device=device)
    grid_size = np_model["grid_size"]
    grid_location = GridLocation(*np_model["grid_center"])

    grid_dim = features.shape[:-1]
    x_voxel_size = grid_size[0] / (grid_dim[0] - 1)
    y_voxel_size = grid_size[1] / (grid_dim[1] - 1)
    z_voxel_size = grid_size[2] / (grid_dim[2] - 1)

    feature_grid = FeatureGrid(
        features=features.permute(3, 0, 1, 2),
        voxel_size=VoxelSize(x_voxel_size, y_voxel_size, z_voxel_size),
        grid_location=grid_location,
        tunable=True,
    )

    render_params = VolumetricModelRenderingParameters(
        num_rays_chunk=1024,
        num_points_chunk=65536,
        num_samples_per_ray=256,
        num_fine_samples_per_ray=0,
        perturb_sampled_points=True,
        density_noise_std=0.0,
    )
    vol_mod = VolumetricModel(
        render_params=render_params,
        grid_dims=grid_dim,
        feature_dims=NUM_RGBA_CHANNELS,
        grid_size=grid_size,
        grid_center=grid_location,
        device=device,
    )
    vol_mod.feature_grid = feature_grid
    torch.save(
        vol_mod.get_save_info(
            extra_info={
                "scene_bounds": SceneBounds(0.1, 2.5),
                "camera_intrinsics": CameraIntrinsics(256, 256, 256),
                "hemispherical_radius": 1.0,
            }
        ),
        f"{args.output_dir}/model_rgba.pth",
    )


if __name__ == "__main__":
    main()
