import torch
import numpy as np
import matplotlib.pyplot as plt
from thre3d_atom.rendering.volumetric.implicit import cast_rays
from thre3d_atom.rendering.volumetric.voxels import (
    FeatureGrid,
    VoxelSize,
    render_feature_grid,
)
from thre3d_atom.utils.imaging_utils import (
    pose_spherical,
    CameraIntrinsics,
    SceneBounds,
)


def _plot_all_cube_sides(
    feature_grid: FeatureGrid,
    camera_intrinsics: CameraIntrinsics,  # shouldn't be too high
    num_samples_per_ray: int,
    scene_bounds: SceneBounds,
    radius: float,
    device: torch.device,
) -> None:
    height, width, _ = camera_intrinsics

    # render all 6 sides of the cube:
    for side, (yaw, pitch) in enumerate(
        ((0, 0), (90, 0), (180, 0), (270, 0), (0, -90), (0, 90)), 1
    ):
        camera_pose = pose_spherical(yaw=yaw, pitch=pitch, radius=radius)
        rays = cast_rays(camera_intrinsics, camera_pose)

        # render the feature_grid:
        rendered_output = render_feature_grid(
            rays.to(device),
            num_samples_per_ray,
            feature_grid,
            scene_bounds,
            raw2alpha=lambda x, _: torch.clip(x, 0.0, 1.0),
            colour_producer=lambda x: x,
            perturb_sampled_points=False,
        )

        # show the rendered_colour:
        plt.figure(f"side {side}")
        plt.imshow(
            rendered_output.colour.reshape(height, width, 3).detach().cpu().numpy()
        )

    plt.show()


def test_trilinear_interpolation_single_cube(device: torch.device) -> None:
    # fmt: off
    feature_grid = FeatureGrid(
        features=torch.tensor(
            [
                1.0, 0.0, 0.0, np.random.uniform(0.0, 0.1, 1).item(),
                0.0, 1.0, 0.0, np.random.uniform(0.0, 0.1, 1).item(),
                0.0, 0.0, 1.0, np.random.uniform(0.0, 0.1, 1).item(),
                1.0, 1.0, 0.0, np.random.uniform(0.0, 0.1, 1).item(),
                0.0, 1.0, 1.0, np.random.uniform(0.0, 0.1, 1).item(),
                1.0, 0.0, 1.0, np.random.uniform(0.0, 0.1, 1).item(),
                1.0, 1.0, 1.0, np.random.uniform(0.0, 0.1, 1).item(),
                0.0, 0.0, 0.0, np.random.uniform(0.0, 0.1, 1).item(),
            ],
            device=device,
            dtype=torch.float32,
        ).reshape(2, 2, 2, 4).permute(3, 0, 1, 2),
        voxel_size=VoxelSize(2, 2, 2),
    )
    # fmt: on

    _plot_all_cube_sides(
        feature_grid,
        CameraIntrinsics(100, 100, 120),
        num_samples_per_ray=128,
        scene_bounds=SceneBounds(2.0, 6.0),
        radius=4.0,
        device=device,
    )


def test_trilinear_interpolation_feature_grid(device: torch.device) -> None:
    # fmt: off
    corner_alpha = np.random.uniform(0.0, 0.1, 1).item()
    feature_grid = FeatureGrid(
        features=torch.tensor(
            [
                1.0, 0.0, 0.0, corner_alpha,
                1.0, 0.0, 0.0, corner_alpha,
                1.0, 0.0, 0.0, corner_alpha,
                1.0, 1.0, 0.0, corner_alpha,
                0.0, 1.0, 1.0, corner_alpha,
                1.0, 1.0, 0.0, corner_alpha,
                1.0, 0.0, 0.0, corner_alpha,
                1.0, 0.0, 0.0, corner_alpha,
                1.0, 0.0, 0.0, corner_alpha,
                0.0, 1.0, 0.0, corner_alpha,
                1.0, 1.0, 1.0, corner_alpha,
                0.0, 1.0, 0.0, corner_alpha,
                1.0, 0.0, 0.0, corner_alpha,
                1.0, 0.0, 0.0, corner_alpha,
                1.0, 0.0, 0.0, corner_alpha,
                0.0, 1.0, 0.0, corner_alpha,
                0.0, 1.0, 0.0, corner_alpha,
                0.0, 1.0, 0.0, corner_alpha,
            ],
            device=device,
            dtype=torch.float32,
        ).reshape(3, 2, 3, 4).permute(3, 0, 1, 2),
        voxel_size=VoxelSize(1, 1, 1),
    )
    # fmt: on

    _plot_all_cube_sides(
        feature_grid,
        CameraIntrinsics(100, 100, 100),
        num_samples_per_ray=128,
        scene_bounds=SceneBounds(2.0, 6.0),
        radius=4.0,
        device=device,
    )
