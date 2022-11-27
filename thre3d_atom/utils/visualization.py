from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from thre3d_atom.data.loaders import PosedImagesDataset
from thre3d_atom.modules.nerf.utils import ndcize_rays
from thre3d_atom.rendering.volumetric.implicit import cast_rays


def visualize_camera_rays(
    dataset: PosedImagesDataset,
    output_dir: Path,
    num_rays_per_image: int = 30,
    do_ndcize_rays: bool = False,
) -> None:
    all_poses = [
        dataset.extract_pose(camera_param)
        for camera_param in dataset.camera_parameters.values()
    ]
    all_camera_locations = []

    fig = plt.figure()
    fig.suptitle("Camera rays visualization")
    ax = fig.add_subplot(111, projection="3d")
    for pose in all_poses:
        rays = cast_rays(dataset.camera_intrinsics, pose)
        if do_ndcize_rays:
            rays = ndcize_rays(rays, dataset.camera_intrinsics)

        # randomly select only num_rays_per_image rays for visualization
        combined_rays = np.concatenate([rays.origins, rays.directions], axis=-1)
        np.random.shuffle(combined_rays)
        selected_rays = combined_rays[:num_rays_per_image]
        selected_ray_origins, selected_ray_directions = (
            selected_rays[:, :3],
            selected_rays[:, 3:],
        )
        # add the ray origin to camera locations
        all_camera_locations.append(selected_ray_origins[0])

        far_plane = dataset.scene_bounds.far if not do_ndcize_rays else 1.0
        points_a = selected_ray_origins
        points_b = selected_ray_origins + (selected_ray_directions * far_plane)
        # plot all the rays (from point_a to point_b) sequentially
        for (point_a, point_b) in zip(points_a, points_b):
            combined = np.stack([point_a, point_b])
            ax.plot(combined[:, 0], combined[:, 1], combined[:, 2], color="b")
    # scatter all the start points in different colour:
    all_camera_locations = np.stack(all_camera_locations, axis=0)
    ax.scatter(
        all_camera_locations[:, 0],
        all_camera_locations[:, 1],
        all_camera_locations[:, 2],
        color="m",
    )

    # save the figure
    plt.tight_layout()
    plt.savefig(output_dir / "casted_camera_rays.png", dpi=600)
    plt.close(fig)
