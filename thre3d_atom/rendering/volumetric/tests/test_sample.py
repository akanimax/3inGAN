import matplotlib.pyplot as plt
import numpy as np
import torch

# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from thre3d_atom.rendering.volumetric.implicit import cast_rays
from thre3d_atom.rendering.volumetric.sample import (
    sample_uniform_points_on_regular_feature_grid,
)
from thre3d_atom.rendering.volumetric.voxels import FeatureGrid, VoxelSize
from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS
from thre3d_atom.utils.imaging_utils import (
    pose_spherical,
    CameraIntrinsics,
    SceneBounds,
)


def test_sample_uniform_points_on_regular_feature_grid(device: torch.device) -> None:
    # Tunable parameters as follows:
    feature_grid = FeatureGrid(
        features=torch.randn(4, 32, 32, 32),
        voxel_size=VoxelSize(0.03125, 0.03125, 0.03125),
    )
    random_pose = pose_spherical(0, -30, 1.5)
    camera_intrinsics = CameraIntrinsics(128, 128, 100)
    scene_bounds = SceneBounds(0.5, 2.0)
    num_samples = 128

    # get the feature-grid's bounding vertices:
    bounding_vertices = feature_grid.get_bounding_vertices().detach().cpu().numpy()

    # fmt: off
    # Start the plotting:
    # plot the cube:
    fig = plt.figure()
    fig.suptitle("Camera rays visualization")
    ax = fig.add_subplot(111, projection="3d")

    # plot vertices
    ax.scatter3D(bounding_vertices[:, 0], bounding_vertices[:, 1], bounding_vertices[:, 2])
    # list of sides' polygons of figure
    polygons = [[bounding_vertices[0], bounding_vertices[1], bounding_vertices[3], bounding_vertices[2]],
                [bounding_vertices[4], bounding_vertices[5], bounding_vertices[7], bounding_vertices[6]],
                [bounding_vertices[0], bounding_vertices[1], bounding_vertices[5], bounding_vertices[4]],
                [bounding_vertices[2], bounding_vertices[3], bounding_vertices[7], bounding_vertices[6]],
                [bounding_vertices[1], bounding_vertices[3], bounding_vertices[7], bounding_vertices[5]],
                [bounding_vertices[0], bounding_vertices[2], bounding_vertices[6], bounding_vertices[4]]]

    # plot sides
    ax.add_collection3d(Poly3DCollection(polygons, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # fmt: on

    # cast rays and sample points on them:
    rays = cast_rays(camera_intrinsics, random_pose)

    sampled_points = sample_uniform_points_on_regular_feature_grid(
        rays=rays,
        bounds=scene_bounds,
        num_samples=num_samples,
        aabb=feature_grid.aabb,
        perturb=True,
    )
    flat_points = sampled_points.points.reshape(-1, NUM_COORD_DIMENSIONS)
    # randomly select 1K points:
    random_selection = np.random.permutation(len(flat_points))[:1000, ...]

    ax.scatter3D(
        flat_points[random_selection, 0],
        flat_points[random_selection, 1],
        flat_points[random_selection, 2],
        color="magenta",
    )

    # show the figure
    plt.show()
