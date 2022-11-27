import numpy as np
import pytest
import torch

from thre3d_atom.rendering.volumetric.implicit import (
    cast_rays,
)
from thre3d_atom.rendering.volumetric.sample import sample_uniform_points_on_rays
from thre3d_atom.rendering.volumetric.render_interface import Rays
from thre3d_atom.utils.imaging_utils import CameraIntrinsics, CameraPose, SceneBounds


@pytest.mark.parametrize(
    "camera_intrinsics, pose",
    [
        (
            CameraIntrinsics(1, 1, 1),
            CameraPose(np.identity(3), np.zeros(shape=(3, 1))),
        ),
        (
            CameraIntrinsics(100, 100, 100),
            CameraPose(np.identity(3), np.random.uniform(low=0, high=100, size=(3, 1))),
        ),
    ],
)
def test_cast_rays(camera_intrinsics: CameraIntrinsics, pose: CameraPose) -> None:
    # TODO: improve this test (more example cases and test more behaviour)
    # GIVEN: camera_intrinsics and pose

    # WHEN: virtual rays are casted
    rays = cast_rays(camera_intrinsics, pose)

    # THEN:
    # 1. The origins of all the rays must be the same
    # 2. The norm of all the direction_vectors must be >= 1
    #    (This is because except for the direction vector at the center (norm == 1.0)
    #     pixel, all the others must have a norm greater than the center one)
    assert np.all(rays.origins.numpy() == rays.origins.numpy()[0])
    assert np.all(np.linalg.norm(rays.directions.numpy(), axis=-1) >= 1.0)


def test_sample_uniform_points_on_rays(
    batch_size: int, num_samples: int, device: torch.device
) -> None:
    # GIVEN: mocked random rays and mocked scene bounds
    random_origins = torch.randn((3,)).unsqueeze(0).repeat(batch_size, 1).to(device)
    random_rays = torch.randn(batch_size, 3).to(device)
    mock_rays = Rays(random_origins, random_rays)
    mock_scene_bounds = SceneBounds(0, 1)

    # WHEN: sampling uniform points on the rays
    sampled_points = sample_uniform_points_on_rays(
        mock_rays, mock_scene_bounds, num_samples
    )

    # THEN: the (ray-origin) norms of points per ray and their depth values
    #  must always be monotonically increasing
    sampled_points_numpy = sampled_points.points.cpu().numpy()
    ray_origins_numpy = mock_rays.origins[..., None, :].cpu().numpy()
    depths_numpy = sampled_points.depths.cpu().numpy()
    assert np.all(
        np.diff(np.linalg.norm(sampled_points_numpy - ray_origins_numpy, axis=-1)) >= 0
    )
    assert np.all(np.diff(depths_numpy) >= 0)
