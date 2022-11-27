from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import trimesh
from torch.utils.data import IterableDataset

from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS
from thre3d_atom.utils.imaging_utils import adjust_dynamic_range
from thre3d_atom.utils.types import AxisAlignedBoundingBox


class PointOccupancyMeshDataset(IterableDataset):
    def __init__(
        self,
        mesh_path: Path,
        sampling_boundary_slack_percentage: float = 5.0,
    ) -> None:
        super().__init__()

        # logic related state information
        self._mesh = trimesh.load_mesh(mesh_path)
        self._sampling_boundary_slack_percentage = sampling_boundary_slack_percentage
        self._batch_size = 1
        self._aabb = self._setup_aabb()
        self._slackened_aabb = self._slacken_aabb(
            self._aabb, self._sampling_boundary_slack_percentage
        )

    def _setup_aabb(self) -> AxisAlignedBoundingBox:
        return AxisAlignedBoundingBox(
            x_range=(
                float(self._mesh.vertices[:, 0].min()),
                float(self._mesh.vertices[:, 0].max()),
            ),
            y_range=(
                float(self._mesh.vertices[:, 1].min()),
                float(self._mesh.vertices[:, 1].max()),
            ),
            z_range=(
                float(self._mesh.vertices[:, 2].min()),
                float(self._mesh.vertices[:, 2].max()),
            ),
        )

    @staticmethod
    def _slacken_aabb(
        tight_aabb: AxisAlignedBoundingBox, slack_percentage: float
    ) -> AxisAlignedBoundingBox:
        x_extent = tight_aabb.x_range[1] - tight_aabb.x_range[0]
        y_extent = tight_aabb.y_range[1] - tight_aabb.y_range[0]
        z_extent = tight_aabb.z_range[1] - tight_aabb.z_range[0]

        total_x_slack = (slack_percentage / 100.0) * x_extent
        total_y_slack = (slack_percentage / 100.0) * y_extent
        total_z_slack = (slack_percentage / 100.0) * z_extent

        return AxisAlignedBoundingBox(
            x_range=(
                tight_aabb.x_range[0] - (total_x_slack / 2),
                tight_aabb.x_range[1] + (total_x_slack / 2),
            ),
            y_range=(
                tight_aabb.y_range[0] - (total_y_slack / 2),
                tight_aabb.y_range[1] + (total_y_slack / 2),
            ),
            z_range=(
                tight_aabb.z_range[0] - (total_z_slack / 2),
                tight_aabb.z_range[1] + (total_z_slack / 2),
            ),
        )

    def _sample_point_on_slackened_aabb(self) -> np.array:
        random_points = np.random.uniform(
            size=(self._batch_size, NUM_COORD_DIMENSIONS)
        ).astype(np.float32)
        random_points[:, 0] = adjust_dynamic_range(
            random_points[:, 0],
            drange_in=(0, 1),
            drange_out=self._slackened_aabb.x_range,
        )
        random_points[:, 1] = adjust_dynamic_range(
            random_points[:, 1],
            drange_in=(0, 1),
            drange_out=self._slackened_aabb.y_range,
        )
        random_points[:, 2] = adjust_dynamic_range(
            random_points[:, 2],
            drange_in=(0, 1),
            drange_out=self._slackened_aabb.z_range,
        )
        return random_points

    @property
    def aabb(self) -> AxisAlignedBoundingBox:
        return self._aabb

    @property
    def grid_size(self) -> Tuple[float, float, float]:
        x_extent = np.abs(self._aabb.x_range[1] - self._aabb.x_range[0])
        y_extent = np.abs(self._aabb.y_range[1] - self._aabb.y_range[0])
        z_extent = np.abs(self._aabb.z_range[1] - self._aabb.z_range[0])
        return x_extent, y_extent, z_extent

    def __len__(self) -> int:
        raise ValueError(
            f"Cannot query the length of the "
            f"PointOccupancyMeshDataset dataset. "
            f"The length of this dataset is infinite ... "
        )

    def __getitem__(self, item) -> None:
        raise ValueError(
            f"Cannot use this "
            f"PointOccupancyMeshDataset dataset "
            f"in map-style indexed mode."
        )

    def __iter__(self):
        # Ensure that the random seed for random point generation
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # means that we are not doing single-process data loading.
            np.random.seed(42 + worker_info.id)
        return self

    def __next__(self) -> np.array:
        # ================================================================================
        # TODO: Accelerate the Ground truth occupancy calculation on the GPU if possible.
        # ================================================================================
        random_point = self._sample_point_on_slackened_aabb()
        occupancy = self._mesh.contains(random_point).astype(np.float32)
        return np.concatenate([random_point[0], occupancy])
        # ================================================================================
