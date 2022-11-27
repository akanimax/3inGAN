import dataclasses
import json
from abc import ABC
from pathlib import Path
from typing import Tuple, Dict, Any, List, Union, NamedTuple

import numpy as np
import torch
import torch.utils.data as torch_data
from PIL import Image
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn.functional import grid_sample
from torch.utils.data import IterableDataset, WeightedRandomSampler

from thre3d_atom.data.constants import (
    INTRINSIC,
    BOUNDS,
    HEIGHT,
    WIDTH,
    FOCAL,
    EXTRINSIC,
    ROTATION,
    TRANSLATION,
)
from thre3d_atom.data.utils import get_torch_vision_image_transform
from thre3d_atom.rendering.volumetric.implicit import cast_rays
from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS, NUM_COLOUR_CHANNELS
from thre3d_atom.utils.imaging_utils import (
    SceneBounds,
    CameraIntrinsics,
    CameraPose,
    adjust_dynamic_range,
    to8b,
)
from thre3d_atom.utils.logging import log


class PosedImagesDataset(torch_data.Dataset):
    # TODO: A unit tests has been added, but we still need to setup the git-lfs
    #  to have the CI in place
    def __init__(
        self,
        images_dir: Path,
        camera_params_json: Path,
        image_data_range: Tuple[int, int] = (-1, 1),
        test_percentage: float = 0.0,
        test_mode: bool = False,
        unit_normalize_scene_scale: bool = False,
        downsample_factor: int = 1,  # no downsampling by default
        rgba_white_bkgd: bool = False,
    ) -> None:
        assert images_dir.exists(), f"Images dir doesn't exist: {images_dir}"
        assert 1 == 1
        assert (
            camera_params_json.exists()
        ), f"CameraParams file doesn't exist: {camera_params_json}"

        super().__init__()
        image_file_paths = list(images_dir.iterdir())
        with open(str(camera_params_json)) as camera_params_json_file:
            self._camera_parameters = json.load(camera_params_json_file)

        filtered_image_file_paths = image_file_paths
        if len(image_file_paths) != len(self._camera_parameters):
            filtered_image_file_paths = []
            img_file_names = [file_path.name for file_path in image_file_paths]
            for index, img_file_name in enumerate(img_file_names):
                if img_file_name in list(self._camera_parameters.keys()):
                    filtered_image_file_paths.append(image_file_paths[index])

        self._setup_train_test_splits(filtered_image_file_paths, test_percentage)

        # assert that test_percentage is > 0.0 if test_mode is requested
        if test_mode:
            assert (
                0.0 < test_percentage <= 100.0
            ), f"Cannot use the Dataset in Test mode if test_percentage ({test_percentage}) is not valid"

        # setup the Dataset based on the specified mode:
        if test_mode:
            self._image_file_paths = self._test_image_file_paths
        else:
            self._image_file_paths = self._training_image_file_paths

        self._images_dir = images_dir
        self._camera_params_json = camera_params_json
        self._downsample_factor = downsample_factor
        self._scene_bounds = self._setup_scene_bounds()
        self._camera_intrinsics = self._setup_camera_intrinsics()
        self._image_transform = get_torch_vision_image_transform(
            new_size=(self._camera_intrinsics.height, self._camera_intrinsics.width)
        )
        self._image_data_range = image_data_range
        self._unit_normalize_scene_scale = unit_normalize_scene_scale
        self._rgba_white_bkgd = rgba_white_bkgd

        if unit_normalize_scene_scale:
            self._normalize_scene_scale()

        self._cached_images, self._cached_poses = None, None

        # Attempting to cache all the data on the memory ...
        self._gpu_cached_mode = False
        try:
            # try caching all data on the gpu memory first
            log.info(f"Trying to cache all the images on the GPU ...")
            if torch.cuda.is_available():
                self._cached_images, self._cached_poses = self._cache_all_data(
                    torch.device("cuda")
                )
            log.info(f"GPU caching of {len(self._cached_images)} images successful!")
            self._gpu_cached_mode = True
        except RuntimeError as gpu_error:
            log.info(f"GPU data caching attempt failed with: {gpu_error}")
            torch.cuda.empty_cache()
            try:
                # If GPU caching didn't work, then try cpu-caching.
                log.info(f"Trying to cache all the images on the CPU ...")
                self._cached_images, self._cached_poses = self._cache_all_data(
                    torch.device("cpu")
                )
                log.info(
                    f"CPU caching of {len(self._cached_images)} images successful!"
                )
            except RuntimeError as cpu_error:
                # If neither works, then revert to reading file from disk behaviour
                log.info(f"CPU data caching failed with: {cpu_error}")
                log.info(
                    f"Reverting to the default behaviour of reading data "
                    f"elements from the disk."
                )

        # Attempting to cache all the rays on the memory ...
        self._cached_rays = None

        if self._gpu_cached_mode:
            try:
                log.info(f"Trying to cache all the Rays on the GPU ...")
                self._cached_rays = self._cache_rays(torch.device("cuda"))
                log.info(f"GPU caching of Rays successful!")
            except RuntimeError as gpu_error:
                log.info(f"GPU Rays caching attempt failed with: {gpu_error}")
                torch.cuda.empty_cache()
                try:
                    log.info(f"Trying to cache all the Rays on the CPU ...")
                    self._cached_rays = self._cache_rays(torch.device("cpu"))
                    log.info(f"CPU caching of Rays successful!")
                except RuntimeError as cpu_error:
                    log.info(f"CPU Rays caching failed with: {cpu_error}")
                    log.info(
                        f"Reverting to the default behaviour of constructing rays during training "
                    )
        else:
            try:
                log.info(f"Trying to cache all the Rays on the CPU ...")
                self._cached_rays = self._cache_rays(torch.device("cpu"))
                log.info(f"CPU caching of Rays successful!")
            except RuntimeError as cpu_error:
                log.info(f"CPU Rays caching attempt failed with: {cpu_error}")
                log.info(
                    f"Reverting to the default behaviour of constructing rays during training "
                )

    def _cache_rays(self, device: torch.device) -> Dict[Path, Tensor]:
        assert self._cached_images is not None
        assert self._cached_poses is not None

        rays_cache = {}
        for key, pose in self._cached_poses.items():
            casted_rays = cast_rays(
                self._camera_intrinsics,
                CameraPose(rotation=pose[:, :3], translation=pose[:, 3:]),
                device=device,
            )
            rays_tensor = torch.cat(
                [casted_rays.origins, casted_rays.directions], dim=-1
            )
            rays_cache[key] = rays_tensor
        return rays_cache

    def _cache_all_data(
        self, cache_device: torch.device
    ) -> Tuple[Dict[Path, Union[Tensor]], Dict[Path, Union[Tensor]]]:
        images_cache, poses_cache = {}, {}
        for image_file_path in self._image_file_paths:
            # Load the image for caching
            # noinspection PyTypeChecker
            image = Image.open(image_file_path)
            image = self._process_image(image).to(cache_device)
            images_cache[image_file_path] = image

            # Load the pose for caching
            camera_params = self._camera_parameters[image_file_path.name]
            pose = self.extract_pose(camera_params)
            unified_pose = torch.from_numpy(
                np.hstack((pose.rotation, pose.translation))
            ).to(cache_device)
            poses_cache[image_file_path] = unified_pose

        return images_cache, poses_cache

    @property
    def gpu_cached_mode(self) -> bool:
        return self._gpu_cached_mode

    def get_condensed_cache(self) -> Tuple[Tensor, Tensor, Tensor]:
        assert self._cached_images is not None, f"No cache found, can't condense"
        assert self._cached_poses is not None, f"No cache found, can't condense"
        assert self._cached_rays is not None, f"No cache found, can't condense"

        all_images, all_poses, all_rays = [], [], []
        for key in self._cached_images.keys():
            image = self._cached_images[key]
            pose = self._cached_poses[key]
            rays = self._cached_rays[key]
            all_images.append(image)
            all_poses.append(pose)
            all_rays.append(rays)

        all_images = torch.stack(all_images, dim=0)
        all_poses = torch.stack(all_poses, dim=0)
        all_rays = torch.stack(all_rays, dim=0)

        return all_images, all_poses, all_rays

    def _normalize_scene_scale(self):
        all_poses = [
            self.extract_pose(camera_param)
            for camera_param in self._camera_parameters.values()
        ]
        all_locations = np.concatenate(
            [pose.translation for pose in all_poses], axis=-1
        )
        max_norm = np.max(np.linalg.norm(all_locations, axis=0))

        # update all the extrinsic translations by scaling them by the max_norm:
        for k, v in self._camera_parameters.items():
            old_values = self._camera_parameters[k][EXTRINSIC][TRANSLATION]
            self._camera_parameters[k][EXTRINSIC][TRANSLATION][0][0] = str(
                (float(old_values[0][0]) / max_norm)
            )
            self._camera_parameters[k][EXTRINSIC][TRANSLATION][1][0] = str(
                (float(old_values[1][0]) / max_norm)
            )
            self._camera_parameters[k][EXTRINSIC][TRANSLATION][2][0] = str(
                (float(old_values[2][0]) / max_norm)
            )

        # also update the scene_bounds
        self._scene_bounds = SceneBounds(
            (self._scene_bounds.near / max_norm),
            (self._scene_bounds.far / max_norm),
        )

    def _setup_train_test_splits(
        self,
        image_file_paths: List[Path],
        test_percentage: float,
    ) -> None:
        # Note that a seed is set so that everytime the same training and testing split is formed
        # for reproducibility
        np.random.seed(42)
        np.random.shuffle(image_file_paths)

        # allocate the test images and their corresponding camera parameters
        test_split = int(np.ceil((test_percentage / 100) * len(image_file_paths)))
        self._test_image_file_paths = image_file_paths[:test_split]

        # use the remaining as the training set:
        self._training_image_file_paths = image_file_paths[test_split:]

    @property
    def scene_bounds(self) -> SceneBounds:
        return self._scene_bounds

    @scene_bounds.setter
    def scene_bounds(self, scene_bounds: SceneBounds) -> None:
        self._scene_bounds = scene_bounds

    @property
    def camera_intrinsics(self) -> CameraIntrinsics:
        return self._camera_intrinsics

    @property
    def camera_parameters(self) -> Dict[str, Any]:
        return self._camera_parameters

    def get_hemispherical_radius_estimate(self) -> float:
        # noinspection PyTypeChecker
        return (
            np.linalg.norm(
                np.array(
                    [
                        camera_param[EXTRINSIC][TRANSLATION]
                        for camera_param in self._camera_parameters.values()
                    ]
                ).astype(np.float32),
                axis=(-1, -2),
            )
            .mean()
            .item()
        )

    # noinspection PyArgumentList
    def _setup_scene_bounds(self) -> SceneBounds:
        all_bounds = np.vstack(
            [
                np.array(camera_param[INTRINSIC][BOUNDS]).astype(np.float32)
                for camera_param in self._camera_parameters.values()
            ]
        )

        near = all_bounds.min() * 1.0
        far = all_bounds.max() * 1.0
        return SceneBounds(near, far)

    def _setup_camera_intrinsics(self) -> CameraIntrinsics:
        all_camera_intrinsics = np.vstack(
            [
                np.array(
                    [
                        camera_param[INTRINSIC][HEIGHT],
                        camera_param[INTRINSIC][WIDTH],
                        camera_param[INTRINSIC][FOCAL],
                    ]
                ).astype(np.float32)
                for camera_param in self._camera_parameters.values()
            ]
        )
        # make sure that all the intrinsics are the same
        assert np.all(all_camera_intrinsics == all_camera_intrinsics[0, :])

        height, width, focal = all_camera_intrinsics[0, :] / self._downsample_factor
        return CameraIntrinsics(int(height), int(width), focal)

    @staticmethod
    def extract_pose(camera_params: Dict[str, Any]) -> CameraPose:
        rotation = np.array(camera_params[EXTRINSIC][ROTATION]).astype(
            np.float32
        )  # 3 x 3 rotation matrix
        translation = np.array(camera_params[EXTRINSIC][TRANSLATION]).astype(
            np.float32
        )  # 3 x 1 translation vector
        return CameraPose(rotation, translation)

    def _process_image(
        self, image: Union[Tensor, np.array, Image.Image]
    ) -> Union[Tensor, np.array, Image.Image]:
        image = self._image_transform(image)
        if image.shape[0] > 3:
            if image.shape[0] == 4:
                # RGBA image case
                if self._rgba_white_bkgd:
                    # need to composite the image on a white background:
                    rgb, alpha = image[:-1, ...], image[-1:, ...]
                    image = (rgb * alpha) + (1 - alpha)
                else:
                    # premultiply the RGB with alpha to get correct
                    # interpolation
                    image = image[:3, ...] * image[3:, ...]
            else:
                image = image[:3, ...]
        return adjust_dynamic_range(
            image, drange_in=(0, 1), drange_out=self._image_data_range
        )

    def __len__(self) -> int:
        return len(self._image_file_paths)

    def __getitem__(
        self, index: int
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor, None]]:
        # pull the image path at the index
        image_file_path = self._image_file_paths[index]

        # read and normalize the image
        image = (
            self._process_image(Image.open(image_file_path))
            if self._cached_images is None
            else self._cached_images[image_file_path]
        )

        # read the pose of the image
        if self._cached_poses is None:
            # retrieve the camera_parameters of the image
            camera_params = self._camera_parameters[image_file_path.name]
            pose = self.extract_pose(camera_params)
            unified_pose = torch.from_numpy(
                np.hstack((pose.rotation, pose.translation))
            )
        else:
            unified_pose = self._cached_poses[image_file_path]

        # Check if cached Rays exist:
        if self._cached_rays is not None:
            rays = self._cached_rays[image_file_path]
            return image, unified_pose, rays

        return image, unified_pose, None


class PosedImagesDatasetWithImportanceSampling(IterableDataset, ABC):
    class AxisAlignedBoundingBox(NamedTuple):
        # min-max range values for all three dimensions
        x_range: Tuple[float, float]
        y_range: Tuple[float, float]
        z_range: Tuple[float, float]

    def __init__(
        self,
        images_dir: Path,
        camera_params_json: Path,
        image_data_range: Tuple[int, int] = (-1, 1),
        unit_normalize_scene_scale: bool = False,
        downsample_factor: int = 1,  # no downsampling by default
        rgba_white_bkgd: bool = False,
    ) -> None:
        posed_image_dataset = PosedImagesDataset(
            images_dir=images_dir,
            camera_params_json=camera_params_json,
            image_data_range=image_data_range,
            unit_normalize_scene_scale=unit_normalize_scene_scale,
            downsample_factor=downsample_factor,
            rgba_white_bkgd=rgba_white_bkgd,
        )
        self._camera_intrinsics = posed_image_dataset.camera_intrinsics

        (
            self._images,
            self._poses,
            self._rays,
        ) = posed_image_dataset.get_condensed_cache()
        # reshape images to have colours on the last dimension
        self._images = self._images.permute(0, 2, 3, 1)
        # reshape the rays from [Num_images * height * width x 6] to [Num_images x height x width x 6]
        self._rays = self._rays.reshape(
            -1,
            self._camera_intrinsics.height,
            self._camera_intrinsics.width,
            2 * NUM_COORD_DIMENSIONS,
        )

        # more state
        self._device = self._images.device

    def __iter__(self):
        return self

    def __getitem__(self, item) -> None:
        raise ValueError(
            f"Cannot use this "
            f"PosedImagesDatasetWithImportanceSampling dataset "
            f"in map-style indexed mode."
        )

    def __len__(self) -> int:
        raise ValueError(
            f"Cannot query the length of "
            f"PosedImagesDatasetWithImportanceSampling dataset. "
            f"It is infinite!"
        )

    @property
    def dataset_size(self) -> int:
        return int(np.prod(self._images.shape[:-1]))


class PosedImagesDatasetWithFullyRandomImportanceSampling(
    PosedImagesDatasetWithImportanceSampling
):
    def __init__(
        self,
        batch_size: int,
        images_dir: Path,
        camera_params_json: Path,
        image_data_range: Tuple[int, int] = (-1, 1),
        unit_normalize_scene_scale: bool = False,
        downsample_factor: int = 1,
        rgba_white_bkgd: bool = False,
    ) -> None:
        super().__init__(
            images_dir,
            camera_params_json,
            image_data_range,
            unit_normalize_scene_scale,
            downsample_factor,
            rgba_white_bkgd,
        )

        # additional state for the current implementation:
        self._batch_size = batch_size

        # flatten both the rays and the images:
        self._rays = self._rays.reshape(-1, 2 * NUM_COORD_DIMENSIONS)
        self._pixels = self._images.reshape(-1, NUM_COLOUR_CHANNELS)

    def __next__(self) -> Tuple[Tensor, Tensor]:
        random_perm = torch.randperm(len(self._rays), device=self._device)
        selected_batch = random_perm[: self._batch_size]
        return self._pixels[selected_batch], self._rays[selected_batch]


@dataclasses.dataclass
class LossWeightedImportanceSamplingConfig:
    batch_size: int
    random_percentage: float = 30
    loss_weights_gamma: float = 3.0


class PosedImagesDatasetWithLossWeightedImportanceSampling(
    PosedImagesDatasetWithImportanceSampling
):
    def __init__(
        self,
        config: LossWeightedImportanceSamplingConfig,
        images_dir: Path,
        camera_params_json: Path,
        image_data_range: Tuple[int, int] = (-1, 1),
        unit_normalize_scene_scale: bool = False,
        downsample_factor: int = 1,
        rgba_white_bkgd: bool = False,
    ) -> None:
        super().__init__(
            images_dir,
            camera_params_json,
            image_data_range,
            unit_normalize_scene_scale,
            downsample_factor,
            rgba_white_bkgd,
        )

        # additional state for the current implementation:
        self._config = config
        self._batch_size = config.batch_size
        self._random_percentage = config.random_percentage

        # flatten both the rays and the images:
        self._rays = self._rays.reshape(-1, 2 * NUM_COORD_DIMENSIONS)
        self._pixels = self._images.reshape(-1, NUM_COLOUR_CHANNELS)

        # a list to keep track of current weights
        # The current weights are initialized to have a uniform empirical distribution
        self._loss_weights = torch.ones(
            (self._rays.shape[0],),
            dtype=torch.float32,
            device=self._device,
            requires_grad=False,
        )
        self._loss_weights_gamma = config.loss_weights_gamma

    def update_loss_weights(self, indices: Tensor, new_weights: Tensor) -> None:
        self._loss_weights[indices] = new_weights

    @property
    def residual_visualization(self) -> np.array:
        height, width, _ = self._camera_intrinsics
        residual_imgs = self._loss_weights.reshape(-1, height, width)

        heat_map_list = []
        for residual_img in residual_imgs:
            residual_img_numpy = residual_img.detach().cpu().numpy()
            colour_map = plt.get_cmap("jet", lut=1024)
            heat_map_list.append(
                to8b(colour_map(residual_img_numpy))[..., :NUM_COLOUR_CHANNELS]
            )
        return np.stack(heat_map_list, axis=0)

    def __next__(self) -> Tuple[Tensor, Tensor, Tensor]:
        num_random_rays = int(
            np.round((self._random_percentage / 100.0) * self._batch_size)
        )
        num_is_rays = self._batch_size - num_random_rays

        # random rays:
        random_perm = torch.randperm(len(self._rays), device=self._device)
        random_ray_indices = random_perm[:num_random_rays]

        # ===============================================================================
        # Importance sampled rays based on current weights
        # ===============================================================================
        sampler_iter = iter(
            WeightedRandomSampler(
                self._loss_weights ** self._loss_weights_gamma,
                num_is_rays,
                replacement=False,
            )
        )
        importance_sampled_indices = torch.tensor(
            list(sampler_iter),
            dtype=torch.int64,
            device=self._device,
            requires_grad=False,
        )
        # ================================================================================

        selected_indices = torch.cat(
            [random_ray_indices, importance_sampled_indices], dim=0
        )

        return (
            selected_indices,
            self._pixels[selected_indices],
            self._rays[selected_indices],
        )


@dataclasses.dataclass
class VolumetricDensityWeightedImportanceSamplingConfig:
    density_grid: Tensor  # shape -> [W x H x D]
    aabb: PosedImagesDatasetWithImportanceSampling.AxisAlignedBoundingBox
    num_samples: int
    num_stochastic_samples: int = 10_000
    patch_percentage: float = 2.5


class PosedImagesDatasetWithVolumetricDensityWeightedImportanceSampling(
    PosedImagesDatasetWithImportanceSampling
):
    def __init__(
        self,
        config: VolumetricDensityWeightedImportanceSamplingConfig,
        images_dir: Path,
        camera_params_json: Path,
        image_data_range: Tuple[int, int] = (-1, 1),
        unit_normalize_scene_scale: bool = False,
        downsample_factor: int = 1,
        rgba_white_bkgd: bool = False,
    ) -> None:
        super().__init__(
            images_dir,
            camera_params_json,
            image_data_range,
            unit_normalize_scene_scale,
            downsample_factor,
            rgba_white_bkgd,
        )
        self._config = config

        # create shorthands for easy use:
        self._density_grid = config.density_grid
        self._aabb = config.aabb
        self._num_samples = config.num_samples
        self._num_stochastic_samples = config.num_stochastic_samples
        self._patch_percentage = config.patch_percentage

    @property
    def density_grid(self) -> Tensor:
        return self._density_grid

    @density_grid.setter
    def density_grid(self, new_density_grid: Tensor) -> None:
        self._config.density_grid = self._density_grid = new_density_grid

    def _random_points_in_std_cube(self) -> Tensor:
        # sample random uniform points
        points = torch.rand(
            self._num_stochastic_samples,
            NUM_COORD_DIMENSIONS,
            dtype=torch.float32,
            device=self._device,
        )

        # scale the points correctly:
        x_p = 2.0 / self._density_grid.shape[0]
        y_p = 2.0 / self._density_grid.shape[1]
        z_p = 2.0 / self._density_grid.shape[2]
        points[:, 0] = (points[:, 0] * (2.0 - x_p)) - (1 - (x_p / 2.0))
        points[:, 1] = (points[:, 1] * (2.0 - y_p)) - (1 - (y_p / 2.0))
        points[:, 2] = (points[:, 2] * (2.0 - z_p)) - (1 - (z_p / 2.0))
        return points

    def _points_unit_cube_to_world(self, points: Tensor) -> Tensor:
        x_p = 2.0 / self._density_grid.shape[0]
        y_p = 2.0 / self._density_grid.shape[1]
        z_p = 2.0 / self._density_grid.shape[2]
        points[:, 0] = adjust_dynamic_range(
            points[:, 0],
            drange_in=((-1 + (x_p / 2.0)), (1 - (x_p / 2.0))),
            drange_out=self._aabb.x_range,
        )
        points[:, 1] = adjust_dynamic_range(
            points[:, 1],
            drange_in=((-1 + (y_p / 2.0)), (1 - (y_p / 2.0))),
            drange_out=self._aabb.y_range,
        )
        points[:, 2] = adjust_dynamic_range(
            points[:, 2],
            drange_in=((-1 + (z_p / 2.0)), (1 - (z_p / 2.0))),
            drange_out=self._aabb.z_range,
        )
        return points

    def __next__(self) -> Tuple[Tensor, Tensor]:
        # Randomly sample points on the grid.
        random_stochastic_points = self._random_points_in_std_cube()
        with torch.no_grad():
            rsp_densities = (
                grid_sample(
                    self._density_grid[None, None, ...],
                    random_stochastic_points[None, None, None, ...],
                    align_corners=False,
                )
                .permute(0, 2, 3, 4, 1)
                .squeeze()
            )

        # --------------------------------------------------------------------------------
        # Importance sampling of points
        # --------------------------------------------------------------------------------
        rsp_densities = torch.relu(rsp_densities)
        sampler_iter = iter(
            WeightedRandomSampler(rsp_densities, self._num_samples, replacement=False)
        )
        selected_points_indices = list(sampler_iter)
        # --------------------------------------------------------------------------------

        selected_points = random_stochastic_points[selected_points_indices]
        selected_points_world = self._points_unit_cube_to_world(selected_points)

        # project all the points on all the camera planes and obtain their coord_locations
        cam_rots, cam_origins = self._poses[:, :, :-1], self._poses[:, :, -1]
        cam_normals = torch.zeros_like(cam_origins)
        cam_normals[:, 2] = -1.0  # [0, 0 -1] vector
        cam_normals = (cam_rots @ cam_normals[..., None]).squeeze(dim=-1)

        cam_plane_centers = cam_origins + (cam_normals * self._camera_intrinsics.focal)
        cam_to_selected_points = (
            cam_origins[None, :, :] - selected_points_world[:, None, :]
        )
        cam_to_selected_points_unit = (
            cam_to_selected_points / cam_to_selected_points.norm(dim=-1, keepdim=True)
        )
        cam_rays_t_vals = (
            (cam_plane_centers[None, :, :] * cam_normals[None, :, :]).sum(
                dim=-1, keepdims=True
            )
        ) / (
            (cam_to_selected_points * cam_normals[None, :, :]).sum(
                dim=-1, keepdims=True
            )
        )
        selected_points_projs = cam_origins[None, :, :] + (
            cam_to_selected_points_unit * cam_rays_t_vals
        )
        image_plane_vecs = cam_plane_centers[None, :, :] - selected_points_projs

        image_y_axes, image_x_axes = torch.zeros_like(cam_normals), torch.zeros_like(
            cam_normals
        )
        image_y_axes[:, 1] = -1.0
        image_x_axes[:, 0] = 1.0
        image_y_axes = (cam_rots @ image_y_axes[..., None]).squeeze(dim=-1)
        image_x_axes = (cam_rots @ image_x_axes[..., None]).squeeze(dim=-1)
        # noinspection PyTypeChecker
        y_dists = torch.round(
            (image_plane_vecs * image_y_axes[None, :, :]).sum(dim=-1, keepdims=True)
            + (self._camera_intrinsics.height / 2.0)
        ).type(torch.int32)
        # noinspection PyTypeChecker
        x_dists = torch.round(
            (image_plane_vecs * image_x_axes[None, :, :]).sum(dim=-1, keepdims=True)
            + (self._camera_intrinsics.width / 2.0)
        ).type(torch.int32)
        pixel_locations = torch.cat([y_dists, x_dists], dim=-1)

        # use the coord_locations to crop patches of images and rays to be returned
        patch_height = int(
            np.round((self._patch_percentage / 100) * self._camera_intrinsics.height)
        )
        patch_width = int(
            np.round((self._patch_percentage / 100) * self._camera_intrinsics.width)
        )
        patch_height += 1 if patch_height % 2 == 0 else 0
        patch_width += 1 if patch_width % 2 == 0 else 0
        half_patch_height = (patch_height - 1) // 2
        half_patch_width = (patch_width - 1) // 2

        # ============================================================================
        # TODO: Accelerate the following slow code-block
        # ============================================================================
        selected_pixels, selected_rays = [], []
        for pixel_location in pixel_locations:
            for img_ind, pix_loc in enumerate(pixel_location):
                if half_patch_height != 0 and half_patch_width != 0:
                    pixel_chunk = self._images[
                        img_ind,
                        pix_loc[0] - half_patch_height : pix_loc[0] + half_patch_height,
                        pix_loc[1] - half_patch_width : pix_loc[1] + half_patch_width,
                        :,
                    ]
                    rays_chunk = self._rays[
                        img_ind,
                        pix_loc[0] - half_patch_height : pix_loc[0] + half_patch_height,
                        pix_loc[1] - half_patch_width : pix_loc[1] + half_patch_width,
                        :,
                    ]
                elif half_patch_height == 0 and half_patch_width != 0:
                    pixel_chunk = self._images[
                        img_ind,
                        pix_loc[0],
                        pix_loc[1] - half_patch_width : pix_loc[1] + half_patch_width,
                        :,
                    ]
                    rays_chunk = self._rays[
                        img_ind,
                        pix_loc[0],
                        pix_loc[1] - half_patch_width : pix_loc[1] + half_patch_width,
                        :,
                    ]
                elif half_patch_height != 0 and half_patch_width == 0:
                    pixel_chunk = self._images[
                        img_ind,
                        pix_loc[0] - half_patch_height : pix_loc[0] + half_patch_height,
                        pix_loc[1],
                        :,
                    ]
                    rays_chunk = self._rays[
                        img_ind,
                        pix_loc[0] - half_patch_height : pix_loc[0] + half_patch_height,
                        pix_loc[1],
                        :,
                    ]
                else:  # both are zeros:
                    pixel_chunk = self._images[img_ind, pix_loc[0], pix_loc[1], :]
                    rays_chunk = self._rays[img_ind, pix_loc[0], pix_loc[1], :]
                selected_pixels.append(pixel_chunk.reshape(-1, pixel_chunk.shape[-1]))
                selected_rays.append(rays_chunk.reshape(-1, rays_chunk.shape[-1]))
        # ============================================================================

        selected_pixels = torch.cat(selected_pixels, dim=0)
        selected_rays = torch.cat(selected_rays, dim=0)

        return selected_pixels, selected_rays
