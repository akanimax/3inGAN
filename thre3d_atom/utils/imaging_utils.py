import math
from typing import NamedTuple, Tuple, List, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS, ZERO_PLUS
from thre3d_atom.utils.misc import check_power_of_2


class CameraIntrinsics(NamedTuple):
    height: int
    width: int
    focal: float


class CameraPose(NamedTuple):
    rotation: np.array  # shape [3 x 3]
    translation: np.array  # shape [3 x 1]


class SceneBounds(NamedTuple):
    near: float
    far: float


def adjust_dynamic_range(
    data: np.array,
    drange_in: Tuple[float, float],
    drange_out: Tuple[float, float],
    slack: bool = False,
) -> np.array:
    """
    converts the data from the range `drange_in` into `drange_out`
    Args:
        data: input data array
        drange_in: data range [total_min_val, total_max_val]
        drange_out: output data range [min_val, max_val]
        slack: whether to cut some slack in range adjustment
    Returns: range_adjusted_data

    """
    if drange_in != drange_out:
        if not slack:
            old_min, old_max = np.float32(drange_in[0]), np.float32(drange_in[1])
            new_min, new_max = np.float32(drange_out[0]), np.float32(drange_out[1])
            data = (
                (data - old_min) / (old_max - old_min) * (new_max - new_min)
            ) + new_min
            data = data.clip(drange_out[0], drange_out[1])
        else:
            scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (
                np.float32(drange_in[1]) - np.float32(drange_in[0])
            )
            bias = np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale
            data = data * scale + bias
    return data


def patchify_images(images: Tensor, patch_size: Tuple[int, int]) -> Tensor:
    """ extracts all possible patches from the given images"""
    kernel_h, kernel_w = patch_size
    num_channels, stride = images.shape[1], 1
    return (
        images.unfold(2, kernel_h, stride)
        .unfold(3, kernel_w, stride)
        .permute(2, 3, 0, 1, 4, 5)
        .reshape(-1, num_channels, kernel_h, kernel_w)
    )


def get_2d_coordinates(
    height: int, width: int, drange: Tuple[float, float] = (-1.0, 1.0)
) -> Tensor:
    range_a, range_b = drange
    return torch.stack(
        torch.meshgrid(
            torch.linspace(range_a, range_b, height, dtype=torch.float32),
            torch.linspace(range_a, range_b, width, dtype=torch.float32),
        ),
        dim=-1,
    )


def to8b(x: np.array) -> np.array:
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def postprocess_disparity_map(
    disparity_map: np.array, scene_bounds: Optional[SceneBounds] = None
) -> np.array:
    drange_in = (
        (scene_bounds.near, scene_bounds.far)
        if scene_bounds is not None
        else (disparity_map.min(), disparity_map.max())
    )
    disparity_map = adjust_dynamic_range(
        disparity_map,
        drange_in=drange_in,
        drange_out=(0, 1),
    )
    colour_map = plt.get_cmap("turbo", lut=1024)
    return to8b(colour_map(disparity_map))[..., :NUM_COLOUR_CHANNELS]


def postprocess_depth_map(
    depth_map: np.array, scene_bounds: Optional[SceneBounds] = None
) -> np.array:
    drange_in = (
        (scene_bounds.near, scene_bounds.far)
        if scene_bounds is not None
        else (depth_map.min(), depth_map.max())
    )
    depth_map = 1 - adjust_dynamic_range(
        depth_map,
        drange_in=drange_in,
        drange_out=(0, 1),
    )
    colour_map = plt.get_cmap("gray", lut=1024)
    return to8b(colour_map(depth_map))[..., :NUM_COLOUR_CHANNELS]


def mse2psnr(x: Union[float, Tensor]) -> float:
    return -10.0 * math.log(x) / math.log(10.0) if x != 0.0 else math.inf


def mse2psnr_torch(x: Tensor) -> Tensor:
    return -10.0 * (torch.log(x + ZERO_PLUS) / math.log(10.0))


def scale_camera_intrinsics(
    camera_intrinsics: CameraIntrinsics, scale_factor: float = 1.0
) -> CameraIntrinsics:
    # note that height and width are integers while focal length is a float
    return CameraIntrinsics(
        height=int(np.ceil(camera_intrinsics.height * scale_factor)),
        width=int(np.ceil(camera_intrinsics.width * scale_factor)),
        focal=camera_intrinsics.focal * scale_factor,
    )


def downsample_camera_intrinsics(
    camera_intrinsics: CameraIntrinsics, downsample_factor: int = 1
) -> CameraIntrinsics:
    assert check_power_of_2(
        downsample_factor
    ), f"downsample_factor ({downsample_factor}) is not a power of 2 :("
    return CameraIntrinsics(
        height=camera_intrinsics.height // downsample_factor,
        width=camera_intrinsics.width // downsample_factor,
        focal=camera_intrinsics.focal // downsample_factor,
    )


def _translate_z(z: float, device=torch.device("cpu")) -> Tensor:
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )


def _rotate_theta(theta: float, device=torch.device("cpu")) -> Tensor:
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )


def _rotate_phi(phi: float, device=torch.device("cpu")) -> Tensor:
    return torch.tensor(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )


def pose_spherical(
    yaw: float, pitch: float, radius: float, device=torch.device("cpu")
) -> CameraPose:
    c2w = _translate_z(radius, device)
    c2w = _rotate_phi(pitch / 180.0 * np.pi, device) @ c2w
    c2w = _rotate_theta(yaw / 180.0 * np.pi, device) @ c2w
    c2w = (
        torch.tensor(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=torch.float32,
            device=device,
        )
        @ c2w
    )
    return CameraPose(rotation=c2w[:3, :3], translation=c2w[:3, 3:])


def get_thre360_animation_poses(
    hemispherical_radius: float, camera_pitch: float, num_poses: int
) -> List[CameraPose]:
    return [
        pose_spherical(yaw, pitch, hemispherical_radius)
        for (pitch, yaw) in zip(
            [camera_pitch] * (num_poses - 1),
            np.linspace(0, 360, num_poses)[:-1],
        )
    ]


# noinspection PyUnresolvedReferences
def get_rotating_animation_poses(
    hemispherical_radius: float, num_poses: int
) -> List[CameraPose]:
    return [
        pose_spherical(yaw, pitch, hemispherical_radius)
        for (pitch, yaw) in zip(
            (
                np.linspace(0, -90, num_poses // 2).tolist()
                + np.linspace(-90, 0, num_poses // 2).tolist()
            ),
            np.linspace(90, 450, num_poses)[:-1],
        )
    ]
