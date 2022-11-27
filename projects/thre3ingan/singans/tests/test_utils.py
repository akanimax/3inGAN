from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.utils.data as torch_data
from torch.backends import cudnn

from thre3d_atom.data.loaders import PosedImagesDataset
from projects.thre3ingan.singans.networks import get_default_render_mlp
from thre3d_atom.modules.volumetric_model.utils import (
    render_image_in_chunks,
)
from thre3d_atom.rendering.volumetric.implicit import cast_rays
from thre3d_atom.rendering.volumetric.utils import (
    shuffle_rays_and_pixels_synchronously,
    collate_rays,
)
from thre3d_atom.rendering.volumetric.voxels import (
    VoxelSize,
    FeatureGrid,
    render_feature_grid,
)
from thre3d_atom.training.losses import huber_loss
from thre3d_atom.utils.imaging_utils import (
    CameraPose,
    mse2psnr,
)
from thre3d_atom.utils.logging import log

cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_render_regular_feature_grid(data_path: Path) -> None:
    # mock objects needed for the rendering to work
    num_iterations = 2000
    feature_size, grid_dim = 32, 128
    num_samples_per_ray = 64
    feedback_step = 200
    num_rays_chunk = 512
    image_batch_cache_size = 16
    density_noise_std = 0.0
    dataset = PosedImagesDataset(
        images_dir=data_path / "images",
        camera_params_json=data_path / "camera_params.json",
        image_data_range=(0, 1),
        downsample_factor=4,
    )
    # setup the data_loader:
    train_dl = torch_data.DataLoader(
        dataset,
        batch_size=image_batch_cache_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    img, pose = dataset[0]

    cam_intrinsics = dataset.camera_intrinsics
    cam_pose = CameraPose(rotation=pose[:, :3], translation=pose[:, 3:])

    # create a feature_grid
    feature_grid = torch.randn(feature_size, grid_dim, grid_dim, grid_dim).to(device)
    depth, height, width = feature_grid.shape[1:]
    sizes = (dataset.scene_bounds.far - dataset.scene_bounds.near) / torch.tensor(
        [width, depth, height], dtype=torch.float32
    )
    voxel_size = VoxelSize(x_size=sizes[0], y_size=sizes[1], z_size=sizes[2])
    feature_grid = FeatureGrid(feature_grid, voxel_size, tunable=True)

    render_mlp = get_default_render_mlp(feature_size).to(device)
    optimizer = torch.optim.Adam(
        params=[
            {"params": feature_grid.parameters(), "lr": 0.1},
            {"params": render_mlp.parameters(), "lr": 0.001},
        ]
    )

    global_step = 0
    while global_step < num_iterations:
        for images, poses in train_dl:
            # cast rays for all images in the current batch:
            rays_list = []
            for pose in poses:
                casted_rays = cast_rays(
                    dataset.camera_intrinsics,
                    CameraPose(rotation=pose[:, :3], translation=pose[:, 3:]),
                )
                rays_list.append(casted_rays)
            rays = collate_rays(rays_list)

            # images are of shape [B x C x H x W] and pixels are [B * H * W x C]
            pixels = images.permute(0, 2, 3, 1).reshape(-1, images.shape[1])

            # shuffle rays and pixels synchronously
            rays, pixels = shuffle_rays_and_pixels_synchronously(rays, pixels)

            # select only num_rays_chunk number of rays from evey image
            selected_rays, selected_pixels = (
                rays[: num_rays_chunk * image_batch_cache_size],
                pixels[: num_rays_chunk * image_batch_cache_size],
            )

            for chunk_index in range(0, len(selected_rays.origins), num_rays_chunk):
                rendered_out = render_feature_grid(
                    selected_rays[chunk_index : chunk_index + num_rays_chunk].to(
                        device
                    ),
                    num_samples=num_samples_per_ray,
                    feature_grid=feature_grid,
                    point_processor_network=render_mlp,
                    scene_bounds=dataset.scene_bounds,
                    density_noise_std=density_noise_std,
                )

                loss = huber_loss(
                    rendered_out.colour,
                    selected_pixels[chunk_index : chunk_index + num_rays_chunk].to(
                        device
                    ),
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(
                    f"Step: {global_step} Current_loss: {loss.item()} Current_psnr: {mse2psnr(loss.item())}"
                )

                if global_step % feedback_step == 0 or global_step == 0:
                    log.info("render intermediate output")
                    rendered_output = render_image_in_chunks(
                        cam_intrinsics,
                        cam_pose,
                        num_rays_chunk,
                        num_samples_per_ray,
                        feature_grid,
                        render_mlp,
                        dataset.scene_bounds,
                        density_noise_std=density_noise_std,
                    )

                    plt.figure()
                    plt.title("Real image")
                    plt.imshow(img.permute(1, 2, 0))

                    plt.figure()
                    plt.title("Rendered image")
                    plt.imshow(rendered_output.colour.cpu())

                    plt.figure()
                    plt.title("Rendered disparity")
                    plt.imshow(rendered_output.disparity.cpu())
                    plt.show()

                global_step += 1
