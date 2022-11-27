import dataclasses
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Iterable, List

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as torch_data
from lpips import lpips
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomCrop
from torchvision.utils import make_grid
from tqdm import tqdm

from thre3d_atom.data.loaders import PosedImagesDataset
from thre3d_atom.data.utils import infinite_dataloader
from projects.thre3ingan.singans.networks import Thre3dGenerator
from thre3d_atom.modules.volumetric_model.utils import render_image_in_chunks
from thre3d_atom.rendering.volumetric.implicit import (
    cast_rays,
)
from thre3d_atom.rendering.volumetric.sample import sample_uniform_points_on_rays
from thre3d_atom.rendering.volumetric.render_interface import RenderOut, Rays
from thre3d_atom.rendering.volumetric.utils import (
    collate_rays,
    shuffle_rays_and_pixels_synchronously,
    reshape_and_rebuild_flat_rays,
    flatten_rays,
    collate_rendered_output,
)
from thre3d_atom.rendering.volumetric.voxels import (
    FeatureGrid,
    VoxelSize,
    get_voxel_size_from_scene_bounds_and_dataset,
    GridLocation,
)
from thre3d_atom.training.adversarial.losses import (
    StandardGanLoss,
    EmbeddingLatentGanLoss,
)
from thre3d_atom.training.adversarial.models import (
    get_convolutional_discriminator,
    Discriminator,
)
from thre3d_atom.training.losses import huber_loss
from thre3d_atom.utils.constants import (
    NUM_COORD_DIMENSIONS,
    NUM_COLOUR_CHANNELS,
)
from thre3d_atom.utils.imaging_utils import (
    SceneBounds,
    CameraIntrinsics,
    CameraPose,
    to8b,
    mse2psnr,
    adjust_dynamic_range,
    pose_spherical,
)
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import check_power_of_2, toggle_grad
from thre3d_atom.utils.visualization import visualize_camera_rays


@dataclasses.dataclass
class Thre3dSinGanRenderingParameters:
    num_rays_chunk: int = 2048
    num_samples_per_ray: int = 64
    perturb_sampled_points: bool = True
    density_noise_std: float = 1.0
    # use blender's (synthetic-scenes) voxel_size by default
    scene_bounds: SceneBounds = SceneBounds(near=2.0, far=6.0)
    voxel_size: VoxelSize = VoxelSize(x_size=0.03125, y_size=0.03125, z_size=0.03125)
    grid_location: GridLocation = GridLocation(x_coord=0.0, y_coord=0.0, z_coord=0.0)


class Thre3dSinGan:
    def __init__(
        self,
        thre3d_gen: Thre3dGenerator,
        render_params: Thre3dSinGanRenderingParameters,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self._device = device
        self._render_params = render_params
        self._generator = thre3d_gen.to(self._device)
        log.info(f"Generator Network configuration: {self._generator}")

        # shorthand:
        self._latent_size = self._generator.input_shape[1]
        self._input_noise_shape = (
            1,
            self._latent_size,
            *self._generator.feature_grid_shape_at_stage(1),
        )
        self._scale_factor = self._generator.scale_factor

    def _get_save_info(
        self, fixed_recon_noise: Tensor, discriminator: Optional[Discriminator] = None
    ) -> Dict[str, Any]:
        return {
            "thre3d_gen": self._generator.get_save_info(),
            "render_params": dataclasses.asdict(self._render_params),
            "discriminator": discriminator.get_save_info()
            if discriminator is not None
            else None,
            "fixed_recon_noise": fixed_recon_noise,
        }

    def _train_ray_chunk(
        self,
        rays: Rays,
        pixels: Tensor,
        optimizer: torch.optim.Optimizer,
        input_noise: Optional[Tensor] = None,
        stage: Optional[int] = None,
    ) -> Tuple[float, float]:
        # render the rays using the generator
        rendered_output = self._generator(
            rays.to(self._device),
            self._render_params.voxel_size,
            self._render_params.scene_bounds,
            self._render_params.num_samples_per_ray,
            input_noise,
            grid_location=self._render_params.grid_location,
            density_noise_std=self._render_params.density_noise_std,
            stage=stage,
        )

        # compute loss
        rendered_colour = rendered_output.colour
        loss = huber_loss(rendered_colour, pixels.to(self._device))

        # perform single step of optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # verbose and visual feedback
        loss_value = loss.item()
        psnr_value = mse2psnr(loss_value)

        return loss_value, psnr_value

    def _create_feature_grid_coords_visualization(
        self, dataset: PosedImagesDataset
    ) -> None:
        for stage in range(1, self._generator.num_stages + 1):
            voxel_size = get_voxel_size_from_scene_bounds_and_dataset(
                dataset,
                max(*self._generator.feature_grid_shape_at_stage(stage)),
                dataset.scene_bounds,
            )

            fig = plt.figure()
            fig.suptitle("feature-grid coordinates visualization")
            ax = fig.add_subplot(111, projection="3d")

            coords = self._generator.get_feature_grid(
                voxel_size,
                self._render_params.grid_location,
                stage=stage,
            ).get_all_feature_grid_coordinates()

            ax.scatter(coords[..., 0], coords[..., 1], coords[..., 2], color="m")
        plt.show()

    def _create_render_bounding_box_visualization(
        self,
        stage: int,
        render_pose: CameraPose,
        camera_intrinsics: CameraIntrinsics,
        output_dir: Path,
        input_noise: Optional[Tensor] = None,
    ) -> None:
        # obtain the feature grid
        feature_grid = self._generator.get_feature_grid(
            self._render_params.voxel_size,
            self._render_params.grid_location,
            input_noise,
            stage=stage,
        )
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
        rays = cast_rays(camera_intrinsics, render_pose)
        sampled_points = sample_uniform_points_on_rays(
            rays=rays,
            bounds=self._render_params.scene_bounds,
            num_samples=self._render_params.num_samples_per_ray,
            perturb=self._render_params.perturb_sampled_points,
            linear_disparity_sampling=False,
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

        # save the figure
        plt.tight_layout()
        plt.savefig(
            output_dir / f"bounding_box_and_render_rays_stage_{stage}.png", dpi=600
        )
        plt.close(fig)

        # free-up non-needed GPU memory:
        del feature_grid
        torch.cuda.empty_cache()

    def _render_recon_feedback(
        self,
        global_step: int,
        camera_intrinsics: CameraIntrinsics,
        render_pose: CameraPose,
        colour_logs_dir: Path,
        disparity_logs_dir: Path,
        stage: Optional[int] = None,
        verbose_rendering: bool = True,
        input_noise: Optional[Tensor] = None,
    ) -> None:
        # render images
        log.info(f"rendering intermediate reconstruction based output for feedback")

        rendered_output = self.render(
            camera_intrinsics,
            render_pose,
            scene_bounds=self._render_params.scene_bounds,
            voxel_size=self._render_params.voxel_size,
            grid_location=self._render_params.grid_location,
            input_noise=input_noise,
            stage=stage,
            verbose=verbose_rendering,
        )
        imageio.imwrite(
            colour_logs_dir / f"iter_{global_step}_colour.png",
            to8b(rendered_output.colour.numpy()),
        )
        imageio.imwrite(
            disparity_logs_dir / f"iter_{global_step}_disparity.png",
            to8b(rendered_output.disparity.squeeze().numpy()),
        )

    def _render_gan_feedback(
        self,
        global_step: int,
        camera_intrinsics: CameraIntrinsics,
        render_pose: CameraPose,
        colour_logs_dir: Path,
        disparity_logs_dir: Path,
        stage: Optional[int] = None,
        verbose_rendering: bool = True,
        input_noise: Tensor = None,
    ) -> None:
        # render images
        log.info(f"rendering intermediate GAN based output for feedback")
        num_images = input_noise.shape[0]
        rendered_colour_images_list = []
        rendered_disparity_images_list = []
        progress_bar = tqdm if verbose_rendering else lambda x: x
        for noise_vector in progress_bar(input_noise):
            rendered_output = self.render(
                camera_intrinsics,
                render_pose,
                scene_bounds=self._render_params.scene_bounds,
                voxel_size=self._render_params.voxel_size,
                grid_location=self._render_params.grid_location,
                input_noise=noise_vector[None, ...],
                stage=stage,
                verbose=False,  # no need to view individual progress bars
            )
            rendered_colour_images_list.append(rendered_output.colour)
            rendered_disparity_images_list.append(rendered_output.disparity)

        # fmt: off
        rendered_colour = torch.stack(rendered_colour_images_list, dim=0).permute(0, 3, 1, 2)
        rendered_disparity = torch.stack(rendered_disparity_images_list, dim=0).permute(0, 3, 1, 2)
        colour_image_grid = make_grid(rendered_colour, nrow=int(np.ceil(np.sqrt(num_images))), padding=0)
        disparity_image_grid = make_grid(rendered_disparity, nrow=int(np.ceil(np.sqrt(num_images))), padding=0)
        # fmt: on

        imageio.imwrite(
            colour_logs_dir / f"iter_{global_step}_colour.png",
            to8b(colour_image_grid.permute(1, 2, 0).numpy()),
        )
        # make_grid turns single channel images into RGB images. that's why the permute is needed below :D
        imageio.imwrite(
            disparity_logs_dir / f"iter_{global_step}_disparity.png",
            to8b(disparity_image_grid.permute(1, 2, 0).numpy()),
        )

    @staticmethod
    def _cast_rays_for_poses(
        poses: Tensor, camera_intrinsics: CameraIntrinsics
    ) -> List[Rays]:
        rays_list = []
        for pose in poses:
            casted_rays = cast_rays(
                camera_intrinsics,
                CameraPose(rotation=pose[:, :3], translation=pose[:, 3:]),
            )
            rays_list.append(casted_rays)
        return rays_list

    def _sample_random_ray_and_pixel_crops(
        self,
        images: Tensor,
        poses: Tensor,
        camera_intrinsics: CameraIntrinsics,
        random_patch_cropper: RandomCrop,
    ) -> Tuple[Rays, Tensor]:
        # cast rays for all poses:
        rays_list = self._cast_rays_for_poses(poses, camera_intrinsics)
        rays = reshape_and_rebuild_flat_rays(rays_list, camera_intrinsics)

        # create a unified image + ray tensor:
        image_ray_tensor = torch.cat(
            (images.permute(0, 2, 3, 1), rays.origins, rays.directions), dim=-1
        )
        patch_batch_tensor = random_patch_cropper(
            image_ray_tensor.permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        patch_images, patch_rays = (
            patch_batch_tensor[..., :NUM_COLOUR_CHANNELS],
            patch_batch_tensor[..., NUM_COLOUR_CHANNELS:],
        )
        patch_ray_origins, patch_ray_directions = (
            patch_rays[..., :NUM_COORD_DIMENSIONS],
            patch_rays[..., NUM_COORD_DIMENSIONS:],
        )

        return (
            Rays(
                origins=patch_ray_origins,
                directions=patch_ray_directions,
            ),
            patch_images.permute(0, 3, 1, 2),  # should be channels_first
        )

    def _sample_random_pixels_and_rays(
        self, images: Tensor, poses: Tensor, camera_intrinsics: CameraIntrinsics
    ) -> Tuple[Rays, Tensor]:
        # cast rays for all poses:
        rays_list = self._cast_rays_for_poses(poses, camera_intrinsics)
        rays = collate_rays(rays_list)

        # images are of shape [B x C x H x W] and pixels are [B * H * W x C]
        pixels = images.permute(0, 2, 3, 1).reshape(-1, images.shape[1])

        # shuffle rays and pixels synchronously
        rays, pixels = shuffle_rays_and_pixels_synchronously(rays, pixels)

        # select only num_rays_chunk number of rays from evey image
        chunk_size = self._render_params.num_rays_chunk
        batch_size = images.shape[0]
        selected_rays, selected_pixels = (
            rays[: chunk_size * batch_size],
            pixels[: chunk_size * batch_size],
        )

        return selected_rays, selected_pixels

    def _get_ray_crops_from_random_hemispherical_poses(
        self,
        batch_size: int,
        hemispherical_radius: float,
        camera_intrinsics: CameraIntrinsics,
        random_patch_cropper: RandomCrop,
    ) -> Rays:
        random_thetas = np.random.uniform(low=-180, high=180, size=batch_size)
        random_phis = np.random.uniform(low=-90, high=0, size=batch_size)
        random_hemispherical_poses = [
            pose_spherical(theta, phi, hemispherical_radius)
            for (theta, phi) in zip(random_thetas, random_phis)
        ]
        casted_rays_list = self._cast_rays_for_poses(
            torch.stack(
                [
                    torch.cat([rh_pose.rotation, rh_pose.translation], dim=-1)
                    for rh_pose in random_hemispherical_poses
                ]
            ),
            camera_intrinsics,
        )
        rays = reshape_and_rebuild_flat_rays(casted_rays_list, camera_intrinsics)
        ray_tensor = torch.cat([rays.origins, rays.directions], dim=-1).permute(
            0, 3, 1, 2
        )
        ray_crops = random_patch_cropper(ray_tensor).permute(0, 2, 3, 1)
        return Rays(
            origins=ray_crops[..., :NUM_COORD_DIMENSIONS],
            directions=ray_crops[..., NUM_COORD_DIMENSIONS:],
        )

    def _render_rays_in_chunks(
        self, flat_rays: Rays, stage: int, input_noise: Tensor
    ) -> RenderOut:
        chunk_size = self._render_params.num_rays_chunk
        rendered_output_list = []
        cached_feature_grid = self._generator.get_feature_grid(
            self._render_params.voxel_size,
            self._render_params.grid_location,
            input_noise,
            stage=stage,
        )
        for chunk_index in range(0, len(flat_rays.origins), chunk_size):
            rendered_chunk = self._generator(
                flat_rays[chunk_index : chunk_index + chunk_size].to(self._device),
                self._render_params.voxel_size,
                self._render_params.scene_bounds,
                self._render_params.num_samples_per_ray,
                input_noise=None,
                grid_location=self._render_params.grid_location,
                density_noise_std=self._render_params.density_noise_std,
                stage=stage,
                cached_feature_grid=cached_feature_grid,
            )
            rendered_output_list.append(rendered_chunk)
        return collate_rendered_output(rendered_output_list)

    @staticmethod
    def _reshape_rendered_output(
        rendered_output: RenderOut, new_shape: Tuple[int, ...]
    ) -> RenderOut:
        return RenderOut(
            colour=rendered_output.colour.reshape(*new_shape),
            disparity=rendered_output.disparity.reshape(*new_shape),
            extra={
                key: value.reshape(*new_shape)
                for key, value in rendered_output.extra.items()
            },
        )

    def _train_gan_steps(
        self,
        data_loader: Iterable[Tuple[Tensor, Tensor]],
        camera_intrinsics: CameraIntrinsics,
        discriminator: Discriminator,
        stage: int,
        optimizer_dis: torch.optim.Optimizer,
        optimizer_gen: torch.optim.Optimizer,
        random_patch_cropper: RandomCrop,
        hemispherical_radius: float,
        num_gen_steps: int = 1,
        num_dis_steps: int = 1,
    ) -> Tuple[float, float, float, float, float]:
        # ----------------------------------------------------------------------------------------------
        # Discriminator step
        # ----------------------------------------------------------------------------------------------
        toggle_grad(self._generator, False)
        toggle_grad(discriminator, True)

        images, poses = next(data_loader)
        batch_size = images.shape[0]
        _, pixel_crops = self._sample_random_ray_and_pixel_crops(
            images, poses, camera_intrinsics, random_patch_cropper
        )  # note that we ignore the rays here
        pixel_crops = pixel_crops.to(self._device)

        # randomly sample camera poses on the estimated hemisphere
        ray_crops = self._get_ray_crops_from_random_hemispherical_poses(
            images.shape[0],
            hemispherical_radius,
            camera_intrinsics,
            random_patch_cropper,
        )

        # render the random patches on the estimated hemisphere
        # note the use of sampled random noise below
        input_noise = torch.randn(*self._input_noise_shape).to(self._device)
        flat_rays = flatten_rays(ray_crops)
        rendered_output = self._render_rays_in_chunks(flat_rays, stage, input_noise)
        rendered_colour = self._reshape_rendered_output(
            rendered_output, (batch_size, *random_patch_cropper.size, -1)
        ).colour.permute(0, 3, 1, 2)

        # fmt: off
        dis_loss, extra_info = 0, None
        for _ in range(num_dis_steps):
            # compute the gan loss
            standard_gan_loss = StandardGanLoss()
            dis_loss, extra_info = standard_gan_loss.dis_loss(discriminator, pixel_crops, rendered_colour)

            # optimize discriminator
            optimizer_dis.zero_grad()
            dis_loss.backward()
            optimizer_dis.step()

        # obtain information for logging
        dis_loss_value = extra_info["discriminator_loss_value"].item()
        dis_real_scores, dis_fake_scores = extra_info["dis_real_scores"].item(), extra_info["dis_fake_scores"].item()
        discriminator_gradient_norm = extra_info["discriminator_real_gradient_norm"].item()
        # fmt: on

        # ----------------------------------------------------------------------------------------------
        # Generator step
        # ----------------------------------------------------------------------------------------------
        toggle_grad(self._generator, True)
        toggle_grad(discriminator, False)

        gen_loss = torch.tensor(0, dtype=torch.float32)
        for _ in range(num_gen_steps):
            # sample random input noise for the generator
            input_noise = torch.randn(*self._input_noise_shape).to(self._device)

            # randomly sample camera poses on the estimated hemisphere
            ray_crops = self._get_ray_crops_from_random_hemispherical_poses(
                batch_size,
                hemispherical_radius,
                camera_intrinsics,
                random_patch_cropper,
            )

            # render the random patches on the estimated hemisphere
            # note the use of sampled random noise below
            flat_rays = flatten_rays(ray_crops)
            rendered_output = self._render_rays_in_chunks(flat_rays, stage, input_noise)
            rendered_colour = self._reshape_rendered_output(
                rendered_output, (batch_size, *random_patch_cropper.size, -1)
            ).colour.permute(0, 3, 1, 2)

            # fmt: off
            standard_gan_loss = StandardGanLoss()
            gen_loss, _ = standard_gan_loss.gen_loss(discriminator, _, rendered_colour)

            optimizer_gen.zero_grad()
            gen_loss.backward()
            optimizer_gen.step()
            # fmt: on

        # obtain information for logging
        gen_loss_value = gen_loss.item()

        # ----------------------------------------------------------------------------------------------
        return (
            dis_loss_value,
            gen_loss_value,
            discriminator_gradient_norm,
            dis_real_scores,
            dis_fake_scores,
        )

    def _train_reconstruction_step(
        self,
        data_loader: Iterable[Tuple[Tensor, Tensor]],
        camera_intrinsics: CameraIntrinsics,
        fixed_recon_noise: Tensor,
        discriminator: Discriminator,
        optimizer: torch.optim.Optimizer,
        random_patch_cropper: RandomCrop,
        stage: int,
        use_gan_based_perceptual_loss: bool = False,
    ) -> Tuple[float, float, float]:
        """ Trains a single reconstruction step. The discriminator is needed for the perceptual loss """
        toggle_grad(self._generator, True)
        toggle_grad(discriminator, False)

        # ----------------------------------------------------------------------------------------------
        # photometric loss_optimization:
        # ----------------------------------------------------------------------------------------------
        images, poses = next(data_loader)
        rays, pixels = self._sample_random_pixels_and_rays(
            images, poses, camera_intrinsics
        )
        chunk_size = self._render_params.num_rays_chunk
        # note that we only train one single chunk per step
        loss_value, psnr_value = self._train_ray_chunk(
            rays[:chunk_size],
            pixels[:chunk_size],
            optimizer,
            fixed_recon_noise,
            stage=stage,
        )

        # ----------------------------------------------------------------------------------------------
        # gan-based perceptual loss optimization:
        # ----------------------------------------------------------------------------------------------
        gan_loss_value = 0.0
        if use_gan_based_perceptual_loss:
            images, poses = next(data_loader)
            batch_size = images.shape[0]
            ray_crops, pixel_crops = self._sample_random_ray_and_pixel_crops(
                images, poses, camera_intrinsics, random_patch_cropper
            )
            pixel_crops = pixel_crops.to(self._device)
            flat_rays = flatten_rays(ray_crops)
            rendered_output_list = []
            cached_feature_grid = self._generator.get_feature_grid(
                self._render_params.voxel_size,
                self._render_params.grid_location,
                fixed_recon_noise,
                stage=stage,
            )
            for chunk_index in range(0, len(flat_rays.origins), chunk_size):
                rendered_chunk = self._generator(
                    flat_rays[chunk_index : chunk_index + chunk_size].to(self._device),
                    self._render_params.voxel_size,
                    self._render_params.scene_bounds,
                    self._render_params.num_samples_per_ray,
                    grid_location=self._render_params.grid_location,
                    density_noise_std=self._render_params.density_noise_std,
                    depth=stage,
                    cached_feature_grid=cached_feature_grid,
                )
                rendered_output_list.append(rendered_chunk)
            rendered_output = collate_rendered_output(rendered_output_list)
            new_shape = (batch_size, *random_patch_cropper.size, -1)
            rendered_output = RenderOut(
                colour=rendered_output.colour.reshape(*new_shape),
                disparity=rendered_output.disparity.reshape(*new_shape),
                extra={
                    key: value.reshape(*new_shape)
                    for key, value in rendered_output.extra.items()
                },
            )
            rendered_colour = rendered_output.colour.permute(0, 3, 1, 2)

            # fmt: off
            standard_gan_loss, embedding_gan_loss = StandardGanLoss(), EmbeddingLatentGanLoss()
            gan_loss1, _ = standard_gan_loss.gen_loss(discriminator, pixel_crops, rendered_colour)
            gan_loss2, _ = embedding_gan_loss.gen_loss(discriminator, pixel_crops, rendered_colour)
            gan_loss = gan_loss1 + gan_loss2
            gan_loss_value = gan_loss.item()

            optimizer.zero_grad()
            gan_loss.backward()
            optimizer.step()
            # fmt: on

        return loss_value, gan_loss_value, psnr_value

    # noinspection PyProtectedMember
    def render(
        self,
        camera_intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
        scene_bounds: SceneBounds,
        voxel_size: VoxelSize,
        grid_location: GridLocation,
        input_noise: Optional[Tensor] = None,
        # following parameter allows for caching
        feature_grid: Optional[FeatureGrid] = None,
        stage: Optional[int] = None,
        verbose: bool = True,
    ) -> RenderOut:
        if feature_grid is None:
            feature_grid = self._generator.get_feature_grid(
                voxel_size=voxel_size,
                grid_location=grid_location,
                input_noise=input_noise,
                stage=stage,
            )
        # renders the image in small chunks based on the num_rays_chunk
        return render_image_in_chunks(
            cam_intrinsics=camera_intrinsics,
            camera_pose=camera_pose,
            num_rays_chunk=self._render_params.num_rays_chunk,
            num_samples_per_ray=self._render_params.num_samples_per_ray,
            feature_grid=feature_grid,
            processor_mlp=self._generator.renderer,
            scene_bounds=scene_bounds,
            density_noise_std=self._render_params.density_noise_std,
            use_dists_in_rendering=self._generator._use_dists_in_rendering,
            device=self._device,
            verbose=verbose,
        )

    def test_reconstruction(
        self,
        fixed_recon_noise: Tensor,
        test_dl: torch_data.DataLoader,
        camera_intrinsics: CameraIntrinsics,
        global_step: Optional[int] = 0,
        tensorboard_writer: Optional[SummaryWriter] = None,
        stage: Optional[int] = None,
    ) -> None:
        log.info(
            f"Testing the model's reconstruction performance on {len(test_dl.dataset)} heldout images"
        )
        assert (
            test_dl.batch_size == 1
        ), f"Cannot test with a batch_size ({test_dl.batch_size}) > 1"
        all_psnrs, all_lpips = [], []

        # cache the feature grid used for reconstruction task
        feature_grid = self._generator.get_feature_grid(
            voxel_size=self._render_params.voxel_size,
            grid_location=self._render_params.grid_location,
            input_noise=fixed_recon_noise,
            stage=stage,
        )

        vgg_lpips_computer = lpips.LPIPS(net="vgg")
        for (image, pose) in tqdm(test_dl):
            image, pose = image[0], pose[0]  # testing batching is always 1
            rendered_output = self.render(
                camera_intrinsics=camera_intrinsics,
                camera_pose=CameraPose(rotation=pose[:, :3], translation=pose[:, 3:]),
                scene_bounds=self._render_params.scene_bounds,
                voxel_size=self._render_params.voxel_size,
                grid_location=self._render_params.grid_location,
                feature_grid=feature_grid,
                stage=stage,
                verbose=False,
            )
            rendered_colour = rendered_output.colour.permute(2, 0, 1)

            # all testing is done in no-gradients mode
            with torch.no_grad():
                # compute the PSNR metric:
                psnr = mse2psnr(mse_loss(rendered_colour, image))

                # compute the LPIPS metric
                vgg_lpips = vgg_lpips_computer(
                    adjust_dynamic_range(rendered_colour[None, ...], (0, 1), (-1, 1)),
                    adjust_dynamic_range(image[None, ...], (0, 1), (-1, 1)),
                )

            all_psnrs.append(psnr)
            all_lpips.append(vgg_lpips)
        mean_psnr, mean_lpips = [
            np.mean(metric_scores) for metric_scores in (all_psnrs, all_lpips)
        ]
        log.info(f"Mean PSNR on holdout set: {mean_psnr}")
        log.info(f"Mean LPIPS on holdout set: {mean_lpips}")
        if tensorboard_writer is not None:
            for metric_tag, metric_value in [
                ("TEST_SET_PSNR", mean_psnr),
                ("TEST_SET_LPIPS", mean_lpips),
            ]:
                tensorboard_writer.add_scalar(
                    metric_tag, metric_value, global_step=global_step
                )

        # delete the vgg_lpips computer from memory for saving up memory:
        del vgg_lpips_computer

    def train_singan(
        self,
        dataset_dir: Path,
        original_dataset_downsample_factor: int = 1,
        num_iterations_per_stage: int = 3e5,
        num_feedback_images: int = 6,
        patch_size: int = 32,
        image_batch_cache_size: int = 3,
        save_freq: int = 1000,
        feedback_freq: int = 1000,
        test_freq: int = 1000,
        gen_learning_rate: float = 0.001,
        gen_renderer_learning_rate: float = 0.001,
        dis_learning_rate: float = 0.001,
        lr_decay_steps: int = 100,
        output_dir: Path = Path(__file__).parent.absolute() / "logs",
        render_feedback_pose: Optional[CameraPose] = None,
        loss_feedback_freq: Optional[int] = None,
        verbose_rendering: bool = True,
        num_workers: int = 4,
        scene_bounds: Optional[SceneBounds] = None,
        voxel_size: Optional[VoxelSize] = None,
        grid_location: Optional[GridLocation] = None,
        num_gan_steps: int = 1,
        num_recon_steps: int = 1,
        use_gan_based_perceptual_recon_loss: bool = False,
    ) -> None:
        total_stages = self._generator.num_stages
        global_step = 0
        fixed_reconstruction_noise, fixed_feedback_noise = None, None

        for stage in range(1, total_stages + 1):
            # construct the training dataset for the current stage
            current_stage_downsample_factor = original_dataset_downsample_factor * (
                self._scale_factor ** (total_stages - stage)
            )
            train_dataset = PosedImagesDataset(
                images_dir=dataset_dir / "images",
                camera_params_json=dataset_dir / "camera_params.json",
                test_percentage=10.0,
                test_mode=False,
                image_data_range=(0, 1),
                downsample_factor=current_stage_downsample_factor,
            )
            test_dataset = PosedImagesDataset(
                images_dir=dataset_dir / "images",
                camera_params_json=dataset_dir / "camera_params.json",
                test_percentage=10.0,
                test_mode=True,
                image_data_range=(0, 1),
                downsample_factor=current_stage_downsample_factor,
            )

            # train the current stage with current dataset:
            (
                fixed_reconstruction_noise,
                fixed_feedback_noise,
                global_step,
            ) = self.train_single_adversarial_stage(
                stage=stage,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                global_step=global_step,
                fixed_reconstruction_noise=fixed_reconstruction_noise,
                fixed_feedback_noise=fixed_feedback_noise,
                num_feedback_images=num_feedback_images,
                patch_size=patch_size,
                image_batch_cache_size=image_batch_cache_size,
                num_iterations=num_iterations_per_stage,
                save_freq=save_freq,
                feedback_freq=feedback_freq,
                test_freq=test_freq,
                gen_learning_rate=gen_learning_rate,
                gen_renderer_learning_rate=gen_renderer_learning_rate,
                dis_learning_rate=dis_learning_rate,
                lr_decay_steps=lr_decay_steps,
                output_dir=output_dir,
                render_feedback_pose=render_feedback_pose,
                loss_feedback_freq=loss_feedback_freq,
                verbose_rendering=verbose_rendering,
                num_workers=num_workers,
                scene_bounds=scene_bounds,
                voxel_size=voxel_size,
                grid_location=grid_location,
                num_gan_steps=num_gan_steps,
                num_recon_steps=num_recon_steps,
                use_gan_based_perceptual_recon_loss=use_gan_based_perceptual_recon_loss,
            )

    def train_single_adversarial_stage(
        self,
        stage: int,
        train_dataset: PosedImagesDataset,
        test_dataset: PosedImagesDataset,
        global_step: int = 0,
        fixed_reconstruction_noise: Optional[Tensor] = None,
        fixed_feedback_noise: Optional[Tensor] = None,
        num_feedback_images: int = 6,
        patch_size: int = 32,
        image_batch_cache_size: int = 3,
        num_iterations: int = 3e5,
        save_freq: int = 1000,
        feedback_freq: int = 1000,
        test_freq: int = 1000,
        gen_learning_rate: float = 0.001,
        gen_renderer_learning_rate: float = 0.001,
        dis_learning_rate: float = 0.001,
        lr_decay_steps: int = 100,
        output_dir: Path = Path(__file__).parent.absolute() / "logs",
        render_feedback_pose: Optional[CameraPose] = None,
        loss_feedback_freq: Optional[int] = None,
        verbose_rendering: bool = True,
        num_workers: int = 4,
        scene_bounds: Optional[SceneBounds] = None,
        voxel_size: Optional[VoxelSize] = None,
        grid_location: Optional[GridLocation] = None,
        num_gan_steps: int = 1,
        num_recon_steps: int = 1,
        use_gan_based_perceptual_recon_loss: bool = False,
    ) -> Tuple[Tensor, Tensor, int]:
        """
        Trains a Thre3dSinGAN model given a dataset (of a single scene :))
        Args:
            stage: Current stage of training the model (corresponds to the model_size and scale)
            train_dataset: object of PosedImageDataset used for training the model
            test_dataset: object of PosedImageDataset used for testing the model
                          (Note you can use the `test_mode` while creating the PosedImageDataset object)
            global_step: global_step of training (cumulative of possible previous stages)
            fixed_reconstruction_noise: noise vector corresponding to the reconstruction module
                                        (can be None only if stage == 1)
            fixed_feedback_noise: noise vectors for generating visual feedback during training
                                  (can be None only if stage == 1)
            num_feedback_images: number of images rendered for GAN based feedback
            patch_size: size of patches (rendered) fed to the discriminator
            image_batch_cache_size: Number of images to be cached in queue from the disk
            num_iterations: total number of iterations to train the model for
            save_freq: save model after these many fine network iterations
            feedback_freq: log feedback after these many iterations
            test_freq: perform reconstruction performance testing after these many iterations
            gen_learning_rate: generator's learning rate for the optimization process
            gen_renderer_learning_rate: learning rate for the mlp used for rendering
            dis_learning_rate: learning rate for the discriminator
            lr_decay_steps: The learning rate decays by 0.1 (gamma) exponentially
                            after these many steps
            output_dir: write the training assets to this directory
            render_feedback_pose: pose for rendering intermediate training feedback images
            loss_feedback_freq: frequency of logging the loss values to the console
            verbose_rendering: get verbose feedback while rendering the model
            num_workers: number of processes used for loading the data
            scene_bounds: overridden scene bounds
            voxel_size: overridden voxel size of the Feature Grid
            grid_location: overridden grid location. Denotes the center of the feature grid
            num_gan_steps: number of gan steps taken per training iteration
            num_recon_steps: number of reconstruction steps taken per training iteration
            use_gan_based_perceptual_recon_loss: whether to use gan based embedding loss for reconstruction
        Returns: Fixed reconstruction noise, also writes multiple files to the disk.
        """
        # assertions:
        # fmt: off
        assert check_power_of_2(patch_size), f"patch_size {patch_size} must be a power of 2"
        assert 1 <= stage <= self._generator.num_stages, f"requested training stage ({stage}) should be in range" \
                                                         f"({(1, self._generator.num_stages)})"
        # fmt: on

        # setup render_feedback_pose
        real_feedback_image = None
        if render_feedback_pose is None:
            render_feedback_pose = CameraPose(
                rotation=test_dataset[0][-1][:, :3].numpy(),
                translation=test_dataset[0][-1][:, 3:].numpy(),
            )
            real_feedback_image = test_dataset[0][0].permute(1, 2, 0).numpy()

        # loss feedback frequency:
        loss_feedback_freq = (
            feedback_freq // 10 if loss_feedback_freq is None else loss_feedback_freq
        )

        # setup the data_loader:
        train_dl = torch_data.DataLoader(
            train_dataset,
            batch_size=image_batch_cache_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        test_dl = torch_data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # dataset size aka number of total pixels
        dataset_size = (
            len(train_dl)
            * train_dataset.camera_intrinsics.height
            * train_dataset.camera_intrinsics.width
        )

        # setup the fixed input noise for performing the reconstruction task
        # And, the fixed noise for generating visual training feedback
        if stage > 1:
            assert (
                fixed_reconstruction_noise is not None
            ), f"Reconstruction noise cannot be None for stage ({stage}) > 1"
            assert (
                fixed_feedback_noise is not None
            ), f"Feedback noise cannot be None for stage ({stage}) > 1"
        else:
            # if this is the first stage, then randomly sample a fixed_reconstruction_noise
            # and a fixed noise vector for generating feedback images
            # fmt: off
            fixed_reconstruction_noise = torch.randn(*self._input_noise_shape).to(self._device)
            fixed_feedback_noise = torch.randn(
                num_feedback_images, *self._input_noise_shape[1:]
            ).to(self._device)
            # fmt: on

        # create a new discriminator for this stage:
        # fmt: off
        discriminator = get_convolutional_discriminator(int(np.log2(patch_size)), latent_size=512,
                                                        fmap_min=128, fmap_max=512, fmap_base=2048,).to(self._device)
        log.info(f"Created new Discriminator: {discriminator}")

        # setup optimizers
        # Thre3d Generator's interface assumes correspondence between depth and resolution
        optimizer_gen = torch.optim.Adam(
            params=[{"params": self._generator.thre3d_generator(stage).parameters(), "lr": gen_learning_rate},
                    {"params": self._generator.renderer.parameters(), "lr": gen_renderer_learning_rate}],
            betas=(0, 0.99),
        )
        optimizer_dis = torch.optim.Adam(params=discriminator.parameters(), lr=dis_learning_rate, betas=(0, 0.99))

        # setup learning rate schedulers for the two optimizers
        lr_schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizer_gen, gamma=0.1),
                         torch.optim.lr_scheduler.ExponentialLR(optimizer_dis, gamma=0.1)]

        # setup output directories
        model_dir, logs_dir = output_dir / "saved_models", output_dir / "training_logs"
        tensorboard_dir, render_dir = logs_dir / "tensorboard", logs_dir / "rendered_output"
        recon_colour_logs_dir = render_dir / "recon" / str(stage) / "colour"
        recon_disparity_logs_dir = render_dir / "recon" / str(stage) / "disparity"
        gan_colour_logs_dir = render_dir / "gan" / str(stage) / "colour"
        gan_disparity_logs_dir = render_dir / "gan" / str(stage) / "disparity"
        for directory in (model_dir, logs_dir, tensorboard_dir, render_dir,
                          recon_colour_logs_dir, recon_disparity_logs_dir,
                          gan_colour_logs_dir, gan_disparity_logs_dir):
            directory.mkdir(exist_ok=True, parents=True)
        # fmt: on

        # save the real_feedback_test_image if it exists:
        if real_feedback_image is not None:
            log.info(f"Logging real feedback image")
            imageio.imwrite(
                recon_colour_logs_dir / f"1__real_log.png",
                to8b(real_feedback_image),
            )

        # setup the feature-grid related render_params.
        # note that we attach these to the object's state, because these are crucial
        # pieces of information for training and rendering in general
        self._render_params.grid_location = grid_location
        self._render_params.scene_bounds = (
            train_dataset.scene_bounds if scene_bounds is None else scene_bounds
        )
        self._render_params.voxel_size = (
            get_voxel_size_from_scene_bounds_and_dataset(
                train_dataset,
                grid_dim=max(self._generator.feature_grid_shape_at_stage(stage)),
                scene_bounds=self._render_params.scene_bounds,
            )
            if voxel_size is None
            else voxel_size
        )
        # use the provided voxel_size if not None, else we compute the voxel_size based on the
        # FeatureGrid (current stage) size and the scene-bounds. The use of scene_bounds is a looser estimate of
        # the volume. And, a tighter bound is preferred.

        # setup tensorboard writer
        tensorboard_writer = SummaryWriter(tensorboard_dir)

        # create visualizations for better visual feedback:
        if stage == 1:
            visualize_camera_rays(
                test_dataset,
                output_dir,
                num_rays_per_image=10,
            )
        self._create_render_bounding_box_visualization(
            stage,
            render_feedback_pose,
            test_dataset.camera_intrinsics,
            output_dir,
            fixed_reconstruction_noise,
        )

        # obtain an estimate of the hemispherical radius from the dataset
        hemispherical_radius = train_dataset.get_hemispherical_radius_estimate()

        # start actual training
        log.info(f"Beginning training stage: {stage}")
        log.info(f"Current stage camera-intrinsics: {train_dataset.camera_intrinsics}")
        train_dl_infinte = infinite_dataloader(train_dl)
        random_patch_cropper = RandomCrop(patch_size)
        for stage_step in range(num_iterations):
            dis_loss_value, gen_loss_value, disc_grad_norm = 0.0, 0.0, 0.0
            dis_real_scores, dis_fake_scores = 0.0, 0.0
            # perform GAN training steps (discriminator and generator)
            for _ in range(num_gan_steps):
                (
                    dis_loss_value,
                    gen_loss_value,
                    disc_grad_norm,
                    dis_real_scores,
                    dis_fake_scores,
                ) = self._train_gan_steps(
                    data_loader=train_dl_infinte,
                    camera_intrinsics=train_dataset.camera_intrinsics,
                    discriminator=discriminator,
                    stage=stage,
                    optimizer_dis=optimizer_dis,
                    optimizer_gen=optimizer_gen,
                    random_patch_cropper=random_patch_cropper,
                    hemispherical_radius=hemispherical_radius,
                )

            # perform reconstruction training step
            # fmt: off
            recon_loss_value, recon_gan_loss_value, recon_psnr = 0.0, 0.0, 0.0
            for _ in range(num_recon_steps):
                recon_loss_value, recon_gan_loss_value, recon_psnr = self._train_reconstruction_step(
                    data_loader=train_dl_infinte,
                    camera_intrinsics=train_dataset.camera_intrinsics,
                    fixed_recon_noise=fixed_reconstruction_noise,
                    discriminator=discriminator,
                    optimizer=optimizer_gen,
                    random_patch_cropper=random_patch_cropper,
                    stage=stage,
                    use_gan_based_perceptual_loss=use_gan_based_perceptual_recon_loss,
                )
            # fmt: on

            # tensorboard summaries feedback
            # fmt: off
            for summary_name, summary_value in (
                ("dis_loss", dis_loss_value), ("gen_loss", gen_loss_value), ("dis_grad_norm", disc_grad_norm),
                ("dis_real_scores", dis_real_scores), ("dis_fake_scores", dis_fake_scores),
                ("recon_loss", recon_loss_value), ("recon_perceptual_loss", recon_gan_loss_value),
                ("psnr", recon_psnr), ("num_epochs", global_step / dataset_size),
            ):
                if summary_value is not None:
                    tensorboard_writer.add_scalar(
                        summary_name, summary_value, global_step=global_step
                    )
            # fmt: on

            # step the learning rate schedulers
            # fmt: off
            if (global_step + 1) % lr_decay_steps == 0:
                for lr_scheduler in lr_schedulers:
                    lr_scheduler.step()
                new_gen_lrs = [param_group["lr"] for param_group in optimizer_gen.param_groups]
                new_dis_lrs = [param_group["lr"] for param_group in optimizer_dis.param_groups]
                log_string = f"Adjusted learning rate | generator: {new_gen_lrs} "
                log_string += f"Adjusted learning rate | discriminator: {new_dis_lrs} "
                log.info(log_string)
            # fmt: on

            # console loss feedback
            if (
                global_step % loss_feedback_freq == 0
                or global_step == 0
                or stage_step == (num_iterations - 1)
            ):
                loss_info_string = (
                    f"Iteration: {global_step} "
                    f"dis_loss: {dis_loss_value: .5f} gen_loss: {gen_loss_value: .5f} "
                    f"dis_real_scores: {dis_real_scores: .5f} dis_fake_scores: {dis_fake_scores: .5f} "
                    f"recon_loss: {recon_loss_value: .5f} recon_perceptual_loss: {recon_gan_loss_value: .5f} "
                    f"psnr: {recon_psnr: .5f}"
                )
                log.info(loss_info_string)

            # rendered feedback
            if (
                global_step % feedback_freq == 0
                or (global_step < 100 and global_step % 10 == 0)
                or stage_step == (num_iterations - 1)
            ):
                # reconstruction based rendered (visual) feedback
                self._render_recon_feedback(
                    global_step,
                    train_dataset.camera_intrinsics,
                    render_feedback_pose,
                    recon_colour_logs_dir,
                    recon_disparity_logs_dir,
                    stage=stage,
                    verbose_rendering=verbose_rendering,
                    input_noise=fixed_reconstruction_noise,
                )
                # gan based rendered (visual) feedback
                self._render_gan_feedback(
                    global_step,
                    train_dataset.camera_intrinsics,
                    render_feedback_pose,
                    gan_colour_logs_dir,
                    gan_disparity_logs_dir,
                    stage=stage,
                    verbose_rendering=verbose_rendering,
                    input_noise=fixed_feedback_noise,
                )

            # reconstruction test performance
            if global_step % test_freq == 0 or (global_step == 0):
                # Obtain the test metrics on the reconstruction performance
                self.test_reconstruction(
                    fixed_recon_noise=fixed_reconstruction_noise,
                    test_dl=test_dl,
                    camera_intrinsics=test_dataset.camera_intrinsics,
                    global_step=global_step,
                    tensorboard_writer=tensorboard_writer,
                    stage=stage,
                )

            # save the model
            if (
                global_step % save_freq == 0
                or global_step == 0
                or stage_step == (num_iterations - 1)
            ):
                torch.save(
                    self._get_save_info(
                        fixed_recon_noise=fixed_reconstruction_noise,
                        discriminator=discriminator,
                    ),
                    model_dir / f"model_iter_{global_step}_stage_{stage}.pth",
                )

            global_step += 1

        # save the final trained model
        torch.save(
            self._get_save_info(
                fixed_recon_noise=fixed_reconstruction_noise,
                discriminator=discriminator,
            ),
            model_dir / f"model_final_stage_{stage}.pth",
        )

        # training complete yay! :)
        log.info(f"Training complete for stage: {stage}")

        return fixed_reconstruction_noise, fixed_feedback_noise, global_step

    def train_reconstruction(
        self,
        train_dataset: PosedImagesDataset,
        test_dataset: PosedImagesDataset,
        image_batch_cache_size: int = 3,
        num_iterations: int = 3e5,
        save_freq: int = 1000,
        feedback_freq: int = 1000,
        test_freq: int = 1000,
        gen_learning_rate: float = 0.001,
        gen_renderer_learning_rate: float = 0.001,
        lr_decay_steps: int = 100,
        output_dir: Path = Path(__file__).parent.absolute() / "logs",
        render_feedback_pose: Optional[CameraPose] = None,
        loss_feedback_freq: Optional[int] = None,
        verbose_rendering: bool = True,
        num_workers: int = 4,
        scene_bounds: Optional[SceneBounds] = None,
        voxel_size: Optional[VoxelSize] = None,
        grid_location: Optional[GridLocation] = GridLocation(),
    ) -> None:
        """
        Trains a Thre3dSinGAN model given a dataset (of a single scene :))
        Args:
            train_dataset: object of PosedImageDataset used for training the model
            test_dataset: object of PosedImageDataset used for testing the model
                          (Note you can use the `test_mode` while creating the PosedImageDataset object)
            image_batch_cache_size: Number of images to be cached in queue from the disk
            num_iterations: total number of iterations to train the model for
            save_freq: save model after these many fine network iterations
            feedback_freq: log feedback after these many iterations
            test_freq: perform reconstruction performance testing after these many iterations
            gen_learning_rate: generator's learning rate for the optimization process
            gen_renderer_learning_rate: learning rate for the mlp used for rendering
            lr_decay_steps: The learning rate decays by 0.1 (gamma) exponentially
                            after these many steps
            output_dir: write the training assets to this directory
            render_feedback_pose: pose for rendering intermediate training feedback images
            loss_feedback_freq: frequency of logging the loss values to the console
            verbose_rendering: get verbose feedback while rendering the model
            num_workers: number of processes used for loading the data
            scene_bounds: overridden scene bounds
            voxel_size: overridden voxel size of the Feature Grid
            grid_location: overridden grid location. Denotes the center of the feature grid
        Returns: None, writes multiple files to the disk.
        """
        # setup render_feedback_pose
        real_test_feedback_image = None
        if render_feedback_pose is None:
            render_feedback_pose = CameraPose(
                rotation=test_dataset[0][-1][:, :3].numpy(),
                translation=test_dataset[0][-1][:, 3:].numpy(),
            )
            real_test_feedback_image = test_dataset[0][0].permute(1, 2, 0).numpy()

        # loss feedback frequency:
        loss_feedback_freq = (
            feedback_freq // 10 if loss_feedback_freq is None else loss_feedback_freq
        )

        # setup the data_loader:
        train_dl = torch_data.DataLoader(
            train_dataset,
            batch_size=image_batch_cache_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        test_dl = torch_data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # dataset size aka number of total pixels
        dataset_size = (
            len(train_dl)
            * train_dataset.camera_intrinsics.height
            * train_dataset.camera_intrinsics.width
        )

        # obtain a fixed input noise for performing the reconstruction task
        fixed_recon_noise = torch.randn(*self._input_noise_shape).to(self._device)

        # setup optimizers
        # fmt: off
        optimizer_gen = torch.optim.Adam(
            params=[
                {"params": self._generator.thre3d_generator().parameters(),
                 "lr": gen_learning_rate},
                {"params": self._generator.renderer.parameters(),
                 "lr": gen_renderer_learning_rate}
            ],
        )
        # fmt: on

        # setup learning rate schedulers for the two optimizers
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer_gen, gamma=0.1)

        # setup output directories
        model_dir = output_dir / "saved_models"
        logs_dir = output_dir / "training_logs"
        tensorboard_dir = logs_dir / "tensorboard"
        render_dir = logs_dir / "rendered_output"
        colour_logs_dir = render_dir / "colour"
        disparity_logs_dir = render_dir / "disparity"
        for directory in (
            model_dir,
            logs_dir,
            tensorboard_dir,
            render_dir,
            colour_logs_dir,
            disparity_logs_dir,
        ):
            directory.mkdir(exist_ok=True, parents=True)

        # save the real_feedback_test_image if it exists:
        if real_test_feedback_image is not None:
            log.info(f"Logging real feedback image ...")
            imageio.imwrite(
                colour_logs_dir / f"1__real_log.png",
                to8b(real_test_feedback_image),
            )

        # setup the feature-grid related render_params.
        # note that we attach the scene_bounds and voxel_size to the object's state, because it's a crucial
        # piece of information for training and rendering in general
        self._render_params.grid_location = grid_location
        self._render_params.scene_bounds = (
            scene_bounds if scene_bounds is not None else train_dataset.scene_bounds
        )
        self._render_params.voxel_size = (
            voxel_size
            if voxel_size is not None
            else get_voxel_size_from_scene_bounds_and_dataset(
                train_dataset,
                grid_dim=max(*self._generator.output_shape[2:]),
                scene_bounds=self._render_params.scene_bounds,
            )
        )
        # use the provided voxel_size if not None, else we compute the voxel_size based on the
        # FeatureGrid size and the scene-bounds. The use of scene_bounds is a looser estimate of
        # the volume. And, a tighter bound is preferred.
        # setup tensorboard writer
        tensorboard_writer = SummaryWriter(tensorboard_dir)

        # create a visualization of the feature-grid bounding_box and render_rays
        visualize_camera_rays(
            test_dataset,
            output_dir,
            num_rays_per_image=10,
        )
        self._create_render_bounding_box_visualization(
            self._generator.num_stages,
            render_feedback_pose,
            test_dataset.camera_intrinsics,
            output_dir,
            fixed_recon_noise,
        )

        # start actual training
        log.info("Beginning training")
        global_step = 0
        while global_step < num_iterations:
            for images, poses in train_dl:
                chunk_size = self._render_params.num_rays_chunk
                selected_rays, selected_pixels = self._sample_random_pixels_and_rays(
                    images, poses, train_dataset.camera_intrinsics
                )

                for chunk_index in range(0, len(selected_rays.origins), chunk_size):
                    loss_value, psnr_value = self._train_ray_chunk(
                        selected_rays[chunk_index : chunk_index + chunk_size],
                        selected_pixels[chunk_index : chunk_index + chunk_size],
                        optimizer_gen,
                        input_noise=fixed_recon_noise,
                    )

                    # tensorboard summaries feedback
                    for summary_name, summary_value in (
                        ("loss value", loss_value),
                        ("psnr value", psnr_value),
                        ("num_epochs", global_step / dataset_size),
                    ):
                        if summary_value is not None:
                            tensorboard_writer.add_scalar(
                                summary_name, summary_value, global_step=global_step
                            )

                    # step the learning rate schedulers
                    if (global_step + 1) % lr_decay_steps == 0:
                        lr_scheduler.step()
                        new_lrs = [
                            param_group["lr"]
                            for param_group in optimizer_gen.param_groups
                        ]
                        log_string = f"Adjusted learning rate | coarse_lr: {new_lrs} "
                        log.info(log_string)

                    # console loss feedback
                    if global_step % loss_feedback_freq == 0 or global_step == 0:
                        loss_info_string = (
                            f"Iteration: {global_step} "
                            f"Loss_coarse: {loss_value: .5f} "
                            f"PSNR_coarse: {psnr_value: .5f} "
                        )
                        log.info(loss_info_string)

                    # rendered feedback
                    if global_step % feedback_freq == 0 or (
                        global_step < 100 and global_step % 10 == 0
                    ):
                        # once the rendering is complete
                        self._render_recon_feedback(
                            global_step,
                            train_dataset.camera_intrinsics,
                            render_feedback_pose,
                            colour_logs_dir,
                            disparity_logs_dir,
                            input_noise=fixed_recon_noise,
                            verbose_rendering=verbose_rendering,
                        )

                    # reconstruction test performance
                    if global_step % test_freq == 0 or (global_step == 0):
                        # Obtain the test metrics on the reconstruction performance
                        self.test_reconstruction(
                            fixed_recon_noise=fixed_recon_noise,
                            test_dl=test_dl,
                            camera_intrinsics=test_dataset.camera_intrinsics,
                            global_step=global_step,
                            tensorboard_writer=tensorboard_writer,
                        )

                    # save the model
                    if global_step % save_freq == 0 or global_step == 0:
                        torch.save(
                            self._get_save_info(fixed_recon_noise=fixed_recon_noise),
                            model_dir / f"model_iter_{global_step}.pth",
                        )

                    global_step += 1

        # save the final trained model
        torch.save(
            self._get_save_info(fixed_recon_noise=fixed_recon_noise),
            model_dir / f"model_final.pth",
        )

        # training complete yay! :)
        log.info("Training complete")
