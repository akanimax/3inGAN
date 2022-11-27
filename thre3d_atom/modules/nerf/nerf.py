"""Module implements the NeRF (Neural Radiance Fields) model"""
import copy
import dataclasses
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as torch_data
from lpips import lpips
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from thre3d_atom.data.loaders import PosedImagesDataset
from thre3d_atom.data.utils import infinite_dataloader
from thre3d_atom.modules.nerf.nerf_network import NerfNet
from thre3d_atom.modules.nerf.utils import hierarchical_sampler, ndcize_rays
from thre3d_atom.rendering.volumetric.implicit import (
    process_points_on_rays_with_network,
    accumulate_processed_points_on_rays,
    cast_rays,
    raw2alpha_base,
)
from thre3d_atom.rendering.volumetric.sample import sample_uniform_points_on_rays
from thre3d_atom.rendering.volumetric.render_interface import (
    Rays,
    RenderOut,
    RaySamplerFunction,
    PointProcessorFunction,
    AccumulatorFunction,
    render,
)
from thre3d_atom.rendering.volumetric.utils import (
    reshape_rendered_output,
    shuffle_rays_and_pixels_synchronously,
    collate_rays,
    collate_coarse_and_fine_rendered_output,
)
from thre3d_atom.utils.constants import EXTRA_POINT_DEPTHS, EXTRA_POINT_WEIGHTS
from thre3d_atom.utils.imaging_utils import (
    CameraIntrinsics,
    SceneBounds,
    CameraPose,
    to8b,
    mse2psnr,
    adjust_dynamic_range,
    postprocess_disparity_map,
)
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import batchify
from thre3d_atom.utils.visualization import visualize_camera_rays


@dataclasses.dataclass
class NerfRenderingParameters:
    num_rays_chunk: int = 1024
    num_points_chunk: int = 64 * 1024
    num_coarse_samples: int = 64
    num_fine_samples: int = num_coarse_samples
    perturb_coarse_sampled_points: bool = True
    linear_disparity_sampling: bool = False
    ndcize_rays: bool = False
    training_density_noise_std: float = 1.0
    use_viewdirs: bool = True


def get_default_vol_rend_components(
    render_params: NerfRenderingParameters,
    network: NerfNet,
    override_perturb_coarse_sampled_points: Optional[bool] = None,
    override_training_density_noise_std: Optional[float] = None,
) -> Tuple[RaySamplerFunction, PointProcessorFunction, AccumulatorFunction]:
    perturb = (
        render_params.perturb_coarse_sampled_points
        if override_perturb_coarse_sampled_points is None
        else override_perturb_coarse_sampled_points
    )
    density_noise_std = (
        render_params.training_density_noise_std
        if override_training_density_noise_std is None
        else override_training_density_noise_std
    )
    sampler_fn = partial(
        sample_uniform_points_on_rays,
        perturb=perturb,
        linear_disparity_sampling=render_params.linear_disparity_sampling,
    )
    point_processor_fn = partial(
        process_points_on_rays_with_network,
        network=network,
        chunk_size=render_params.num_points_chunk,
        use_viewdirs=render_params.use_viewdirs,
    )
    accumulator_fn = partial(
        accumulate_processed_points_on_rays,
        density_noise_std=density_noise_std,
        raw2_alpha=raw2alpha_base,
    )
    return sampler_fn, point_processor_fn, accumulator_fn


class Nerf:
    def __init__(
        self,
        nerf_net_coarse: NerfNet,
        render_params: NerfRenderingParameters,
        nerf_net_fine: Optional[NerfNet] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        self._device = device
        self._render_params = render_params
        self._net_coarse = nerf_net_coarse.to(self._device)
        self._net_fine = (
            nerf_net_fine.to(self._device) if nerf_net_fine is not None else None
        )
        log.info(f"Coarse Network Configuration: {self._net_coarse}")
        log.info(f"Fine Network Configuration: {self._net_fine}")

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def _get_save_info(self, extra_info: Dict[str, Any]) -> Dict[str, Any]:
        save_info = {
            "nerf_net_coarse": self._net_coarse.get_save_info(),
            "render_params": dataclasses.asdict(self._render_params),
            "extra_info": extra_info,
        }

        if self._net_fine is not None:
            save_info["nerf_net_fine"] = self._net_fine.get_save_info()

        return save_info

    def _render_rays(
        self, rays: Rays, scene_bounds: SceneBounds, test_mode: bool = False
    ) -> Tuple[RenderOut, RenderOut]:
        override_perturb_coarse_samples = False if test_mode else None
        override_training_density_noise_std = 0.0 if test_mode else None

        (
            sampler_fn,
            point_processor_fn,
            accumulator_fn,
        ) = get_default_vol_rend_components(
            self._render_params,
            self._net_coarse,
            override_perturb_coarse_sampled_points=override_perturb_coarse_samples,
            override_training_density_noise_std=override_training_density_noise_std,
        )

        # bring the rays on the GPU if they are not on the gpu:
        if rays.origins.device != self._device:
            rays = rays.to(self._device)

        # Render the rays with a batchified point processor for better efficiency
        rendered_output = render(
            rays=rays,
            scene_bounds=scene_bounds,
            num_samples=self._render_params.num_coarse_samples,
            sampler_fn=sampler_fn,
            point_processor_fn=point_processor_fn,
            accumulator_fn=accumulator_fn,
        )

        if self._net_fine is not None:
            # use the hierarchical sampler for sampling fine points
            fine_sampler_fn = partial(
                hierarchical_sampler,
                coarse_point_weights=rendered_output.extra[
                    EXTRA_POINT_WEIGHTS
                ].detach(),
                coarse_point_depths=rendered_output.extra[EXTRA_POINT_DEPTHS].detach(),
            )

            # create a point_processor_function using the `net_fine`
            fine_processor_fn = partial(
                process_points_on_rays_with_network,
                network=self._net_fine,
                chunk_size=self._render_params.num_points_chunk,
                use_viewdirs=True,
            )

            # render the fine output using the same accumulator function
            # but the specialized fine_sampling and fine_point_processing functions
            fine_rendered_output = render(
                rays=rays,
                scene_bounds=scene_bounds,
                num_samples=self._render_params.num_fine_samples,
                sampler_fn=fine_sampler_fn,
                point_processor_fn=fine_processor_fn,
                accumulator_fn=accumulator_fn,
            )

            return rendered_output, fine_rendered_output

        return rendered_output, None

    @staticmethod
    def _shuffle_rays(rays: Rays) -> Rays:
        random_perm = torch.randperm(rays.origins.shape[0])
        shuffled_ray_origins = rays.origins[random_perm, :]
        shuffled_ray_directions = rays.directions[random_perm, :]
        return Rays(shuffled_ray_origins, shuffled_ray_directions)

    def visualize_network_evaluation_distribution(
        self,
        dataset: PosedImagesDataset,
        output_file_path: Path,
        num_rays_per_image: int = 3,
    ) -> None:
        scene_bounds = (
            dataset.scene_bounds
            if not self._render_params.ndcize_rays
            else SceneBounds(0.0, 1.0)
        )
        all_poses = [
            dataset.extract_pose(camera_param)
            for camera_param in dataset.camera_parameters.values()
        ]
        all_camera_locations = []

        fig, ax = plt.subplots(2, 2)
        fig.suptitle("Network evaluation distribution")

        all_rays_for_visualization = []
        for pose in all_poses:
            rays = cast_rays(dataset.camera_intrinsics, pose)
            if self._render_params.ndcize_rays:
                rays = ndcize_rays(rays, dataset.camera_intrinsics)

            # add the ray origin to camera location
            all_camera_locations.append(rays.origins[0])

            # randomly select only num_rays_per_image rays for visualization
            shuffled_rays = self._shuffle_rays(rays)
            selected_rays = shuffled_rays[:num_rays_per_image]
            all_rays_for_visualization.append(selected_rays)
        all_rays_for_visualization = collate_rays(all_rays_for_visualization)

        # render all the rays to obtain the depths:
        batchified_render = batchify(
            partial(self._render_rays, scene_bounds=scene_bounds),
            collate_fn=collate_coarse_and_fine_rendered_output,
            chunk_size=self._render_params.num_rays_chunk,
            verbose=True,
        )
        with torch.no_grad():
            log.info("Creating network_evaluation distribution visualization ...")
            coarse_rendered_output, fine_rendered_output = batchified_render(
                all_rays_for_visualization
            )
        if fine_rendered_output is None:
            fine_rendered_output = coarse_rendered_output
        evaluated_space_points_depths = (
            fine_rendered_output.extra[EXTRA_POINT_DEPTHS][..., None].detach().cpu()
        )
        density_distribution = all_rays_for_visualization.origins[:, None, ...] + (
            evaluated_space_points_depths
            * all_rays_for_visualization.directions[:, None, ...]
        )
        all_points = density_distribution.reshape(
            -1, all_rays_for_visualization.origins.shape[-1]
        ).numpy()

        # 2D-projection based heatmaps
        cmin = 10 if self._render_params.linear_disparity_sampling else None
        ax[0, 0].set_title("XY-projection")
        ax[0, 0].set_xlabel("X-axis")
        ax[0, 0].set_ylabel("Y-axis")
        ax[0, 0].hist2d(
            all_points[:, 0], all_points[:, 1], bins=500, cmin=cmin, cmap="jet"
        )
        ax[1, 0].set_title("YZ-projection")
        ax[1, 0].set_xlabel("Y-axis")
        ax[1, 0].set_ylabel("Z-axis")
        ax[1, 0].hist2d(
            all_points[:, 1], all_points[:, 2], bins=500, cmin=cmin, cmap="jet"
        )
        ax[1, 1].set_title("XZ-projection")
        ax[1, 1].set_xlabel("X-axis")
        ax[1, 1].set_ylabel("Z-axis")
        ax[1, 1].hist2d(
            all_points[:, 0], all_points[:, 2], bins=500, cmin=cmin, cmap="jet"
        )

        # displaying top-down view of all camera locations to use the space :)
        all_camera_locations = torch.stack(all_camera_locations, dim=0)
        ax[0, 1].set_title("XY-camera_locations")
        ax[0, 1].set_xlabel("X-axis")
        ax[0, 1].set_ylabel("Y-axis")
        ax[0, 1].scatter(
            all_camera_locations[:, 0],
            all_camera_locations[:, 1],
            color="m",
            marker="^",
        )

        # save the figure
        plt.tight_layout()
        plt.savefig(output_file_path, dpi=600)
        plt.close(fig)

    def toggle_all_networks(self, mode: str):
        all_nets = [
            network
            for network in (self._net_coarse, self._net_fine)
            if network is not None
        ]
        for network in all_nets:
            if mode.lower() == "train":
                network.train()
            elif mode.lower() == "eval":
                network.eval()
            else:
                raise ValueError(f"Unknown mode requested: {mode}")

    def render(
        self,
        camera_intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
        scene_bounds: SceneBounds,
        verbose: bool = True,
    ) -> Tuple[RenderOut, RenderOut]:
        """The rendered output will be the final fine output if one is being used.
        Else, it will be the coarse network's output"""
        if verbose:
            log.info("Casting rays for rendering")
        rays = cast_rays(camera_intrinsics, camera_pose)
        if self._render_params.ndcize_rays:
            rays = ndcize_rays(rays, camera_intrinsics)

        # batchified rendering (aka. ray_processing):
        batchified_render = batchify(
            partial(self._render_rays, scene_bounds=scene_bounds, test_mode=True),
            collate_fn=collate_coarse_and_fine_rendered_output,
            chunk_size=self._render_params.num_rays_chunk,
            verbose=verbose,
        )

        if verbose:
            log.info("Rendering the NeRF output chunk-by-chunk")

        with torch.no_grad():
            coarse_render_out, fine_render_out = batchified_render(rays)
            # bring them both on the cpu and reshape the output properly:
            coarse_render_out = reshape_rendered_output(
                coarse_render_out.to(torch.device("cpu")), camera_intrinsics
            )
            if fine_render_out is not None:
                fine_render_out = reshape_rendered_output(
                    fine_render_out.to(torch.device("cpu")), camera_intrinsics
                )

        return coarse_render_out, fine_render_out

    def _render_feedback(
        self,
        global_step: int,
        camera_intrinsics: CameraIntrinsics,
        render_pose: CameraPose,
        scene_bounds: SceneBounds,
        colour_logs_dir: Path,
        disparity_logs_dir: Path,
        verbose_rendering: bool = True,
    ) -> None:
        # render images
        log.info(f"rendering intermediate output for feedback")
        coarse_rendered_output, fine_rendered_output = self.render(
            camera_intrinsics,
            render_pose,
            scene_bounds,
            verbose=verbose_rendering,
        )

        for training_mode, rendered_output in (
            ("coarse", coarse_rendered_output),
            ("fine", fine_rendered_output),
        ):
            if rendered_output is not None:
                imageio.imwrite(
                    colour_logs_dir / f"{training_mode}_iter_{global_step}_colour.png",
                    to8b(rendered_output.colour.numpy()),
                )
                imageio.imwrite(
                    disparity_logs_dir
                    / f"{training_mode}_iter_{global_step}_disparity.png",
                    postprocess_disparity_map(
                        rendered_output.disparity.squeeze().numpy()
                    ),
                )

    def test(
        self,
        test_dl: torch_data.DataLoader,
        camera_intrinsics: CameraIntrinsics,
        scene_bounds: SceneBounds,
        global_step: Optional[int] = 0,
        tensorboard_writer: Optional[SummaryWriter] = None,
    ) -> None:
        log.info(f"Testing the model on {len(test_dl.dataset)} heldout images")
        all_psnrs, all_lpips = [], []
        vgg_lpips_computer = lpips.LPIPS(net="vgg")
        for (image, pose) in tqdm(test_dl):
            image, pose = image[0], pose[0]  # testing batching is always 1
            coarse_rendered_output, fine_rendered_output = self.render(
                camera_intrinsics=camera_intrinsics,
                camera_pose=CameraPose(rotation=pose[:, :3], translation=pose[:, 3:]),
                scene_bounds=scene_bounds,
                verbose=False,
            )
            rendered_output = (
                coarse_rendered_output
                if fine_rendered_output is None
                else fine_rendered_output
            )
            rendered_colour = rendered_output.colour.permute(2, 0, 1)

            with torch.no_grad():
                # compute the PSNR metric:
                psnr = mse2psnr(torch.nn.functional.mse_loss(rendered_colour, image))

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

    def train(
        self,
        train_dataset: PosedImagesDataset,
        test_dataset: PosedImagesDataset,
        image_batch_cache_size: int = 3,
        num_iterations: int = 1e6,
        save_freq: int = 500,
        feedback_freq: int = 500,
        test_freq: int = 5000,
        learning_rate: float = 0.003,
        lr_decay_steps: int = 250,
        lr_decay_gamma: float = 0.1,
        output_dir: Path = Path(__file__).parent.absolute() / "logs",
        render_feedback_pose: Optional[CameraPose] = None,
        loss_feedback_freq: Optional[int] = None,
        num_workers: int = 4,
        verbose_rendering: bool = True,
        fast_debug_mode: bool = False,
    ) -> None:
        """
        Trains a Nerf model given a dataset
        Args:
            train_dataset: object of PosedImageDataset used for training the model
            test_dataset: object of PosedImageDataset used for testing the model
                          (Note you can use the `test_mode` while creating the PosedImageDataset object)
            image_batch_cache_size: Number of images to be cached in queue from the disk
            num_iterations: total number of iterations to train the model for
            save_freq: save model after these many fine network iterations
            feedback_freq: log feedback after these many iterations
            test_freq: compute test scores on a heldout set after these many
            learning_rate: learning rate for the optimization process
            lr_decay_steps: The learning rate decays by gamma (lr_decay_gamma) exponentially
                            after these many steps
            lr_decay_gamma: gamma value used for exponential learning_rate decay
            output_dir: write the training assets to this directory
            render_feedback_pose: pose for rendering intermediate training feedback images
            num_workers:  num_workers (processes) used for loading the training data
            loss_feedback_freq: frequency of logging the loss values to the console
            verbose_rendering: get verbose feedback while rendering the model
            fast_debug_mode: when true, skips camera_rays visualization and testing
        Returns: None, writes multiple files to the disk.
        """
        # setup render_feedback_pose and the real image if using the test dataset
        render_feedback_real_image = None
        if render_feedback_pose is None:
            render_feedback_pose = CameraPose(
                rotation=test_dataset[0][-1][:, :3].numpy(),
                translation=test_dataset[0][-1][:, 3:].numpy(),
            )
            render_feedback_real_image = test_dataset[0][0].permute(1, 2, 0).numpy()

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
        inf_train_dl = iter(infinite_dataloader(train_dl))
        test_dl = torch_data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # create a camera_rays visualization:
        if not fast_debug_mode:
            log.info("Creating a camera rays visualization ...")
            visualize_camera_rays(
                train_dataset,
                output_dir,
                num_rays_per_image=10,
                do_ndcize_rays=self._render_params.ndcize_rays,
            )

        # dataset size aka number of total pixels
        dataset_size = (
            len(train_dl)
            * train_dataset.camera_intrinsics.height
            * train_dataset.camera_intrinsics.width
        )

        # setup optimizers
        optimizer = torch.optim.Adam(
            params=[
                {"params": self._net_coarse.parameters(), "lr": learning_rate},
                {
                    "params": self._net_fine.parameters()
                    if self._net_fine is not None
                    else [],
                    "lr": learning_rate,
                },
            ],
            betas=(0.9, 0.999),
        )

        # setup learning rate schedulers for the two optimizers
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_decay_gamma
        )

        # setup output directories
        model_dir = output_dir / "saved_models"
        logs_dir = output_dir / "training_logs"
        tensorboard_dir = logs_dir / "tensorboard"
        render_dir = logs_dir / "rendered_output"
        colour_logs_dir = render_dir / "colour"
        disparity_logs_dir = render_dir / "disparity"
        density_distribution_dir = render_dir / "density_distribution"
        for directory in (
            model_dir,
            logs_dir,
            tensorboard_dir,
            render_dir,
            colour_logs_dir,
            disparity_logs_dir,
            density_distribution_dir,
        ):
            directory.mkdir(exist_ok=True, parents=True)

        # log the real_feedback image if there is one to the colour_logs_dir
        if render_feedback_real_image is not None:
            imageio.imwrite(
                colour_logs_dir / "1_real_log.png", to8b(render_feedback_real_image)
            )

        # setup the scene bounds for the model to train
        scene_bounds = (
            train_dataset.scene_bounds
            if not self._render_params.ndcize_rays
            else SceneBounds(0.0, 1.0)
        )

        # setup tensorboard writer
        tensorboard_writer = SummaryWriter(tensorboard_dir)

        # start actual training
        log.info("Beginning training")
        global_step = 0
        self.toggle_all_networks("train")  # put all networks in training mode
        while global_step < num_iterations:
            images, poses = next(inf_train_dl)

            # cast rays for all images in the current batch:
            rays_list = []
            for pose in poses:
                casted_rays = cast_rays(
                    train_dataset.camera_intrinsics,
                    CameraPose(rotation=pose[:, :3], translation=pose[:, 3:]),
                    device=self._device,
                )
                if self._render_params.ndcize_rays:
                    casted_rays = ndcize_rays(
                        casted_rays, train_dataset.camera_intrinsics
                    )
                rays_list.append(casted_rays)
            rays = collate_rays(rays_list)

            # a shorthand for chunk_size
            chunk_size = self._render_params.num_rays_chunk

            # images are of shape [B x C x H x W] and pixels are [B * H * W x C]
            pixels = (
                images.permute(0, 2, 3, 1).reshape(-1, images.shape[1]).to(self._device)
            )

            # shuffle rays and pixels synchronously
            rays, pixels = shuffle_rays_and_pixels_synchronously(rays, pixels)

            # select only num_rays_chunk number of rays from evey image
            selected_rays, selected_pixels = (
                rays[: chunk_size * image_batch_cache_size],
                pixels[: chunk_size * image_batch_cache_size],
            )

            # iterate of the selected_rays and selected_pixels to train the model
            # chunk-by-chunk
            for chunk_index in range(0, selected_pixels.shape[0], chunk_size):
                rays_chunk = selected_rays[chunk_index : chunk_index + chunk_size]
                pixel_colours_chunk = selected_pixels[
                    chunk_index : chunk_index + chunk_size
                ]

                # render a small chunk of rays and compute a loss on it
                (
                    rendered_rays_coarse_chunk,
                    rendered_rays_fine_chunk,
                ) = self._render_rays(rays_chunk, scene_bounds)

                # compute loss
                coarse_rendered_colour = rendered_rays_coarse_chunk.colour
                loss_coarse = mse_loss(coarse_rendered_colour, pixel_colours_chunk)
                psnr_coarse = mse2psnr(loss_coarse)

                loss = loss_coarse
                loss_fine, psnr_fine = None, None
                if rendered_rays_fine_chunk is not None:
                    fine_rendered_colour = rendered_rays_fine_chunk.colour
                    loss_fine = mse_loss(fine_rendered_colour, pixel_colours_chunk)
                    psnr_fine = mse2psnr(loss_fine)
                    loss = loss_coarse + loss_fine

                # optimization step:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # step the learning rate schedulers
                if (global_step + 1) % lr_decay_steps == 0:
                    lr_scheduler.step()
                    new_lrs = [
                        param_group["lr"] for param_group in optimizer.param_groups
                    ]
                    log_string = f"Adjusted learning rate | New LRs: {new_lrs} "
                    log.info(log_string)

                # tensorboard summaries feedback
                for summary_name, summary_value in (
                    ("loss_coarse", loss_coarse),
                    ("psnr_coarse", psnr_coarse),
                    ("loss_fine", loss_fine),
                    ("psnr_fine", psnr_fine),
                    ("total_loss", loss.item()),
                    ("num_epochs", global_step / dataset_size),
                ):
                    if summary_value is not None:
                        tensorboard_writer.add_scalar(
                            summary_name, summary_value, global_step=global_step
                        )

                # console loss feedback
                if global_step % loss_feedback_freq == 0 or global_step == 0:
                    loss_info_string = (
                        f"Iteration: {global_step} "
                        f"Loss_coarse: {loss_coarse: .5f} "
                        f"PSNR_coarse: {psnr_coarse: .5f} "
                    )
                    if self._net_fine is not None:
                        loss_info_string += (
                            f"Loss_fine: {loss_fine: .5f} "
                            f"PSNR_fine: {psnr_fine: .5f} "
                        )
                    loss_info_string += f"Total_loss: {loss.item(): .5f} "
                    log.info(loss_info_string)

                # rendered feedback
                if global_step % feedback_freq == 0 or global_step == 0:
                    self._render_feedback(
                        global_step,
                        train_dataset.camera_intrinsics,
                        render_feedback_pose,
                        scene_bounds,
                        colour_logs_dir,
                        disparity_logs_dir,
                        verbose_rendering=verbose_rendering,
                    )

                    self.visualize_network_evaluation_distribution(
                        copy.deepcopy(train_dataset),
                        density_distribution_dir / f"iter_{global_step}.png",
                        num_rays_per_image=100,
                    )

                if not fast_debug_mode and ((global_step + 1) % test_freq == 0):
                    # obtain and log test metrics
                    self.test(
                        test_dl=test_dl,
                        camera_intrinsics=test_dataset.camera_intrinsics,
                        scene_bounds=test_dataset.scene_bounds,
                        global_step=global_step,
                        tensorboard_writer=tensorboard_writer,
                    )

                # save the model
                if global_step % save_freq == 0 or global_step == 0:
                    torch.save(
                        self._get_save_info(
                            extra_info={
                                "scene_bounds": scene_bounds,
                                "camera_intrinsics": train_dataset.camera_intrinsics,
                                "hemispherical_radius": train_dataset.get_hemispherical_radius_estimate(),
                            }
                        ),
                        model_dir / f"model_iter_{global_step}.pth",
                    )

                global_step += 1

                if global_step == num_iterations - 1:
                    break

        # save the final trained model
        torch.save(
            self._get_save_info(
                extra_info={
                    "scene_bounds": scene_bounds,
                    "camera_intrinsics": train_dataset.camera_intrinsics,
                    "hemispherical_radius": train_dataset.get_hemispherical_radius_estimate(),
                }
            ),
            model_dir / f"model_final.pth",
        )

        # training complete yay! :)
        log.info("Training complete")
