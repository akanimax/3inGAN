import dataclasses
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Tuple, Dict, Any, Optional, Union, List

import imageio
import lpips
import numpy as np
import torch
import torch.utils.data as torch_data
from matplotlib import pyplot as plt
from skimage.morphology import dilation
from torch import Tensor, conv3d
from torch.nn.functional import mse_loss, l1_loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from thre3d_atom.data.loaders import (
    PosedImagesDataset,
    PosedImagesDatasetWithVolumetricDensityWeightedImportanceSampling,
    VolumetricDensityWeightedImportanceSamplingConfig,
    PosedImagesDatasetWithFullyRandomImportanceSampling,
    PosedImagesDatasetWithLossWeightedImportanceSampling,
    LossWeightedImportanceSamplingConfig,
)
from thre3d_atom.data.utils import infinite_dataloader
from projects.thre3ingan.singans.networks import (
    RenderMLP,
    get_default_render_mlp,
    get_big_render_mlp,
    get_tiny_render_mlp,
)
from thre3d_atom.modules.volumetric_model.utils import render_image_in_chunks
from thre3d_atom.rendering.volumetric.implicit import cast_rays, raw2alpha_base
from thre3d_atom.rendering.volumetric.render_interface import RenderOut, Rays
from thre3d_atom.rendering.volumetric.utils import (
    collate_rays,
    select_shuffled_rays_and_pixels_synchronously,
)
from thre3d_atom.rendering.volumetric.voxels import (
    FeatureGrid,
    GridLocation,
    VoxelSize,
    render_feature_grid,
    scale_feature_grid_with_scale_factor,
    HybridRGBAFeatureGrid,
    MultiSphereFeatureGrid,
)
from thre3d_atom.utils.constants import (
    NUM_RGBA_CHANNELS,
    NUM_COORD_DIMENSIONS,
    NUM_COLOUR_CHANNELS,
    EXTRA_ACCUMULATED_WEIGHTS,
)
from thre3d_atom.utils.imaging_utils import (
    CameraIntrinsics,
    CameraPose,
    SceneBounds,
    mse2psnr,
    to8b,
    postprocess_disparity_map,
    get_thre360_animation_poses,
)
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import batchify
from thre3d_atom.utils.visualization import visualize_camera_rays


@dataclasses.dataclass
class VolumetricModelRenderingParameters:
    num_rays_chunk: int = 1024
    num_samples_per_ray: int = 64
    num_fine_samples_per_ray: int = 128
    num_points_chunk: int = num_rays_chunk * num_samples_per_ray
    perturb_sampled_points: bool = True
    density_noise_std: float = 0.0
    white_bkgd: bool = False


class VolumetricModel:
    def __init__(
        self,
        render_params: VolumetricModelRenderingParameters,
        grid_dims: Tuple[int, int, int],
        feature_dims: int,
        grid_size: Tuple[float, float, float],
        grid_center: GridLocation = GridLocation(0.0, 0.0, 0.0),
        hybrid_rgba_mode: bool = False,
        render_mlp: Optional[RenderMLP] = None,
        fine_render_mlp: Optional[RenderMLP] = None,
        background_render_mlp: Optional[RenderMLP] = None,
        background_msfeature_grid_dims: Optional[Tuple[int, int, int]] = None,
        feature_scale: float = 100.0,
        use_sh: bool = False,
        apply_diffuse_render_reg: bool = True,
        use_relu_field: bool = True,
        sh_degree: int = 0,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        """ Note that feature_dims default to NUM_RGBA_CHANNELS (for RGBA) if render_mlp is None """
        self._grid_dims = grid_dims

        if render_mlp is not None and hybrid_rgba_mode:
            self._feature_dims = NUM_RGBA_CHANNELS + feature_dims
        elif render_mlp is not None:
            self._feature_dims = feature_dims
        else:
            if use_sh:
                # for spherical harmonics based colour coefficients:
                self._feature_dims = NUM_COLOUR_CHANNELS * ((sh_degree + 1) ** 2)
                # for density
                self._feature_dims = self._feature_dims + 1
            else:
                self._feature_dims = NUM_RGBA_CHANNELS

        self._grid_size = grid_size
        self._background_msfeature_grid_dims = background_msfeature_grid_dims
        self._grid_center = grid_center
        self._render_params = render_params
        self._hybrid_rgba_mode = hybrid_rgba_mode
        self._feature_scale = feature_scale
        self._use_sh = use_sh
        self._use_relu_field = use_relu_field
        self._apply_diffuse_render_reg = apply_diffuse_render_reg
        self._sh_degree = sh_degree
        self._device = device

        features = torch.empty((self._feature_dims, *grid_dims), device=device)
        torch.nn.init.uniform_(features, -1.0, 1.0)
        voxel_size = VoxelSize(
            *[
                grid_dim_size / grid_dim_num
                for grid_dim_size, grid_dim_num in zip(grid_size, grid_dims)
            ]
        )
        if self._hybrid_rgba_mode:
            log.info(f"JUGGERNAUT: going there ... inside hybrid rgba mode")
            self.feature_grid = HybridRGBAFeatureGrid(
                features, voxel_size, grid_center, tunable=True
            )
        else:
            log.info(f"JUGGERNAUT: coming here ... setting identity preactivation")

            # The following is the only change to make this voxel grid be a Relu-Field.
            self.feature_grid = FeatureGrid(
                features,
                voxel_size,
                grid_center,
                tunable=True,
                preactivation=None if self._use_relu_field else torch.abs,
                colour_preactivation=None if self._use_sh else torch.nn.Sigmoid(),
                use_sh=use_sh,
            )
        self.render_mlp = (
            render_mlp.to(self._device) if render_mlp is not None else None
        )
        self.fine_render_mlp = (
            fine_render_mlp.to(self._device) if fine_render_mlp is not None else None
        )
        self.background_render_mlp = (
            background_render_mlp.to(self._device)
            if background_render_mlp is not None
            else None
        )

        if self._background_msfeature_grid_dims is not None:
            bg_features = torch.empty(
                (NUM_RGBA_CHANNELS, *self._background_msfeature_grid_dims),
                device=device,
            )
            torch.nn.init.uniform_(bg_features, -1.0, 1.0)
            self.background_msfg = MultiSphereFeatureGrid(
                bg_features,
                tunable=True,
                preactivation=torch.nn.Sigmoid(),
                colour_preactivation=torch.nn.Sigmoid(),
            )
        else:
            self.background_msfg = None

        if self.fine_render_mlp is not None:
            # copy the weights of the render_mlp into fine_render_mlp
            # so that they get initialized similarly
            self.fine_render_mlp.load_state_dict(self.render_mlp.state_dict())

        # create a shorthand:
        self._hierarchical_sampling = self.fine_render_mlp is not None

        if self._hierarchical_sampling:
            assert self.render_mlp is not None, (
                f"You need both coarse render-mlp and fine render-mlp "
                f"when using hierarchical sampling"
            )

        log.info(f"Feature-Grid Configuration: {self.feature_grid}")
        log.info(f"RenderMLP Configuration: {self.render_mlp}")
        log.info(f"Background Configuration: {self.background_msfg}")
        if self._hierarchical_sampling:
            log.info(f"Fine RenderMLP Configuration: {self.fine_render_mlp}")
        if self.background_render_mlp is not None:
            log.info(
                f"Background RenderMLP Configuration: {self.background_render_mlp}"
            )

        # create a proper colour producer function and transmittance function based on the operating mode:
        if self.render_mlp is None:
            # self._colour_producer = lambda x: adjust_dynamic_range(
            #     torch.tanh(x), (-1.0, 1.0), (0.0, 1.0)
            # )
            if self._use_sh:
                self._colour_producer = (
                    torch.sigmoid
                )  # this is needed after applying the SH coefficients
            else:
                self._colour_producer = lambda x: x
            if self._use_relu_field:
                self._transmittance_behaviour = partial(
                    raw2alpha_base, act_fn=torch.relu, raw_scale=self._feature_scale
                )
            else:
                self._transmittance_behaviour = partial(
                    raw2alpha_base, act_fn=torch.nn.Identity(), raw_scale=1.0
                )
            # self._transmittance_behaviour = lambda x, y: torch.clip(
            #     torch.nn.ReLU()(x * self._feature_scale) * y, 0.0, 1.0
            # )
            # self._transmittance_behaviour = lambda x, y: torch.clip(torch.clip(x, min=0.0) * y, max=1.0)
            # self._transmittance_behaviour = lambda x, y: torch.sigmoid(x * y)
        else:
            self._colour_producer = torch.sigmoid
            self._transmittance_behaviour = lambda x, y: raw2alpha_base(
                x, torch.ones_like(y, device=y.device)
            )

    def get_save_info(self, extra_info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "conf": {
                # don't use self._grid_dims here, because at earlier point in training,
                # the actual spatial-dimensions of the feature could be smaller
                "grid_dims": self.feature_grid.features.shape[:-1],
                "feature_dims": self._feature_dims,
                "grid_size": self._grid_size,
                "grid_center": self._grid_center,
                "hybrid_rgba_mode": self._hybrid_rgba_mode,
                "feature_scale": self._feature_scale,
                "use_sh": self._use_sh,
                "use_relu_field": self._use_relu_field,
                "apply_diffuse_render_reg": self._apply_diffuse_render_reg,
                "background_msfeature_grid_dims": self._background_msfeature_grid_dims,
                "sh_degree": self._sh_degree,
            },
            "render_mlp": self.render_mlp.get_save_info()
            if self.render_mlp is not None
            else None,
            "fine_render_mlp": self.fine_render_mlp.get_save_info()
            if self.fine_render_mlp is not None
            else None,
            "background_render_mlp": self.background_render_mlp.get_save_info()
            if self.background_render_mlp is not None
            else None,
            "render_params": dataclasses.asdict(self._render_params),
            "state_dict": self.feature_grid.state_dict(),
            "bg_msfg_state_dict": self.background_msfg.state_dict()
            if self.background_msfg is not None
            else None,
            "extra_info": extra_info,
        }

    def render(
        self,
        camera_intrinsics: CameraIntrinsics,
        camera_pose: CameraPose,
        scene_bounds: SceneBounds,
        verbose: bool = False,
        hybrid_mode_rgba_only: bool = False,
        diffuse_render: bool = False,
        profile: bool = False,
    ) -> Union[RenderOut, Tuple[RenderOut, RenderOut]]:
        colour_producer, transmittance_behaviour = (
            self._colour_producer,
            self._transmittance_behaviour,
        )
        if hybrid_mode_rgba_only:
            colour_producer = lambda x: torch.clip(x, 0.0, 1.0)
            transmittance_behaviour = lambda x, y: torch.clip(x, 0.0, 1.0)
        return render_image_in_chunks(
            cam_intrinsics=camera_intrinsics,
            camera_pose=camera_pose,
            num_rays_chunk=self._render_params.num_rays_chunk,
            num_samples_per_ray=self._render_params.num_samples_per_ray,
            feature_grid=self.feature_grid,
            processor_mlp=self.render_mlp,
            secondary_processor_mlp=self.fine_render_mlp,
            background_processor_mlp=self.background_render_mlp,
            background_feature_grid=self.background_msfg,
            num_samples_fine=self._render_params.num_fine_samples_per_ray,
            scene_bounds=scene_bounds,
            density_noise_std=self._render_params.density_noise_std,
            perturb_sampled_points=self._render_params.perturb_sampled_points,
            colour_producer=colour_producer,
            raw2alpha=transmittance_behaviour,
            hybrid_mode_only_rgba=hybrid_mode_rgba_only,
            use_sh_based_rendering=self._use_sh,
            white_bkgd=self._render_params.white_bkgd,
            diffuse_render=diffuse_render,
            optimized_sampling_mode=False,
            gpu_render=True,
            device=self._device,
            verbose=verbose,
            profile=profile,
        )

    def _render_feedback(
        self,
        *,
        global_step: int,
        stage: int,
        hemispherical_radius: float,
        camera_intrinsics: CameraIntrinsics,
        render_pose: CameraPose,
        scene_bounds: SceneBounds,
        colour_logs_dir: Path,
        disparity_logs_dir: Path,
        vis_residual_heatmaps: Optional[Tensor] = None,
        render_spinning_animation: bool = True,
        verbose_rendering: bool = True,
    ) -> None:
        # render images
        log.info(f"rendering intermediate output for feedback")

        rendered_output = self.render(
            camera_intrinsics,
            render_pose,
            scene_bounds,
            verbose=verbose_rendering,
        )

        rendered_output_only_rgba_hybrid = None
        if self._hybrid_rgba_mode:
            rendered_output_only_rgba_hybrid = self.render(
                camera_intrinsics,
                render_pose,
                scene_bounds,
                hybrid_mode_rgba_only=True,
                verbose=verbose_rendering,
            )

        rendered_output_diffuse_only = None
        if self._use_sh:
            rendered_output_diffuse_only = self.render(
                camera_intrinsics,
                render_pose,
                scene_bounds,
                diffuse_render=True,
                verbose=verbose_rendering,
            )

        if self._hierarchical_sampling:
            coarse_render, fine_render = rendered_output
            imageio.imwrite(
                colour_logs_dir / f"coarse_stage_{stage}_iter_{global_step}_colour.png",
                to8b(coarse_render.colour.cpu().numpy()),
            )
            imageio.imwrite(
                disparity_logs_dir
                / f"coarse_stage_{stage}_iter_{global_step}_disparity.png",
                postprocess_disparity_map(
                    coarse_render.disparity.squeeze().cpu().numpy()
                ),
            )
            imageio.imwrite(
                colour_logs_dir / f"fine_stage_{stage}_iter_{global_step}_colour.png",
                to8b(fine_render.colour.cpu().numpy()),
            )
            imageio.imwrite(
                disparity_logs_dir
                / f"fine_stage_{stage}_iter_{global_step}_disparity.png",
                postprocess_disparity_map(
                    fine_render.disparity.squeeze().cpu().numpy()
                ),
            )
        else:
            imageio.imwrite(
                colour_logs_dir / f"stage_{stage}_iter_{global_step}_colour.png",
                to8b(rendered_output.colour.cpu().numpy()),
            )
            imageio.imwrite(
                disparity_logs_dir / f"stage_{stage}_iter_{global_step}_disparity.png",
                postprocess_disparity_map(
                    rendered_output.disparity.squeeze().cpu().numpy()
                ),
            )
            imageio.imwrite(
                disparity_logs_dir / f"stage_{stage}_iter_{global_step}_acc.png",
                to8b(rendered_output.extra[EXTRA_ACCUMULATED_WEIGHTS].cpu().numpy()),
            )

        if rendered_output_diffuse_only is not None:
            imageio.imwrite(
                colour_logs_dir
                / f"stage_{stage}_iter_{global_step}_diffuse_colour.png",
                to8b(rendered_output_diffuse_only.colour.cpu().numpy()),
            )
        if rendered_output_only_rgba_hybrid is not None:
            imageio.imwrite(
                colour_logs_dir
                / f"hybrid_rgba_stage_{stage}_iter_{global_step}_colour.png",
                to8b(rendered_output_only_rgba_hybrid.colour.cpu().numpy()),
            )
            imageio.imwrite(
                disparity_logs_dir
                / f"hybrid_rgba_stage_{stage}_iter_{global_step}_disparity.png",
                postprocess_disparity_map(
                    rendered_output_only_rgba_hybrid.disparity.squeeze().cpu().numpy()
                ),
            )
        if vis_residual_heatmaps is not None:
            imageio.mimwrite(
                colour_logs_dir
                / f"stage_{stage}_iter_{global_step}_residual_heatmap_vis.mp4",
                vis_residual_heatmaps,
            )
        if render_spinning_animation:
            # also render the entire spinning animation:
            animation_poses = get_thre360_animation_poses(
                hemispherical_radius=hemispherical_radius,
                camera_pitch=60.0,
                num_poses=42,
            )
            log.info("rendering the 360 degree spinning animation ...")
            rendered_frames = [
                to8b(
                    self.render(camera_intrinsics, anim_pose, scene_bounds)
                    .colour.cpu()
                    .numpy()
                )
                for anim_pose in tqdm(animation_poses)
            ]
            rendered_animation = np.stack(rendered_frames, axis=0)
            imageio.mimwrite(
                colour_logs_dir / f"stage_{stage}_iter_{global_step}_spin_anim.mp4",
                rendered_animation,
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
        vgg_lpips_computer = lpips.LPIPS(net="vgg").to(self._device)
        # TODO: speed up rendering by using cached rays instead of casting
        #  rays again.
        for (image, pose, _) in tqdm(test_dl):
            image, pose = image[0], pose[0]  # testing batching is always 1
            image = image.to(self._device)  # need to bring the image on the GPU
            rendered_output = self.render(
                camera_intrinsics=camera_intrinsics,
                camera_pose=CameraPose(rotation=pose[:, :3], translation=pose[:, 3:]),
                scene_bounds=scene_bounds,
            )
            if self._hierarchical_sampling:
                # all scores are computed on the final output
                rendered_output = rendered_output[1]
            rendered_colour = rendered_output.colour.permute(2, 0, 1)

            with torch.no_grad():
                # compute the PSNR metric:
                psnr = mse2psnr(mse_loss(rendered_colour, image).item())

                # compute the LPIPS metric
                vgg_lpips = vgg_lpips_computer(
                    rendered_colour[None, ...],
                    image[None, ...],
                    normalize=True,
                ).item()

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

    def _get_densities_at_random_points(self, num_points: int = 2048) -> Tensor:
        # sample uniform random points given the feature_grid
        x_vals = self.feature_grid.aabb.x_range[0] + (
            (self.feature_grid.aabb.x_range[1] - self.feature_grid.aabb.x_range[0])
            * torch.rand((num_points, 1), device=self._device, dtype=torch.float32)
        )
        y_vals = self.feature_grid.aabb.y_range[0] + (
            (self.feature_grid.aabb.y_range[1] - self.feature_grid.aabb.y_range[0])
            * torch.rand((num_points, 1), device=self._device, dtype=torch.float32)
        )
        z_vals = self.feature_grid.aabb.z_range[0] + (
            (self.feature_grid.aabb.z_range[1] - self.feature_grid.aabb.z_range[0])
            * torch.rand((num_points, 1), device=self._device, dtype=torch.float32)
        )
        points = torch.cat([x_vals, y_vals, z_vals], dim=-1)

        # Evaluate the feature grid at these points and compute their densities:
        features = self.feature_grid(points)
        random_directions = torch.randn(
            features.shape[0],
            NUM_COORD_DIMENSIONS,
            device=self._device,
            dtype=torch.float32,
        )
        minus_infinity_features = torch.full(
            size=(points.shape[0], self.render_mlp.output_shape[-1]),
            fill_value=-10.0,
            device=self._device,
        )
        raw_densities = torch.where(
            torch.logical_and(
                self.feature_grid.is_inside_volume(points),
                features.abs().sum(dim=-1, keepdim=True) > 0.0,
            ),
            self.render_mlp(
                torch.cat(
                    [features, random_directions],
                    dim=-1,
                )
            ),
            minus_infinity_features,
        )[:, -1:]
        densities = self._transmittance_behaviour(
            raw_densities, torch.ones_like(raw_densities)
        )

        return densities

    def _initialize_mse_loss_weights(
        self,
        train_dataset: PosedImagesDatasetWithLossWeightedImportanceSampling,
        scene_bounds: SceneBounds,
    ) -> None:
        chunk_size = self._render_params.num_rays_chunk
        log.info(
            "Initializing the loss-buffer for Importance Sampling. Please wait ... "
        )
        with torch.no_grad():
            for ray_pixel_index in tqdm(range(0, len(train_dataset._rays), chunk_size)):
                selected_rays = train_dataset._rays[
                    ray_pixel_index : ray_pixel_index + chunk_size
                ]
                selected_pixels = train_dataset._pixels[
                    ray_pixel_index : ray_pixel_index + chunk_size
                ]
                ray_chunk_indices = torch.arange(
                    ray_pixel_index,
                    ray_pixel_index + len(selected_rays),
                    dtype=torch.long,
                    device=self._device,
                )

                # noinspection PyUnboundLocalVariable
                specular_rendered_chunk, _ = render_feature_grid(
                    rays=Rays(
                        origins=selected_rays[:, :NUM_COORD_DIMENSIONS],
                        directions=selected_rays[:, NUM_COORD_DIMENSIONS:],
                    ),
                    num_samples=self._render_params.num_samples_per_ray,
                    feature_grid=self.feature_grid,
                    scene_bounds=scene_bounds,
                    point_processor_network=self.render_mlp,
                    secondary_point_processor_network=self.fine_render_mlp,
                    background_processor_network=self.background_render_mlp,
                    background_feature_grid=self.background_msfg,
                    num_samples_fine=self._render_params.num_fine_samples_per_ray,
                    chunk_size=self._render_params.num_points_chunk,
                    density_noise_std=self._render_params.density_noise_std,
                    colour_producer=self._colour_producer,
                    raw2alpha=self._transmittance_behaviour,
                    perturb_sampled_points=self._render_params.perturb_sampled_points,
                    use_sh_based_rendering=self._use_sh,
                    render_diffuse=False,
                    optimized_sampling_mode=False,
                    white_bkgd=self._render_params.white_bkgd,
                )

                # noinspection PyUnboundLocalVariable
                specular_loss = mse_loss(
                    specular_rendered_chunk.colour, selected_pixels, reduction="none"
                ).mean(dim=-1)

                # noinspection PyUnresolvedReferences
                train_dataset.update_loss_weights(ray_chunk_indices, specular_loss)

    def train(
        self,
        train_dataset: PosedImagesDataset,
        test_dataset: PosedImagesDataset,
        num_iterations_per_stage: List[int],
        num_stages: int,
        # Learning rate related parameters:
        lr_per_stage: List[float],
        lr_decay_gamma_per_stage: List[float],
        lr_decay_steps_per_stage: List[int],
        # Optional parameters:
        scale_factor: float = 2.0,
        save_freq: int = 1000,
        testing_freq: int = 1000,
        feedback_freq: int = 500,
        os_lambda: float = 0.1,
        output_dir: Path = Path(__file__).parent.absolute() / "logs",
        render_feedback_pose: Optional[CameraPose] = None,
        loss_feedback_freq: int = 100,
        num_workers: int = 4,
        # Importance sampling modes: Use only one at a time
        # voxel-crop based importance sampling mode:
        use_voxel_crop_based_sampling: bool = False,
        num_points_vcbis: int = 2,
        patch_percentage_vcbis: float = 2.5,
        # fully random importance sampling mode:
        use_fully_random_importance_sampling: bool = False,
        # mse-loss based importance sampling mode:
        use_mse_loss_weighted_importance_sampling: bool = False,
        mselis_random_percentage: float = 30.0,
        mselis_loss_weights_gamma: float = 3.0,
        # Note that otherwise the sampling mode reverts to the following:
        # Selected image-batch random-subset sampling mode
        image_batch_cache_size: int = 8,
        # other miscellaneous parameters
        verbose_rendering: bool = True,
        fast_debug_mode: bool = False,
        profiling: bool = False,
    ) -> None:
        """
        Trains the volumetric model given a dataset
        Args:
            train_dataset: object of PosedImageDataset used for training the model
            test_dataset: object of PosedImageDataset used for testing the model
                          (Note you can use the `test_mode` while creating the PosedImageDataset object)
            image_batch_cache_size: Number of images to be cached in queue from the disk
            num_stages: number of stages in the coarse-to-fine training
            lr_per_stage: learning rates used per training stage
            lr_decay_gamma_per_stage: value of gamma used for lr_decay per training stage
            lr_decay_steps_per_stage: number of steps-per-stage after which learning rate is decayed
            num_iterations_per_stage: total number of iterations to train the model for
            scale_factor: factor by which the feature-grid is upsampled after each stage
            save_freq: save model after these many fine network iterations
            testing_freq: test the model on the holdout set after these many iterations
            feedback_freq: log feedback after these many iterations
            os_lambda: value of lambda for the OS (opacity sparsity) regularization
            output_dir: write the training assets to this directory
            render_feedback_pose: pose for rendering intermediate training feedback images
            loss_feedback_freq: frequency of logging the loss values to the console
            num_workers: number of data-loader workers
            use_voxel_crop_based_sampling: whether to use VoxelCropBased importance-sampling
            num_points_vcbis: number of points/voxels used per iteration
            patch_percentage_vcbis: patch size for vcbis in terms of %age of original image dimensions
            use_fully_random_importance_sampling: whether to use FullyRandom importance-sampling
            use_mse_loss_weighted_importance_sampling: whether to use mse-loss based importance-sampling
            mselis_random_percentage: the percentage of rays per batch (chunk) which are randomly sampled
            mselis_loss_weights_gamma: values > 1.0 makes the distribution peakier while < 1.0 flattens it
            verbose_rendering: get verbose feedback while rendering the model
            fast_debug_mode: if True, then skips testing and camera_rays_visualization
            profiling: if True, log all the profiling (runtime) information during training
        Returns: None, writes multiple files to the disk.
        """

        assert (
            len(num_iterations_per_stage) == num_stages
        ), f"({num_iterations_per_stage}) num_iterations_per_stage are incompatible with ({num_stages}) num_stages."
        assert (
            len(lr_per_stage) == num_stages
        ), f"({lr_per_stage}) lr_per_stage are incompatible with ({num_stages}) num_stages."
        assert (
            len(lr_decay_gamma_per_stage) == num_stages
        ), f"({lr_decay_gamma_per_stage}) lr_decay_gamma_per_stage are incompatible with ({num_stages}) num_stages."
        assert (
            len(lr_decay_steps_per_stage) == num_stages
        ), f"({lr_decay_steps_per_stage}) lr_decay_steps_per_stage are incompatible with ({num_stages}) num_stages."

        # downscale the feature-grid to the smallest size:
        with torch.no_grad():
            for _ in range(num_stages - 1):
                self.feature_grid = scale_feature_grid_with_scale_factor(
                    self.feature_grid, scale_factor=1 / scale_factor, mode="trilinear"
                )
                if self.background_msfg is not None:
                    self.background_msfg = scale_feature_grid_with_scale_factor(
                        self.background_msfg,
                        scale_factor=1 / scale_factor,
                        mode="trilinear",
                    )

        # setup render_feedback_pose
        real_feedback_image = None
        if render_feedback_pose is None:
            render_feedback_pose = CameraPose(
                rotation=test_dataset[0][1][:, :3],
                translation=test_dataset[0][1][:, 3:],
            )
            real_feedback_image = test_dataset[0][0].permute(1, 2, 0).cpu().numpy()

        # loss feedback frequency:
        loss_feedback_freq = (
            feedback_freq // 10 if loss_feedback_freq is None else loss_feedback_freq
        )

        # setup the scene bounds and camera-intrinsics for the model to train
        scene_bounds = train_dataset.scene_bounds
        camera_intrinsics = train_dataset.camera_intrinsics
        hemispherical_radius = train_dataset.get_hemispherical_radius_estimate()

        # create camera-rays visualization:
        log.info("creating a camera-rays visualization")
        if not fast_debug_mode:
            visualize_camera_rays(
                train_dataset,
                output_dir,
                num_rays_per_image=1,
            )

        # setup the data_loader:
        num_workers = num_workers if not train_dataset.gpu_cached_mode else 0
        if use_voxel_crop_based_sampling:
            log.info(
                f"Using voxel-crop based Importance sampling with {num_points_vcbis} voxel-crops per iteration"
            )
            new_train_dataset = PosedImagesDatasetWithVolumetricDensityWeightedImportanceSampling(
                config=VolumetricDensityWeightedImportanceSamplingConfig(
                    density_grid=self.feature_grid.features[..., -1].detach(),
                    aabb=self.feature_grid.aabb,
                    num_samples=num_points_vcbis,
                    patch_percentage=patch_percentage_vcbis,
                ),
                images_dir=train_dataset._images_dir,
                camera_params_json=train_dataset._camera_params_json,
                image_data_range=train_dataset._image_data_range,
                unit_normalize_scene_scale=train_dataset._unit_normalize_scene_scale,
                downsample_factor=train_dataset._downsample_factor,
                rgba_white_bkgd=train_dataset._rgba_white_bkgd,
            )
            torch.cuda.empty_cache()
            del train_dataset
            train_dl = torch_data.DataLoader(new_train_dataset, batch_size=None)
            infinite_train_dl = iter(train_dl)
            dataset_size = new_train_dataset.dataset_size
        elif use_fully_random_importance_sampling:  # use_fully_random_ray_sampling:
            log.info(f"Using fully random Importance sampling of rays")
            new_train_dataset = PosedImagesDatasetWithFullyRandomImportanceSampling(
                batch_size=self._render_params.num_rays_chunk,
                images_dir=train_dataset._images_dir,
                camera_params_json=train_dataset._camera_params_json,
                image_data_range=train_dataset._image_data_range,
                unit_normalize_scene_scale=train_dataset._unit_normalize_scene_scale,
                downsample_factor=train_dataset._downsample_factor,
                rgba_white_bkgd=train_dataset._rgba_white_bkgd,
            )
            torch.cuda.empty_cache()
            del train_dataset
            train_dl = torch_data.DataLoader(new_train_dataset, batch_size=None)
            infinite_train_dl = iter(train_dl)
            dataset_size = new_train_dataset.dataset_size
        elif use_mse_loss_weighted_importance_sampling:
            log.info(f"Using instantaneous loss based Importance sampling of rays")
            new_train_dataset = PosedImagesDatasetWithLossWeightedImportanceSampling(
                config=LossWeightedImportanceSamplingConfig(
                    batch_size=self._render_params.num_rays_chunk,
                    random_percentage=mselis_random_percentage,
                    loss_weights_gamma=mselis_loss_weights_gamma,
                ),
                images_dir=train_dataset._images_dir,
                camera_params_json=train_dataset._camera_params_json,
                image_data_range=train_dataset._image_data_range,
                unit_normalize_scene_scale=train_dataset._unit_normalize_scene_scale,
                downsample_factor=train_dataset._downsample_factor,
                rgba_white_bkgd=train_dataset._rgba_white_bkgd,
            )
            torch.cuda.empty_cache()
            del train_dataset
            self._initialize_mse_loss_weights(new_train_dataset, scene_bounds)
            train_dl = torch_data.DataLoader(new_train_dataset, batch_size=None)
            infinite_train_dl = iter(train_dl)
            dataset_size = new_train_dataset.dataset_size
        else:
            train_dl = torch_data.DataLoader(
                train_dataset,
                batch_size=image_batch_cache_size,
                shuffle=True,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=num_workers > 0,
                prefetch_factor=num_workers if num_workers > 0 else 2,
                persistent_workers=num_workers > 0,
            )
            infinite_train_dl = iter(infinite_dataloader(train_dl))
            dataset_size = (
                len(train_dl)
                * train_dataset.camera_intrinsics.height
                * train_dataset.camera_intrinsics.width
            )

        test_dl = torch_data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            prefetch_factor=num_workers if num_workers > 0 else 2,
            persistent_workers=num_workers > 0,
        )

        # setup output directories
        # fmt: off
        model_dir = output_dir / "saved_models"
        logs_dir = output_dir / "training_logs"
        tensorboard_dir = logs_dir / "tensorboard"
        render_dir = logs_dir / "rendered_output"
        colour_logs_dir = render_dir / "colour"
        disparity_logs_dir = render_dir / "disparity"
        for directory in (model_dir, logs_dir, tensorboard_dir,
                          render_dir, colour_logs_dir, disparity_logs_dir):
            directory.mkdir(exist_ok=True, parents=True)
        # fmt: on

        # save the real_feedback_test_image if it exists:
        if real_feedback_image is not None:
            log.info(f"Logging real feedback image")
            imageio.imwrite(
                colour_logs_dir / f"1__real_log.png",
                to8b(real_feedback_image),
            )

        # setup tensorboard writer
        tensorboard_writer = SummaryWriter(tensorboard_dir)

        # start actual training
        log.info("Beginning training")
        global_step = 0
        time_spent_actually_training = 0
        for stage in range(1, num_stages + 1):
            # setup model optimizer
            log.info("Using new_modified learning rate schedule ...")

            # pick the learning rate parameters for the current stage:
            current_stage_lr = lr_per_stage[stage - 1]
            current_stage_lr_decay_gamma = lr_decay_gamma_per_stage[stage - 1]
            current_stage_lr_decay_steps = lr_decay_steps_per_stage[stage - 1]

            optimizer = torch.optim.Adam(
                params=[
                    {"params": self.feature_grid.parameters(), "lr": current_stage_lr},
                    {
                        "params": self.render_mlp.parameters()
                        if self.render_mlp is not None
                        else [],
                        "lr": 0.1 * current_stage_lr,
                    },
                    {
                        "params": self.fine_render_mlp.parameters()
                        if self.fine_render_mlp is not None
                        else [],
                        "lr": 0.1 * current_stage_lr,
                    },
                    {
                        "params": self.background_render_mlp.parameters()
                        if self.background_render_mlp is not None
                        else [],
                        "lr": 0.1 * current_stage_lr,
                    },
                    {
                        "params": self.background_msfg.parameters()
                        if self.background_msfg is not None
                        else [],
                        "lr": current_stage_lr,
                    },
                ],
                betas=(0.9, 0.999),
            )

            # setup learning rate schedulers for the optimizer
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=current_stage_lr_decay_gamma
            )

            log.info(
                f"Training stage: {stage}   Feature grid resolution: {self.feature_grid.features.shape}"
            )
            if self.background_msfg is not None:
                log.info(
                    f"Background grid resolution: {self.background_msfg.features.shape}"
                )

            # log the current learning rates being used:
            current_stage_lrs = [
                param_group["lr"] for param_group in optimizer.param_groups
            ]
            log_string = f"Current stage learning rates: {current_stage_lrs} "
            log.info(log_string)

            num_iterations_in_current_stage = num_iterations_per_stage[stage - 1]
            stage_iteration = 0
            global_last_time = time.time()
            ray_chunk_indices = None
            while stage_iteration < num_iterations_in_current_stage:
                start_time = time.time()
                last_time = start_time

                # =========================================================================================
                # Ray sampling block (Can be one of the three modes we support)
                # =========================================================================================
                if use_voxel_crop_based_sampling:
                    # noinspection PyUnresolvedReferences
                    infinite_train_dl._dataset.density_grid = (
                        self.feature_grid.features[..., -1].detach()
                    )
                    selected_pixels, selected_rays = next(infinite_train_dl)
                    selected_rays = Rays(
                        selected_rays[:, :NUM_COORD_DIMENSIONS],
                        selected_rays[:, NUM_COORD_DIMENSIONS:],
                    )
                elif use_fully_random_importance_sampling:
                    selected_pixels, selected_rays = next(infinite_train_dl)
                    selected_rays = Rays(
                        selected_rays[:, :NUM_COORD_DIMENSIONS],
                        selected_rays[:, NUM_COORD_DIMENSIONS:],
                    )
                elif use_mse_loss_weighted_importance_sampling:
                    if ray_chunk_indices is not None:
                        # march the selected rays from previous iteration in specular and diffuse modes
                        # and obtain the updated loss-weights
                        with torch.no_grad():
                            # noinspection PyUnboundLocalVariable
                            specular_rendered_chunk, _ = render_feature_grid(
                                rays=selected_rays,
                                num_samples=self._render_params.num_samples_per_ray,
                                feature_grid=self.feature_grid,
                                scene_bounds=scene_bounds,
                                point_processor_network=self.render_mlp,
                                secondary_point_processor_network=self.fine_render_mlp,
                                background_processor_network=self.background_render_mlp,
                                background_feature_grid=self.background_msfg,
                                num_samples_fine=self._render_params.num_fine_samples_per_ray,
                                chunk_size=self._render_params.num_points_chunk,
                                density_noise_std=self._render_params.density_noise_std,
                                colour_producer=self._colour_producer,
                                raw2alpha=self._transmittance_behaviour,
                                perturb_sampled_points=self._render_params.perturb_sampled_points,
                                use_sh_based_rendering=self._use_sh,
                                render_diffuse=False,
                                optimized_sampling_mode=False,
                                white_bkgd=self._render_params.white_bkgd,
                            )
                            # noinspection PyUnboundLocalVariable
                            specular_loss = mse_loss(
                                specular_rendered_chunk.colour.detach(),
                                selected_pixels,
                                reduction="none",
                            ).mean(dim=-1)

                            # noinspection PyUnresolvedReferences
                            infinite_train_dl._dataset.update_loss_weights(
                                ray_chunk_indices, specular_loss
                            )

                    ray_chunk_indices, selected_pixels, selected_rays = next(
                        infinite_train_dl
                    )
                    selected_rays = Rays(
                        selected_rays[:, :NUM_COORD_DIMENSIONS],
                        selected_rays[:, NUM_COORD_DIMENSIONS:],
                    )
                else:
                    images, poses, possible_rays = next(infinite_train_dl)
                    images, poses = images.to(self._device), poses.to(self._device)

                    if profiling:
                        log.info(
                            f"Loading batch of images and poses "
                            f"to {self._device} took time: {(time.time() - last_time) * 1000} ms"
                        )
                        last_time = time.time()

                    if possible_rays is not None:
                        # use the cached rays
                        rays = possible_rays.reshape(-1, possible_rays.shape[-1])
                    else:
                        # cast rays for all images in the current batch:
                        rays_list = []
                        for pose in poses:
                            casted_rays = cast_rays(
                                camera_intrinsics,
                                CameraPose(
                                    rotation=pose[:, :3], translation=pose[:, 3:]
                                ),
                                device=self._device,
                            )
                            rays_list.append(casted_rays)
                        rays = collate_rays(rays_list)

                    if profiling:
                        log.info(
                            f"Ray creation took time: {(time.time() - last_time) * 1000} ms"
                        )
                        last_time = time.time()

                    # images are of shape [B x C x H x W] and pixels are [B * H * W x C]
                    pixels = images.permute(0, 2, 3, 1).reshape(-1, images.shape[1])

                    # shuffle rays and pixels synchronously
                    (
                        selected_rays,
                        selected_pixels,
                    ) = select_shuffled_rays_and_pixels_synchronously(
                        rays, pixels, self._render_params.num_rays_chunk
                    )

                if profiling:
                    log.info(
                        f"Ray random sampling took time: {(time.time() - last_time) * 1000} ms"
                    )
                    last_time = time.time()
                # =========================================================================================

                # render a small chunk of rays and compute a loss on it
                specular_rendered_chunk, profiling_info = render_feature_grid(
                    rays=selected_rays,
                    num_samples=self._render_params.num_samples_per_ray,
                    feature_grid=self.feature_grid,
                    scene_bounds=scene_bounds,
                    point_processor_network=self.render_mlp,
                    secondary_point_processor_network=self.fine_render_mlp,
                    background_processor_network=self.background_render_mlp,
                    background_feature_grid=self.background_msfg,
                    num_samples_fine=self._render_params.num_fine_samples_per_ray,
                    chunk_size=self._render_params.num_points_chunk,
                    density_noise_std=self._render_params.density_noise_std,
                    colour_producer=self._colour_producer,
                    raw2alpha=self._transmittance_behaviour,
                    perturb_sampled_points=self._render_params.perturb_sampled_points,
                    use_sh_based_rendering=self._use_sh,
                    optimized_sampling_mode=False,
                    white_bkgd=self._render_params.white_bkgd,
                )
                if profiling:
                    log.info(f"SPECULAR_render")
                    log.info(f"Sampling time: {profiling_info['sampling']} ms")
                    log.info(f"Processing time: {profiling_info['processing']} ms")
                    log.info(f"Accumulation time: {profiling_info['accumulation']} ms")

                # compute loss and perform gradient update
                loss_coarse, loss_fine = None, None
                mse_coarse, mse_fine = None, None
                if self._hierarchical_sampling:
                    rendered_chunk_coarse, rendered_chunk_fine = specular_rendered_chunk
                    rendered_colour_coarse = rendered_chunk_coarse.colour
                    rendered_colour_fine = rendered_chunk_fine.colour
                    loss_coarse = l1_loss(rendered_colour_coarse, selected_pixels)
                    mse_coarse = mse_loss(rendered_colour_coarse, selected_pixels)
                    loss_fine = l1_loss(rendered_colour_fine, selected_pixels)
                    mse_fine = mse_loss(rendered_colour_fine, selected_pixels)
                    total_loss = loss_coarse + loss_fine
                else:
                    total_loss = l1_loss(
                        specular_rendered_chunk.colour, selected_pixels
                    )
                    loss_coarse = total_loss
                    mse_coarse = mse_loss(
                        specular_rendered_chunk.colour, selected_pixels
                    )

                opacity_sparsity_loss = None
                if self.render_mlp is not None:
                    # we apply sparsity regularization only when using the Feature-Grid + MLP model
                    # not in RGBA volume optimization
                    opacity_sparsity_loss = (
                        self._get_densities_at_random_points().mean()
                    )

                    total_loss = total_loss + (os_lambda * opacity_sparsity_loss)

                loss_hybrid_rgba, mse_hybrid_rgba = None, None
                if self._hybrid_rgba_mode:
                    # obtain the rgba_only part of the grid's output and apply a loss on it:
                    rendered_chunk_rgba_hybrid = render_feature_grid(
                        rays=selected_rays,
                        num_samples=self._render_params.num_samples_per_ray,
                        feature_grid=self.feature_grid,
                        scene_bounds=scene_bounds,
                        point_processor_network=self.render_mlp,
                        secondary_point_processor_network=self.fine_render_mlp,
                        background_processor_network=self.background_render_mlp,
                        background_feature_grid=self.background_msfg,
                        num_samples_fine=self._render_params.num_fine_samples_per_ray,
                        chunk_size=self._render_params.num_points_chunk,
                        density_noise_std=self._render_params.density_noise_std,
                        colour_producer=lambda x: torch.clip(x, 0.0, 1.0),
                        raw2alpha=lambda x, y: torch.clip(x, 0.0, 1.0),
                        perturb_sampled_points=self._render_params.perturb_sampled_points,
                        hybrid_mode_only_rgba=True,
                        white_bkgd=self._render_params.white_bkgd,
                    )

                    rgba_only_loss = l1_loss(
                        rendered_chunk_rgba_hybrid.colour, selected_pixels
                    )

                    total_loss = total_loss + rgba_only_loss
                    loss_hybrid_rgba = rgba_only_loss
                    mse_hybrid_rgba = mse_loss(
                        rendered_chunk_rgba_hybrid.colour, selected_pixels
                    )

                loss_diffuse, mse_diffuse = None, None
                if self._use_sh and self._apply_diffuse_render_reg:
                    # we also apply the regularization on diffuse colours:
                    diffuse_render_chunk, profiling_info = render_feature_grid(
                        rays=selected_rays,
                        num_samples=self._render_params.num_samples_per_ray,
                        feature_grid=self.feature_grid,
                        scene_bounds=scene_bounds,
                        point_processor_network=self.render_mlp,
                        secondary_point_processor_network=self.fine_render_mlp,
                        background_processor_network=self.background_render_mlp,
                        background_feature_grid=self.background_msfg,
                        num_samples_fine=self._render_params.num_fine_samples_per_ray,
                        chunk_size=self._render_params.num_points_chunk,
                        density_noise_std=self._render_params.density_noise_std,
                        colour_producer=self._colour_producer,
                        raw2alpha=self._transmittance_behaviour,
                        perturb_sampled_points=self._render_params.perturb_sampled_points,
                        use_sh_based_rendering=self._use_sh,
                        render_diffuse=True,
                        optimized_sampling_mode=False,
                        white_bkgd=self._render_params.white_bkgd,
                    )
                    if profiling:
                        log.info(f"DIFFUSE_render")
                        log.info(f"Sampling time: {profiling_info['sampling']} ms")
                        log.info(f"Processing time: {profiling_info['processing']} ms")
                        log.info(
                            f"Accumulation time: {profiling_info['accumulation']} ms"
                        )

                    loss_diffuse = l1_loss(diffuse_render_chunk.colour, selected_pixels)
                    mse_diffuse = mse_loss(diffuse_render_chunk.colour, selected_pixels)
                    total_loss = total_loss + loss_diffuse

                if profiling:
                    log.info(
                        f"Total Forward pass (aka. rendering) took time: {(time.time() - last_time) * 1000} ms"
                    )
                    last_time = time.time()

                # optimization steps:
                optimizer.zero_grad()

                if profiling:
                    log.info(
                        f"Gradient buffer flushing took time: {(time.time() - last_time) * 1000} ms"
                    )
                    last_time = time.time()

                total_loss.backward()

                if profiling:
                    log.info(
                        f"Gradient computation took time: {(time.time() - last_time) * 1000} ms"
                    )
                    last_time = time.time()

                optimizer.step()

                if profiling:
                    log.info(
                        f"Parameter update took time: {(time.time() - last_time) * 1000} ms"
                    )

                global_step += 1
                stage_iteration += 1

                if profiling:
                    log.info(
                        f"Total time for a single iteration: {(time.time() - start_time) * 1000} ms"
                    )

                time_spent_actually_training += time.time() - global_last_time

                # ====================================================================
                # Non-training block
                # ====================================================================
                # Compute training batch-wise psnrs:
                psnr_coarse, psnr_fine, psnr_hybrid_rgba, psnr_diffuse = [None] * 4
                if (
                    global_step % loss_feedback_freq == 0
                    or stage_iteration == 0
                    or stage_iteration == num_iterations_in_current_stage - 1
                ):
                    psnr_coarse = (
                        mse2psnr(mse_coarse) if mse_coarse is not None else None
                    )
                    psnr_fine = mse2psnr(mse_fine) if mse_fine is not None else None
                    psnr_hybrid_rgba = (
                        mse2psnr(mse_hybrid_rgba)
                        if mse_hybrid_rgba is not None
                        else None
                    )
                    psnr_diffuse = (
                        mse2psnr(mse_diffuse) if mse_diffuse is not None else None
                    )

                # tensorboard summaries feedback
                if (
                    global_step % loss_feedback_freq == 0
                    or stage_iteration == 0
                    or stage_iteration == num_iterations_in_current_stage - 1
                ):
                    for summary_name, summary_value in (
                        ("loss_coarse", loss_coarse),
                        ("loss_fine", loss_fine),
                        ("psnr_coarse", psnr_coarse),
                        ("psnr_fine", psnr_fine),
                        ("opacity_regularizer", opacity_sparsity_loss),
                        ("loss_hybrid_rgba", loss_hybrid_rgba),
                        ("psnr_hybrid_rgba", psnr_hybrid_rgba),
                        ("diffuse_loss", loss_diffuse),
                        ("diffuse_psnr", psnr_diffuse),
                        ("total_loss", total_loss),
                        (
                            "num_epochs",
                            (self._render_params.num_rays_chunk * global_step)
                            / dataset_size
                            if dataset_size is not None
                            else None,
                        ),
                    ):
                        if summary_value is not None:
                            tensorboard_writer.add_scalar(
                                summary_name, summary_value, global_step=global_step
                            )

                # console loss feedback
                if (
                    global_step % loss_feedback_freq == 0
                    or stage_iteration == 0
                    or stage_iteration == num_iterations_in_current_stage - 1
                ):
                    loss_info_string = (
                        f"Stage: {stage} "
                        f"Global Iteration: {global_step} "
                        f"Stage Iteration: {stage_iteration} "
                    )

                    if loss_coarse is not None and psnr_coarse is not None:
                        loss_info_string += (
                            f"Loss_coarse: {loss_coarse: .5f} "
                            f"PSNR_coarse: {psnr_coarse: .5f} "
                        )

                    if loss_fine is not None and psnr_fine is not None:
                        loss_info_string += (
                            f"Loss_fine: {loss_fine: .5f} "
                            f"PSNR_fine: {psnr_fine: .5f} "
                        )
                    if loss_hybrid_rgba is not None and psnr_hybrid_rgba is not None:
                        loss_info_string += (
                            f"Loss_hybrid_rgba: {loss_hybrid_rgba: .5f} "
                            f"PSNR_hybrid_rgba: {psnr_hybrid_rgba: .5f} "
                        )
                    if loss_diffuse is not None and psnr_diffuse is not None:
                        loss_info_string += (
                            f"Loss_diffuse: {loss_diffuse: .5f} "
                            f"PSNR_diffuse: {psnr_diffuse: .5f} "
                        )
                    if opacity_sparsity_loss is not None:
                        loss_info_string += f"Sparsity_loss: {opacity_sparsity_loss} "

                    loss_info_string += f"Total_loss: {total_loss: .5f} "
                    log.info(loss_info_string)

                # step the learning rate schedulers
                if stage_iteration % current_stage_lr_decay_steps == 0:
                    lr_scheduler.step()
                    new_lrs = [
                        param_group["lr"] for param_group in optimizer.param_groups
                    ]
                    log_string = f"Adjusted learning rate | learning rates: {new_lrs} "
                    log.info(log_string)

                if stage_iteration == num_iterations_in_current_stage:
                    break
                    # no need to render and or test or save model at the end

                # rendered feedback
                if (
                    global_step % feedback_freq == 0
                    or stage_iteration == 1
                    or stage_iteration == num_iterations_in_current_stage - 1
                ):
                    # noinspection PyUnresolvedReferences,PyProtectedMember
                    self._render_feedback(
                        global_step=global_step,
                        stage=stage,
                        hemispherical_radius=hemispherical_radius,
                        camera_intrinsics=camera_intrinsics,
                        render_pose=render_feedback_pose,
                        scene_bounds=scene_bounds,
                        colour_logs_dir=colour_logs_dir,
                        disparity_logs_dir=disparity_logs_dir,
                        vis_residual_heatmaps=infinite_train_dl._dataset.residual_visualization
                        if use_mse_loss_weighted_importance_sampling
                        else None,
                        render_spinning_animation=True,
                        verbose_rendering=verbose_rendering,
                    )

                # obtain and log the test metrics
                if not fast_debug_mode and (
                    global_step % testing_freq == 0
                    or stage_iteration == num_iterations_in_current_stage - 1
                ):
                    self.test(
                        test_dl=test_dl,
                        camera_intrinsics=camera_intrinsics,
                        scene_bounds=test_dataset.scene_bounds,
                        global_step=global_step,
                        tensorboard_writer=tensorboard_writer,
                    )

                # save the model
                if (
                    global_step % save_freq == 0
                    or stage_iteration == num_iterations_in_current_stage - 1
                ):
                    torch.save(
                        self.get_save_info(
                            extra_info={
                                "scene_bounds": scene_bounds,
                                "camera_intrinsics": camera_intrinsics,
                                "hemispherical_radius": hemispherical_radius,
                            }
                        ),
                        model_dir / f"model_stage_{stage}_iter_{global_step}.pth",
                    )
                # ====================================================================

                global_last_time = time.time()

            # plot a histogram of #times a pixel is sampled during training:
            if use_voxel_crop_based_sampling:
                flat_counts = (
                    infinite_train_dl._dataset._count.cpu().numpy().reshape(-1)
                )
                fig = plt.figure()
                plt.plot(range(len(flat_counts)), flat_counts)
                plt.tight_layout()
                plt.savefig(
                    colour_logs_dir / f"pixel_sample_dist_stage{stage}.png", dpi=600
                )
                plt.close(fig)

            # don't upsample the feature grid if the last stage is complete
            if stage != num_stages:
                # upsample the feature-grid after the completion of the stage:
                with torch.no_grad():
                    self.feature_grid = scale_feature_grid_with_scale_factor(
                        self.feature_grid, scale_factor=scale_factor, mode="trilinear"
                    )
                    if self.background_msfg is not None:
                        self.background_msfg = scale_feature_grid_with_scale_factor(
                            self.background_msfg,
                            scale_factor=scale_factor,
                            mode="trilinear",
                        )
                # # register new feature-grid's parameters with the optimizer
                # optimizer.param_groups[0]["params"] = list(
                #     self.feature_grid.parameters()
                # )

        # save the final trained model
        torch.save(
            self.get_save_info(
                extra_info={
                    "scene_bounds": scene_bounds,
                    "camera_intrinsics": camera_intrinsics,
                    "hemispherical_radius": hemispherical_radius,
                }
            ),
            model_dir / f"model_final.pth",
        )

        # training complete yay! :)
        log.info("Training complete")
        log.info(
            f"Total actual training time: {timedelta(seconds=time_spent_actually_training)}"
        )


def create_vol_mod_from_saved_model(
    saved_model: Path, device: torch.device
) -> Tuple[VolumetricModel, Dict[str, Any]]:
    loaded_model = torch.load(saved_model)
    extra_info = loaded_model["extra_info"]

    if loaded_model["render_mlp"] is not None:
        if len(loaded_model["render_mlp"]["point_mlp"]["config"]["layer_depths"]) > 5:
            mlp_creator = get_big_render_mlp
        elif len(loaded_model["render_mlp"]["point_mlp"]["config"]["layer_depths"]) < 3:
            mlp_creator = get_tiny_render_mlp
        else:
            mlp_creator = get_default_render_mlp
        render_mlp = mlp_creator(
            feature_size=loaded_model["render_mlp"]["point_embedder"]["input_dims"],
            feature_embeddings_dims=loaded_model["render_mlp"]["point_embedder"][
                "emb_dims"
            ],
            dir_embedding_dims=loaded_model["render_mlp"]["dir_embedder"]["emb_dims"],
            normalize_features=loaded_model["render_mlp"]["normalize_features"],
        )
        render_mlp.load_weights(loaded_model["render_mlp"])
    else:
        render_mlp = None

    if (
        "fine_render_mlp" in loaded_model
        and loaded_model["fine_render_mlp"] is not None
    ):
        if (
            len(loaded_model["fine_render_mlp"]["point_mlp"]["config"]["layer_depths"])
            > 5
        ):
            mlp_creator = get_big_render_mlp
        elif (
            len(loaded_model["fine_render_mlp"]["point_mlp"]["config"]["layer_depths"])
            < 3
        ):
            mlp_creator = get_tiny_render_mlp
        else:
            mlp_creator = get_default_render_mlp
        fine_render_mlp = mlp_creator(
            feature_size=loaded_model["fine_render_mlp"]["point_embedder"][
                "input_dims"
            ],
            feature_embeddings_dims=loaded_model["fine_render_mlp"]["point_embedder"][
                "emb_dims"
            ],
            dir_embedding_dims=loaded_model["fine_render_mlp"]["dir_embedder"][
                "emb_dims"
            ],
            normalize_features=loaded_model["fine_render_mlp"]["normalize_features"],
        )
        fine_render_mlp.load_weights(loaded_model["fine_render_mlp"])
    else:
        fine_render_mlp = None

    if (
        "background_render_mlp" in loaded_model
        and loaded_model["background_render_mlp"] is not None
    ):
        if (
            len(
                loaded_model["background_render_mlp"]["point_mlp"]["config"][
                    "layer_depths"
                ]
            )
            > 5
        ):
            mlp_creator = get_big_render_mlp
        elif (
            len(
                loaded_model["background_render_mlp"]["point_mlp"]["config"][
                    "layer_depths"
                ]
            )
            < 3
        ):
            mlp_creator = get_tiny_render_mlp
        else:
            mlp_creator = get_default_render_mlp
        background_render_mlp = mlp_creator(
            feature_size=NUM_COORD_DIMENSIONS,
            feature_embeddings_dims=loaded_model["background_render_mlp"][
                "point_embedder"
            ]["emb_dims"],
            dir_embedding_dims=loaded_model["background_render_mlp"]["dir_embedder"][
                "emb_dims"
            ],
            normalize_features=loaded_model["background_render_mlp"][
                "normalize_features"
            ],
        )
        background_render_mlp.load_weights(loaded_model["background_render_mlp"])
    else:
        background_render_mlp = None

    if "grid_center" in loaded_model["conf"]:
        # We reset the Grid location to Origin after it's trained :)
        del loaded_model["conf"]["grid_center"]

    if "linear_disparity_sampling" in loaded_model["render_params"]:
        del loaded_model["render_params"]["linear_disparity_sampling"]

    if (
        "hybrid_rgba_mode" in loaded_model["conf"]
        and loaded_model["conf"]["hybrid_rgba_mode"]
    ):
        loaded_model["conf"]["feature_dims"] -= NUM_RGBA_CHANNELS

    volumetric_model = VolumetricModel(
        **loaded_model["conf"],
        render_params=VolumetricModelRenderingParameters(
            **loaded_model["render_params"]
        ),
        render_mlp=render_mlp,
        fine_render_mlp=fine_render_mlp,
        background_render_mlp=background_render_mlp,
        device=device,
    )
    volumetric_model.feature_grid.load_state_dict(loaded_model["state_dict"])
    if volumetric_model.background_msfg is not None:
        volumetric_model.background_msfg.load_state_dict(
            loaded_model["bg_msfg_state_dict"]
        )
    return volumetric_model, extra_info


# noinspection PyProtectedMember
def process_rgba_model(
    vol_mod: VolumetricModel,
    threshold_value: float = 0,
    num_dilation_passes: int = 1,
) -> VolumetricModel:
    features = vol_mod.feature_grid.features
    rgb, alpha = features[..., :-1], features[..., -1:]
    log.info(
        f"Preprocessing rgba volume ... using a density threshold of {threshold_value} "
        f"and {num_dilation_passes} number of alpha morph-dilation passes"
    )
    if vol_mod.feature_grid._preactivation is not None:
        rgb_mask = (
            vol_mod.feature_grid._preactivation(alpha) > threshold_value
        ).float()
    else:
        rgb_mask = (alpha > threshold_value).float()
    dilated_rgb_mask = rgb_mask
    for _ in range(num_dilation_passes):
        dilated_rgb_mask = conv3d(
            dilated_rgb_mask[None, ...].permute(0, 4, 1, 2, 3),
            torch.ones(1, 1, 3, 3, 3, device=dilated_rgb_mask.device),
            padding=1,
        )[0].permute(1, 2, 3, 0)
        dilated_rgb_mask = torch.clip(dilated_rgb_mask, 0.0, 1.0)
    rgb_masked = rgb * dilated_rgb_mask
    alpha_masked = alpha * dilated_rgb_mask

    log.info(f"using a different empty space value than 0.0 ...")
    # change the 0.0 values to -10.0
    # note the -10.0 is needed so that when a tanh is applied, it becomes -1.0
    rgb_masked[rgb_masked == 0.0] = -10.0
    alpha_masked[alpha_masked == 0.0] = -10.0

    processed_features = torch.cat([rgb_masked, alpha_masked], dim=-1)
    vol_mod.feature_grid.features = processed_features
    return vol_mod


def process_hybrid_rgba_volumetric_model(
    vol_mod: VolumetricModel,
    threshold_value: float = 0.12,
    num_dilation_passes: int = 1,
) -> VolumetricModel:
    assert vol_mod.get_save_info({})["conf"][
        "hybrid_rgba_mode"
    ], "Can't process a non-hybrid feature-grid using this processor"

    features = vol_mod.feature_grid.features

    # decode the features using the fine_render_mlp (or the render_mlp):
    _, _, _, feats_dim = features.shape
    rgba_part = features[..., :NUM_RGBA_CHANNELS]
    _, a_values = (
        rgba_part[..., :NUM_COLOUR_CHANNELS],
        rgba_part[..., NUM_COLOUR_CHANNELS:],
    )
    # noinspection PyProtectedMember
    # a_values = vol_mod._transmittance_behaviour(a_values, torch.ones_like(a_values))

    only_features = features[..., NUM_RGBA_CHANNELS:]

    log.info(
        f"Preprocessing the feature-grid volume ... using a density threshold of {threshold_value} "
        f"and with number of dilation passes of {num_dilation_passes}"
    )

    features_mask = (a_values > threshold_value).float()
    dilated_features_mask = features_mask
    for _ in range(num_dilation_passes):
        dilated_features_mask = conv3d(
            dilated_features_mask[None, ...].permute(0, 4, 1, 2, 3),
            torch.ones(1, 1, 3, 3, 3, device=dilated_features_mask.device),
            padding=1,
        )[0].permute(1, 2, 3, 0)
        dilated_features_mask = torch.clip(dilated_features_mask, 0.0, 1.0)

    masked_features = only_features * dilated_features_mask
    old_feature_grid = vol_mod.feature_grid
    vol_mod.feature_grid = FeatureGrid(
        masked_features.permute(3, 0, 1, 2),
        old_feature_grid.voxel_size,
        old_feature_grid.grid_location,
        old_feature_grid.tunable,
    )

    return vol_mod


def process_volumetric_model(
    vol_mod: VolumetricModel,
    threshold_value: float = 0.12,
    num_dilation_passes: int = 1,
    chunk_size: int = 65536,
) -> VolumetricModel:
    features = vol_mod.feature_grid.features

    # decode the features using the fine_render_mlp (or the render_mlp):
    x_dim, y_dim, z_dim, feats_dim = features.shape
    flat_features = features.reshape(-1, feats_dim)
    random_directions = torch.randn(
        flat_features.shape[0], NUM_COORD_DIMENSIONS, device=flat_features.device
    )
    network_input = torch.cat([flat_features, random_directions], dim=-1)

    log.info(
        f"Preprocessing the feature-grid volume ... using a density threshold of {threshold_value} "
        f"and with number of dilation passes of {num_dilation_passes}"
    )
    with torch.no_grad():
        decoded_features = batchify(
            processor_fn=vol_mod.render_mlp,
            collate_fn=partial(torch.cat, dim=0),
            chunk_size=chunk_size,
            verbose=True,
        )(network_input)

    # noinspection PyUnresolvedReferences
    decoded_features = decoded_features.reshape(x_dim, y_dim, z_dim, -1)
    _, alpha = decoded_features[..., :-1], decoded_features[..., -1:]

    # noinspection PyProtectedMember
    alpha = vol_mod._transmittance_behaviour(alpha, torch.ones_like(alpha))

    features_mask = (alpha > threshold_value).cpu().numpy()
    dilated_features_mask = features_mask
    for _ in range(num_dilation_passes):
        dilated_features_mask_z = np.stack(
            [
                dilation(mask)
                for mask in dilated_features_mask[..., 0].transpose(2, 0, 1)
            ],
            axis=0,
        ).transpose(1, 2, 0)[..., None]
        dilated_features_mask_y = np.stack(
            [
                dilation(mask)
                for mask in dilated_features_mask[..., 0].transpose(1, 0, 2)
            ],
            axis=0,
        ).transpose(1, 0, 2)[..., None]
        dilated_features_mask_x = np.stack(
            [
                dilation(mask)
                for mask in dilated_features_mask[..., 0].transpose(0, 1, 2)
            ],
            axis=0,
        ).transpose(0, 1, 2)[..., None]
        dilated_features_mask = (
            dilated_features_mask_x | dilated_features_mask_y | dilated_features_mask_z
        )

    dilated_features_mask = torch.from_numpy(dilated_features_mask).to(features.device)

    filtered_features = features * dilated_features_mask
    vol_mod.feature_grid.features = filtered_features
    return vol_mod
