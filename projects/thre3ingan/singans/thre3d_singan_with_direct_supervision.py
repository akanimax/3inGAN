import copy
import dataclasses
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List, Union

import imageio
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import mse_loss, grid_sample, interpolate
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from projects.thre3ingan.singans.networks import (
    RenderMLP,
    Thre3dSinGanGeneratorDS,
    Thre3dSinGanDiscriminatorDS,
    get_default_render_mlp,
    get_big_render_mlp,
    get_tiny_render_mlp,
    TwodSinGanDiscriminator,
)
from thre3d_atom.modules.volumetric_model.utils import render_image_in_chunks
from thre3d_atom.rendering.volumetric.implicit import cast_rays, raw2alpha_base
from thre3d_atom.rendering.volumetric.render_interface import RenderOut, Rays
from thre3d_atom.rendering.volumetric.utils import (
    collate_rendered_output,
    reshape_and_rebuild_flat_rays,
)
from thre3d_atom.rendering.volumetric.voxels import (
    FeatureGrid,
    get_voxel_size_from_scene_bounds_and_hem_rad,
    scale_feature_grid_with_output_size,
    render_feature_grid,
    VoxelSize,
)
from thre3d_atom.training.adversarial.losses import WganGPGanLoss
from thre3d_atom.training.adversarial.models import (
    Discriminator,
)
from thre3d_atom.utils.constants import (
    NUM_RGBA_CHANNELS,
    NUM_COORD_DIMENSIONS,
    NUM_COLOUR_CHANNELS,
)
from thre3d_atom.utils.imaging_utils import (
    CameraPose,
    CameraIntrinsics,
    SceneBounds,
    mse2psnr,
    pose_spherical,
    to8b,
    postprocess_disparity_map,
    scale_camera_intrinsics,
)
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import toggle_grad, batchify


@dataclasses.dataclass
class Thre3dSinGanWithDirectSupervisionRenderingParameters:
    num_rays_chunk: int = 2048
    num_points_chunk: int = 64 * 1024
    num_samples_per_ray: int = 64
    num_fine_samples_per_ray: int = 64
    perturb_sampled_points: bool = True
    density_noise_std: float = 1.0
    # We initialize camera_intrinsics and scene_bounds to random values
    camera_intrinsics: CameraIntrinsics = CameraIntrinsics(100, 100, 100)
    scene_bounds: SceneBounds = SceneBounds(1.0, 10.0)


class Thre3dSinGanWithDirectSupervision:
    def __init__(
        self,
        render_params: Thre3dSinGanWithDirectSupervisionRenderingParameters,
        render_mlp: Optional[RenderMLP] = None,
        fine_render_mlp: Optional[RenderMLP] = None,
        num_stages: int = 8,
        output_grid_resolution: Tuple[int, int, int] = (256, 256, 256),
        intermediate_features: Union[int, List[int]] = 32,
        scale_factor: float = (1 / 0.75),
        use_eql: bool = True,
        noise_scale: float = 0.1,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        # state of the object:
        self._render_params = render_params
        self._num_stages = num_stages
        self._output_grid_resolution = output_grid_resolution
        self._feature_dims = (
            render_mlp.feature_dims if render_mlp is not None else NUM_RGBA_CHANNELS
        )
        self._scale_factor = scale_factor
        self._use_eql = use_eql
        self._noise_scale = noise_scale
        self._device = device

        # attach the renderMLP (Rosetta Stone ;)) to the object properly
        if render_mlp is not None:
            self._render_mlp = render_mlp.to(self._device)
            # turn off gradients for the render_mlp
            toggle_grad(self._render_mlp, False)
        else:
            self._render_mlp = None

        if fine_render_mlp is not None:
            self._fine_render_mlp = fine_render_mlp.to(self._device)
            # turn off gradients for the fine_render_mlp
            toggle_grad(self._fine_render_mlp, False)
        else:
            self._fine_render_mlp = None

        # create a Thre3dSinGanDS generator object:
        self._generator = Thre3dSinGanGeneratorDS(
            output_resolution=self._output_grid_resolution,
            num_features=self._feature_dims,
            num_intermediate_features=intermediate_features,
            num_stages=self._num_stages,
            scale_factor=self._scale_factor,
            use_eql=self._use_eql,
            device=self._device,
            noise_scale=self._noise_scale,
        )

        # create a trainable render_mlp for decoding the generator:
        # noinspection PyProtectedMember
        self._trainable_render_mlp = (
            get_tiny_render_mlp(
                self._feature_dims,
                feature_embeddings_dims=self._render_mlp._point_embedder._emb_dims,
                dir_embedding_dims=self._render_mlp._dir_embedder._emb_dims,
                normalize_features=False,
            ).to(self._device)
            if self._render_mlp is not None
            else None
        )
        if self._trainable_render_mlp is not None:
            self._trainable_render_mlp.load_weights(self._render_mlp.get_save_info())
        log.info(f"Trainable MLP architecture: {self._trainable_render_mlp}")

        self._grid_sizes = self._generator.grid_sizes  # short hand

        # this is attached later because generator's init builds the list
        # noinspection PyProtectedMember
        self._intermediate_features = self._generator._num_intermediate_features

        log.info(
            f"Created a 3D-SinGanDS Generator \n"
            f"with stage-wise output grid_sizes: {self._grid_sizes}"
        )
        log.info(f"Generator Architecture: {self._generator}")

        # create a short-hand for the specific noise vector which is used for the reconstruction task:
        self._reconstruction_noise = self._generator.reconstruction_noise

        # create a proper colour producer function and transmittance function based on the operating mode:
        if self._render_mlp is None:
            self._colour_producer = lambda x: torch.clip(x, 0.0, 1.0)
            self._transmittance_behaviour = lambda x, y: torch.clip(x, 0.0, 1.0)
        else:
            self._colour_producer = torch.sigmoid
            self._transmittance_behaviour = lambda x, y: raw2alpha_base(
                x, torch.ones_like(y, device=y.device)
            )

    @property
    def render_params(self) -> Thre3dSinGanWithDirectSupervisionRenderingParameters:
        return self._render_params

    @property
    def reconstruction_noise(self) -> Tensor:
        return self._reconstruction_noise

    @reconstruction_noise.setter
    def reconstruction_noise(self, recon_noise: Tensor) -> None:
        self._reconstruction_noise = recon_noise

    def load_generator_weights(self, state_dict: Dict[str, Any]) -> None:
        self._generator.load_state_dict(state_dict)

    def _get_save_info(
        self,
        discriminator_3d: Optional[Thre3dSinGanDiscriminatorDS] = None,
        discriminator_2d: Optional[Discriminator] = None,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "render_params": dataclasses.asdict(self._render_params),
            "conf": {
                "num_stages": self._num_stages,
                "output_grid_resolution": self._output_grid_resolution,
                "intermediate_features": self._intermediate_features,
                "scale_factor": self._scale_factor,
                "use_eql": self._use_eql,
                "noise_scale": self._noise_scale,
            },
            "trainable_render_mlp": self._trainable_render_mlp.get_save_info()
            if self._trainable_render_mlp is not None
            else None,
            "render_mlp": self._render_mlp.get_save_info()
            if self._render_mlp is not None
            else None,
            "fine_render_mlp": self._fine_render_mlp.get_save_info()
            if self._fine_render_mlp is not None
            else None,
            "reconstruction_noise": self._reconstruction_noise,
            "2d_discriminator": discriminator_2d.get_save_info()
            if discriminator_2d is not None
            else None,
            "3d_discriminator": discriminator_3d.get_save_info()
            if discriminator_3d is not None
            else None,
            "generator": self._generator.get_save_info(),
            "extra_info": extra_info if extra_info is not None else {},
        }

    @staticmethod
    def _average_stats(stats_dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        total_values = len(stats_dict_list)
        averaged_stats_dict = {}
        for stats_dict in stats_dict_list:
            for key, value in stats_dict.items():
                averaged_stats_dict[key] = averaged_stats_dict.get(key, 0) + (
                    (value.item() if hasattr(value, "item") else value) / total_values
                )
        return averaged_stats_dict

    def _setup_stage_wise_training_reals(
        self, training_feature_grid: FeatureGrid
    ) -> List[FeatureGrid]:
        activated_train_fg = copy.deepcopy(training_feature_grid)
        activated_train_fg.features = torch.tanh(activated_train_fg.features)
        activated_train_fg._preactivation = torch.nn.Identity()
        current_grid = copy.deepcopy(activated_train_fg)
        stage_wise_feature_grids = []
        # full_dim = max(*self._grid_sizes[-1])
        log.info("Preprocessing the Grid to obtain the multi-scale pyramid ... ")
        log.info(
            "This is a time consuming process when done right, so please be patient ..."
        )
        for fg_size in tqdm(reversed(self._grid_sizes[:-1])):
            fg = scale_feature_grid_with_output_size(
                feature_grid=current_grid, output_size=fg_size
            )
            current_grid = copy.deepcopy(fg)
            # clip_min = -(max(*fg_size) / full_dim)
            # fg.features = torch.clip(fg.features, clip_min, 1)
            stage_wise_feature_grids.append(fg)
        stage_wise_feature_grids = list(reversed(stage_wise_feature_grids))
        stage_wise_feature_grids.append(activated_train_fg)
        return stage_wise_feature_grids

    def _setup_stage_wise_camera_intrinsics(
        self,
        virtual_camera_intrinsics: Optional[CameraIntrinsics] = None,
    ) -> List[CameraIntrinsics]:
        if virtual_camera_intrinsics is not None:
            new_focal_length = self._render_params.camera_intrinsics.focal * (
                virtual_camera_intrinsics.height
                / self._render_params.camera_intrinsics.height
            )
            current_cam_int = CameraIntrinsics(
                virtual_camera_intrinsics.height,
                virtual_camera_intrinsics.width,
                new_focal_length,
            )
            camera_ints = [current_cam_int]
            for _ in range(self._num_stages - 1):
                current_cam_int = scale_camera_intrinsics(
                    current_cam_int, 1 / self._scale_factor
                )
                camera_ints.append(current_cam_int)
            return list(reversed(camera_ints))
        else:
            camera_ints = [self._render_params.camera_intrinsics]
            current_cam_int = copy.deepcopy(camera_ints[-1])
            for _ in range(self._num_stages - 1):
                camera_ints.append(
                    scale_camera_intrinsics(current_cam_int, 1 / self._scale_factor)
                )
                current_cam_int = scale_camera_intrinsics(
                    current_cam_int, 1 / self._scale_factor
                )
            return list(reversed(camera_ints))

    def _render_stage_wise_reals(
        self,
        stage_wise_reals: List[FeatureGrid],
        hem_rad: float,
        render_pose: CameraPose,
        output_dir: Path,
    ) -> None:
        rendered_stage_wise_real_colours = []
        rendered_stage_wise_real_disparities = []
        for feature_grid in tqdm(stage_wise_reals):
            if self._render_mlp is None:
                with torch.no_grad():
                    # RGBA mode
                    # feature_grid.features = adjust_dynamic_range(
                    #     feature_grid.features, drange_in=(0, 1), drange_out=(-1, 1)
                    # )
                    pass
            rendered_output = self.render_random_sample(
                hem_rad=hem_rad,
                camera_pose=render_pose,
                camera_intrinsics=self._render_params.camera_intrinsics,
                scene_bounds=self._render_params.scene_bounds,
                feature_grid=feature_grid,
            )
            rendered_stage_wise_real_colours.append(rendered_output.colour)
            rendered_stage_wise_real_disparities.append(rendered_output.disparity)
        colour_samples = torch.stack(rendered_stage_wise_real_colours, dim=0)
        disparity_samples = torch.stack(rendered_stage_wise_real_disparities, dim=0)
        colour_sheet = make_grid(
            colour_samples.permute(0, 3, 1, 2), nrow=len(colour_samples)
        ).permute(1, 2, 0)
        disparity_sheet = make_grid(
            disparity_samples.permute(0, 3, 1, 2), nrow=len(disparity_samples)
        ).permute(1, 2, 0)

        # postprocess:
        colour_sheet = to8b(colour_sheet.cpu().numpy())
        disparity_sheet = postprocess_disparity_map(disparity_sheet.numpy()[..., 0])

        # attach them together:
        rendered_sample = np.concatenate([colour_sheet, disparity_sheet], axis=0)

        # save the rendered_sample
        imageio.imwrite(output_dir / f"stage_wise_real_samples.png", rendered_sample)

    def _render_reconstruction_feedback(
        self,
        stage: int,
        camera_hemisphere_radius: float,
        render_feedback_pose: CameraPose,
        recon_logs_dir: Path,
        global_step: int,
    ) -> None:
        recon_render = self.render_reconstruction(
            stage=stage,
            hem_rad=camera_hemisphere_radius,
            camera_pose=render_feedback_pose,
            camera_intrinsics=self._render_params.camera_intrinsics,
            scene_bounds=self._render_params.scene_bounds,
        )
        imageio.imwrite(
            recon_logs_dir / f"colour_stage_{stage}_iter_{global_step}.png",
            to8b(recon_render.colour.numpy()),
        )

        imageio.imwrite(
            recon_logs_dir / f"disparity_stage_{stage}_iter_{global_step}.png",
            postprocess_disparity_map(recon_render.disparity.squeeze().numpy()),
        )

    def _render_gan_feedback(
        self,
        gan_feedback_noise: Tensor,
        stage: int,
        camera_hemisphere_radius: float,
        render_feedback_pose: CameraPose,
        gan_logs_dir: Path,
        global_step: int,
    ) -> None:
        rendered_colour_samples, rendered_disparity_samples = [], []
        log.info("Rendering GAN based feedback ...")
        for noise_sample in tqdm(gan_feedback_noise):
            rendered_random_sample = self.render_random_sample(
                stage=stage,
                hem_rad=camera_hemisphere_radius,
                camera_pose=render_feedback_pose,
                camera_intrinsics=self._render_params.camera_intrinsics,
                scene_bounds=self._render_params.scene_bounds,
                random_noise=noise_sample[None, ...],
            )
            rendered_colour_samples.append(rendered_random_sample.colour)
            rendered_disparity_samples.append(rendered_random_sample.disparity)

        # create sample sheets of the rendered colours and disparities
        # fmt: off
        rendered_disparity_samples = torch.stack(rendered_disparity_samples).permute(0, 3, 1, 2)
        rendered_colour_samples = torch.stack(rendered_colour_samples).permute(0, 3, 1, 2)
        nrow = int(np.ceil(np.sqrt(len(gan_feedback_noise))))
        colour_sheet = make_grid(rendered_colour_samples, nrow=nrow, padding=0).permute(1, 2, 0)
        disparity_sheet = make_grid(rendered_disparity_samples, nrow=nrow, padding=0).permute(1, 2, 0)
        # fmt: on

        imageio.imwrite(
            gan_logs_dir / f"colour_stage_{stage}_iter_{global_step}.png",
            to8b(colour_sheet.numpy()),
        )
        imageio.imwrite(
            gan_logs_dir / f"disparity_stage_{stage}_iter_{global_step}.png",
            postprocess_disparity_map(disparity_sheet.numpy()[..., 0]),
        )

    # noinspection PyUnresolvedReferences
    def _render_synchronized_patches(
        self,
        features_real: Tensor,
        features_fake: Tensor,
        camera_intrinsics: CameraIntrinsics,
        hem_rad: float,
        scene_bounds: SceneBounds,
        patch_size: int,
        patch_batch_size: int,
        voxel_size: VoxelSize,
    ) -> Tuple[Tensor, Tensor]:
        # TODO: refactor this and the `_render_random_patches`
        #  methods. There is a bit of code that can be reused

        height, width, _ = camera_intrinsics
        assert (patch_size <= height) and (
            patch_size <= width
        ), f"Patch size can't be greater than CameraIntrinsics"

        # create the real and the fake feature-grids:
        feature_grid_real = FeatureGrid(
            features=features_real[0],
            voxel_size=voxel_size,
            tunable=False,
            preactivation=torch.nn.Identity(),
        )
        feature_grid_fake = FeatureGrid(
            features=features_fake[0],
            voxel_size=voxel_size,
            tunable=False,
            preactivation=torch.nn.Identity(),
        )

        # obtain random camera poses on the hemisphere
        # please note that we use the same poses for both real
        # and fake feature-grids
        random_yaws = np.random.uniform(low=0, high=360, size=patch_batch_size)
        random_pitches = np.random.uniform(low=-90, high=0, size=patch_batch_size)
        random_hemispherical_poses = [
            pose_spherical(yaw, pitch, hem_rad, device=self._device)
            for (yaw, pitch) in zip(random_yaws, random_pitches)
        ]

        # cast rays for the random poses
        casted_rays_list = []
        for pose in random_hemispherical_poses:
            casted_rays = cast_rays(camera_intrinsics, pose, device=self._device)
            casted_rays_list.append(casted_rays)
        casted_rays = reshape_and_rebuild_flat_rays(casted_rays_list, camera_intrinsics)
        full_rays_tensor = torch.cat(
            [casted_rays.origins, casted_rays.directions], dim=-1
        )

        # cut random patches of rays from this full_rays tensor:
        random_height_indices = torch.randint(
            low=0, high=(height - patch_size + 1), size=(patch_batch_size,)
        )
        random_width_indices = torch.randint(
            low=0, high=(width - patch_size + 1), size=(patch_batch_size,)
        )
        selected_ray_crops = torch.stack(
            [
                full_rays_tensor[
                    patch_num,
                    low_height : low_height + patch_size,
                    low_width : low_width + patch_size,
                    :,
                ]
                for (patch_num, (low_height, low_width)) in enumerate(
                    zip(random_height_indices, random_width_indices)
                )
            ],
            dim=0,
        )
        flat_selected_rays = Rays(
            origins=selected_ray_crops.reshape(-1, selected_ray_crops.shape[-1])[
                :, :NUM_COORD_DIMENSIONS
            ],
            directions=selected_ray_crops.reshape(-1, selected_ray_crops.shape[-1])[
                :, NUM_COORD_DIMENSIONS:
            ],
        )

        # render the rays differentiably (in batches)
        def specialized_collate_fn(
            rendered_chunks: Union[List[RenderOut], List[Tuple[RenderOut, RenderOut]]]
        ) -> RenderOut:
            if isinstance(rendered_chunks[0], RenderOut):
                return collate_rendered_output(rendered_chunks)
            else:
                fine_rendered_chunks = [chunk[-1] for chunk in rendered_chunks]
                return collate_rendered_output(fine_rendered_chunks)

        ray_chunk_size = (
            patch_batch_size * (patch_size * patch_size)
            if self._render_mlp is None
            else self._render_params.num_rays_chunk
        )

        points_chunk_size = (
            patch_batch_size
            * (patch_size * patch_size * self._render_params.num_samples_per_ray)
            if self._render_mlp is None
            else self._render_params.num_points_chunk
        )

        render_function_real, render_function_fake = [
            partial(
                render_feature_grid,
                num_samples=self._render_params.num_samples_per_ray,
                feature_grid=feature_grid,
                scene_bounds=scene_bounds,
                point_processor_network=self._render_mlp,
                secondary_point_processor_network=self._fine_render_mlp,
                num_samples_fine=self._render_params.num_fine_samples_per_ray,
                chunk_size=points_chunk_size,
                perturb_sampled_points=self._render_params.perturb_sampled_points,
                raw2alpha=self._transmittance_behaviour,
                colour_producer=self._colour_producer,
            )
            for feature_grid in (feature_grid_real, feature_grid_fake)
        ]

        # use a batchified rendering mechanism if an MLP is
        # involved, otherwise just go for it :)
        if self._render_mlp is not None:
            # note that the real grid is rendered without
            # creating a pytorch graph
            with torch.no_grad():
                rendered_pixels_real = batchify(
                    processor_fn=render_function_real,
                    collate_fn=specialized_collate_fn,
                    chunk_size=ray_chunk_size,
                )(flat_selected_rays)
            rendered_pixels_fake = batchify(
                processor_fn=render_function_real,
                collate_fn=specialized_collate_fn,
                chunk_size=ray_chunk_size,
            )(flat_selected_rays)
        else:
            with torch.no_grad():
                rendered_pixels_real, _ = render_function_real(flat_selected_rays)
            rendered_pixels_fake, _ = render_function_fake(flat_selected_rays)

        real_patches, fake_patches = [
            rendered_pixels.colour.reshape(
                patch_batch_size, patch_size, patch_size, -1
            ).permute(0, 3, 1, 2)
            for rendered_pixels in (rendered_pixels_real, rendered_pixels_fake)
        ]

        return real_patches, fake_patches

    def _render_random_patches(
        self,
        features: Tensor,
        camera_intrinsics: CameraIntrinsics,
        hem_rad: float,
        scene_bounds: SceneBounds,
        patch_size: int,
        patch_batch_size: int,
        voxel_size: VoxelSize,
    ) -> Tensor:
        # with profile(
        #     activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True
        # ) as prof:
        #     with record_function("patch_render"):
        height, width, _ = camera_intrinsics
        assert (patch_size <= height) and (
            patch_size <= width
        ), f"Patch size can't be greater than CameraIntrinsics"

        # create a feature-grid object from the given features:
        if self._render_mlp is None:
            # features = adjust_dynamic_range(
            #     features, drange_in=(-1, 1), drange_out=(0, 1)
            # )
            pass
        feature_grid = FeatureGrid(
            features=features[0],
            voxel_size=voxel_size,
            tunable=False,
            preactivation=torch.nn.Identity(),
        )

        # obtain random camera poses on the hemisphere
        random_yaws = np.random.uniform(low=0, high=360, size=patch_batch_size)
        random_pitches = np.random.uniform(low=-90, high=0, size=patch_batch_size)
        random_hemispherical_poses = [
            pose_spherical(yaw, pitch, hem_rad, device=self._device)
            for (yaw, pitch) in zip(random_yaws, random_pitches)
        ]

        # cast rays for the random poses
        casted_rays_list = []
        for pose in random_hemispherical_poses:
            casted_rays = cast_rays(camera_intrinsics, pose, device=self._device)
            casted_rays_list.append(casted_rays)
        casted_rays = reshape_and_rebuild_flat_rays(casted_rays_list, camera_intrinsics)
        full_rays_tensor = torch.cat(
            [casted_rays.origins, casted_rays.directions], dim=-1
        )

        # cut random patches of rays from this full_rays tensor:
        random_height_indices = torch.randint(
            low=0, high=(height - patch_size + 1), size=(patch_batch_size,)
        )
        random_width_indices = torch.randint(
            low=0, high=(width - patch_size + 1), size=(patch_batch_size,)
        )
        selected_ray_crops = torch.stack(
            [
                full_rays_tensor[
                    patch_num,
                    low_height : low_height + patch_size,
                    low_width : low_width + patch_size,
                    :,
                ]
                for (patch_num, (low_height, low_width)) in enumerate(
                    zip(random_height_indices, random_width_indices)
                )
            ],
            dim=0,
        )
        flat_selected_rays = Rays(
            origins=selected_ray_crops.reshape(-1, selected_ray_crops.shape[-1])[
                :, :NUM_COORD_DIMENSIONS
            ],
            directions=selected_ray_crops.reshape(-1, selected_ray_crops.shape[-1])[
                :, NUM_COORD_DIMENSIONS:
            ],
        )

        # render the rays differentiably (in batches)
        def specialized_collate_fn(
            rendered_chunks: Union[List[RenderOut], List[Tuple[RenderOut, RenderOut]]]
        ) -> RenderOut:
            if isinstance(rendered_chunks[0], RenderOut):
                return collate_rendered_output(rendered_chunks)
            else:
                fine_rendered_chunks = [chunk[-1] for chunk in rendered_chunks]
                return collate_rendered_output(fine_rendered_chunks)

        ray_chunk_size = (
            patch_batch_size * (patch_size * patch_size)
            if self._render_mlp is None
            else self._render_params.num_rays_chunk
        )

        points_chunk_size = (
            patch_batch_size
            * (patch_size * patch_size * self._render_params.num_samples_per_ray)
            if self._render_mlp is None
            else self._render_params.num_points_chunk
        )

        render_function = partial(
            render_feature_grid,
            num_samples=self._render_params.num_samples_per_ray,
            feature_grid=feature_grid,
            scene_bounds=scene_bounds,
            point_processor_network=self._render_mlp,
            secondary_point_processor_network=self._fine_render_mlp,
            num_samples_fine=self._render_params.num_fine_samples_per_ray,
            chunk_size=points_chunk_size,
            perturb_sampled_points=self._render_params.perturb_sampled_points,
            raw2alpha=self._transmittance_behaviour,
            colour_producer=self._colour_producer,
        )
        if self._render_mlp is not None:
            rendered_pixels = batchify(
                processor_fn=render_function,
                collate_fn=specialized_collate_fn,
                chunk_size=ray_chunk_size,
            )(flat_selected_rays)
        else:
            rendered_pixels, _ = render_function(flat_selected_rays)

        # print(
        #     prof.key_averages(group_by_stack_n=5).table(
        #         sort_by="cpu_time_total", row_limit=10
        #     )
        # )
        # extract the patches from the rendered output:
        # noinspection PyUnresolvedReferences
        return rendered_pixels.colour.reshape(
            patch_batch_size, patch_size, patch_size, -1
        ).permute(0, 3, 1, 2)

        # return torch.randn(
        #     patch_batch_size,
        #     NUM_COLOUR_CHANNELS,
        #     patch_size,
        #     patch_size,
        #     device=self._device,
        # )

    def get_feature_grid(
        self,
        hem_rad: float,
        scene_bounds: SceneBounds,
        random_noise: Optional[Tensor] = None,
        stage: Optional[int] = None,
        apply_noise: bool = True,
        use_fixed_noise: bool = False,
    ) -> FeatureGrid:
        feature_volume = self._generator(
            input_noise=random_noise,
            stage=stage,
            apply_noise=apply_noise,
            use_intermediate_fixed_noise=use_fixed_noise,
        )
        voxel_size = get_voxel_size_from_scene_bounds_and_hem_rad(
            hem_rad, grid_dim=max(feature_volume.shape[2:]), scene_bounds=scene_bounds
        )
        if random_noise.shape[2:] != self._reconstruction_noise.shape[2:]:
            # we need to compute the voxel size differently:
            voxel_size = get_voxel_size_from_scene_bounds_and_hem_rad(
                hem_rad,
                grid_dim=max(self._output_grid_resolution),
                scene_bounds=scene_bounds,
            )

        return FeatureGrid(
            features=feature_volume[0],
            voxel_size=voxel_size,
            # grid will always be at the center in the generative modelling :)
            # so not providing the grid_location
            preactivation=torch.nn.Identity(),
        )

    def render_reconstruction(
        self,
        hem_rad: float,
        camera_pose: CameraPose,
        camera_intrinsics: CameraIntrinsics,
        scene_bounds: SceneBounds,
        # use the following parameter to cache the feature grid
        feature_grid: Optional[FeatureGrid] = None,
        stage: Optional[int] = None,
    ) -> RenderOut:
        if feature_grid is None:
            with torch.no_grad():
                feature_grid = self.get_feature_grid(
                    hem_rad=hem_rad,
                    scene_bounds=scene_bounds,
                    random_noise=self._reconstruction_noise,
                    stage=stage,
                    apply_noise=False,
                )
        if self._render_mlp is None:
            with torch.no_grad():
                # feature_grid.features = adjust_dynamic_range(
                #     feature_grid.features,
                #     drange_in=(-1, 1),
                #     drange_out=(0, 1),
                #     slack=True,
                # )
                pass

        rendered_output = render_image_in_chunks(
            cam_intrinsics=camera_intrinsics,
            camera_pose=camera_pose,
            num_rays_chunk=self._render_params.num_rays_chunk,
            num_samples_per_ray=self._render_params.num_samples_per_ray,
            num_samples_fine=self._render_params.num_fine_samples_per_ray,
            feature_grid=feature_grid,
            processor_mlp=self._trainable_render_mlp,
            secondary_processor_mlp=self._fine_render_mlp,
            scene_bounds=scene_bounds,
            density_noise_std=self._render_params.density_noise_std,
            perturb_sampled_points=self._render_params.perturb_sampled_points,
            colour_producer=self._colour_producer,
            raw2alpha=self._transmittance_behaviour,
            device=self._device,
        )

        if isinstance(rendered_output, RenderOut):
            return rendered_output
        else:
            return rendered_output[-1]  # only return the fine output

    # def fast_render_multiple_random_samples(
    #     self,
    #     hem_rad: float,
    #     camera_poses: List[CameraPose],
    #     camera_intrinsics: CameraIntrinsics,
    #     scene_bounds: SceneBounds,
    #     stage: Optional[int] = None,
    # ) -> Tensor:
    #     assert (
    #         self._render_mlp is None
    #     ), f"Can't use the real-time renderer when MLP is involved"
    #
    #     # cast rays for all the camera poses:
    #     casted_rays_list = []
    #     for pose in camera_poses:
    #         casted_rays = cast_rays(camera_intrinsics, pose, device=self._device)
    #         casted_rays_list.append(casted_rays)
    #     casted_rays = collate_rays(casted_rays_list)
    #
    #     with torch.no_grad():
    #         feature_grid = self.get_feature_grid(hem_rad, scene_bounds, stage=stage)
    #         torch.cuda.empty_cache()
    #
    #         render_function = partial(
    #             render_feature_grid,
    #             num_samples=self._render_params.num_samples_per_ray,
    #             feature_grid=feature_grid,
    #             scene_bounds=scene_bounds,
    #             point_processor_network=self._render_mlp,
    #             secondary_point_processor_network=self._fine_render_mlp,
    #             num_samples_fine=self._render_params.num_fine_samples_per_ray,
    #             chunk_size=512,
    #             perturb_sampled_points=self._render_params.perturb_sampled_points,
    #             raw2alpha=self._transmittance_behaviour,
    #             colour_producer=self._colour_producer,
    #         )
    #         rendered_pixels = render_function(casted_rays)
    #
    #     return rendered_pixels.colour.reshape(
    #         len(camera_poses), camera_intrinsics.height, camera_intrinsics.width, -1
    #     )

    def render_random_sample(
        self,
        hem_rad: float,
        camera_pose: CameraPose,
        camera_intrinsics: CameraIntrinsics,
        scene_bounds: SceneBounds,
        random_noise: Optional[Tensor] = None,
        # use the following parameter to cache the feature grid
        feature_grid: Optional[FeatureGrid] = None,
        stage: Optional[int] = None,
        use_fixed_noise: bool = False,
    ) -> RenderOut:
        if feature_grid is None:
            with torch.no_grad():
                feature_grid = self.get_feature_grid(
                    hem_rad=hem_rad,
                    scene_bounds=scene_bounds,
                    random_noise=random_noise,
                    stage=stage,
                    apply_noise=True,
                    use_fixed_noise=use_fixed_noise,
                )
        if self._render_mlp is None:
            with torch.no_grad():
                # feature_grid.features = adjust_dynamic_range(
                #     feature_grid.features,
                #     drange_in=(-1, 1),
                #     drange_out=(0, 1),
                #     slack=True,
                # )
                pass

        rendered_output = render_image_in_chunks(
            cam_intrinsics=camera_intrinsics,
            camera_pose=camera_pose,
            num_rays_chunk=self._render_params.num_rays_chunk,
            num_samples_per_ray=self._render_params.num_samples_per_ray,
            num_samples_fine=self._render_params.num_fine_samples_per_ray,
            feature_grid=feature_grid,
            processor_mlp=self._trainable_render_mlp,
            secondary_processor_mlp=self._fine_render_mlp,
            scene_bounds=scene_bounds,
            density_noise_std=self._render_params.density_noise_std,
            perturb_sampled_points=self._render_params.perturb_sampled_points,
            colour_producer=self._colour_producer,
            raw2alpha=self._transmittance_behaviour,
            device=self._device,
        )

        if isinstance(rendered_output, RenderOut):
            return rendered_output
        else:
            return rendered_output[-1]  # only return the fine output

    def _decode_real_fake_at_random_locations_synchronously(
        self,
        real_grid: Tensor,
        recon_grid: Tensor,
        real_render_mlp: RenderMLP,
        fake_render_mlp: RenderMLP,
    ) -> Tuple[Tensor, Tensor]:
        _, num_channels, x_dim, y_dim, z_dim = recon_grid.shape
        x_size, y_size, z_size = (2 / (x_dim - 1)), (2 / (y_dim - 1)), (2 / (z_dim - 1))

        regular_points = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    -1, 1 - x_size, x_dim - 1, dtype=torch.float32, device=self._device
                ),
                torch.linspace(
                    -1, 1 - y_size, y_dim - 1, dtype=torch.float32, device=self._device
                ),
                torch.linspace(
                    -1, 1 - z_size, z_dim - 1, dtype=torch.float32, device=self._device
                ),
            ),
            dim=-1,
        )

        jitter_offset = (
            torch.rand(
                size=(x_dim - 1, y_dim - 1, z_dim - 1, NUM_COORD_DIMENSIONS),
                device=self._device,
            )
            * torch.tensor(
                [x_size, y_size, z_size], dtype=torch.float32, device=self._device
            )[None, None, None, :]
        )

        jittered_points = regular_points + jitter_offset

        real_point_features = grid_sample(
            real_grid.permute(0, 1, 4, 3, 2),
            jittered_points[None, ...],
            align_corners=False,
        ).permute(0, 2, 3, 4, 1)
        real_flat_point_features = real_point_features.reshape(-1, num_channels)

        fake_point_features = grid_sample(
            recon_grid.permute(0, 1, 4, 3, 2),
            jittered_points[None, ...],
            align_corners=False,
        ).permute(0, 2, 3, 4, 1)
        fake_flat_point_features = fake_point_features.reshape(-1, num_channels)

        # create viewing directions
        view_dirs = torch.randn(
            (1, NUM_COORD_DIMENSIONS), dtype=torch.float32, device=self._device
        ).repeat(len(real_flat_point_features), 1)

        # make sure that the camera always look down:
        view_dirs[..., -1] = -torch.abs(view_dirs[..., -1])
        view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)

        real_rgba_values = batchify(
            processor_fn=real_render_mlp,
            collate_fn=partial(torch.cat, dim=0),
            chunk_size=self._render_params.num_points_chunk,
            verbose=False,
        )(torch.cat([real_flat_point_features, view_dirs], dim=-1))

        real_rgb_values, real_a_values = (
            real_rgba_values[:, :NUM_COLOUR_CHANNELS],
            real_rgba_values[:, NUM_COLOUR_CHANNELS:],
        )
        real_rgb_values = self._colour_producer(real_rgb_values).reshape(
            1, x_dim - 1, y_dim - 1, z_dim - 1, -1
        )
        real_a_values = self._transmittance_behaviour(
            real_a_values, torch.ones_like(real_a_values, device=self._device)
        ).reshape(1, x_dim - 1, y_dim - 1, z_dim - 1, -1)

        real_decoded_values = torch.cat(
            [real_rgb_values, real_a_values], dim=-1
        ).permute(0, 4, 1, 2, 3)

        fake_rgba_values = batchify(
            processor_fn=fake_render_mlp,
            collate_fn=partial(torch.cat, dim=0),
            chunk_size=self._render_params.num_points_chunk,
            verbose=False,
        )(torch.cat([fake_flat_point_features, view_dirs], dim=-1))

        fake_rgb_values, fake_a_values = (
            fake_rgba_values[:, :NUM_COLOUR_CHANNELS],
            fake_rgba_values[:, NUM_COLOUR_CHANNELS:],
        )
        fake_rgb_values = self._colour_producer(fake_rgb_values).reshape(
            1, x_dim - 1, y_dim - 1, z_dim - 1, -1
        )
        fake_a_values = self._transmittance_behaviour(
            fake_a_values, torch.ones_like(fake_a_values, device=self._device)
        ).reshape(1, x_dim - 1, y_dim - 1, z_dim - 1, -1)

        fake_decoded_values = torch.cat(
            [fake_rgb_values, fake_a_values], dim=-1
        ).permute(0, 4, 1, 2, 3)

        return real_decoded_values, fake_decoded_values

    def _decode_feature_grid_at_random_locations(
        self,
        features: Tensor,
        render_mlp: RenderMLP,
    ) -> Tensor:
        _, num_channels, x_dim, y_dim, z_dim = features.shape
        x_size, y_size, z_size = (2 / (x_dim - 1)), (2 / (y_dim - 1)), (2 / (z_dim - 1))

        regular_points = torch.stack(
            torch.meshgrid(
                torch.linspace(
                    -1, 1 - x_size, x_dim - 1, dtype=torch.float32, device=self._device
                ),
                torch.linspace(
                    -1, 1 - y_size, y_dim - 1, dtype=torch.float32, device=self._device
                ),
                torch.linspace(
                    -1, 1 - z_size, z_dim - 1, dtype=torch.float32, device=self._device
                ),
            ),
            dim=-1,
        )

        raw_jitter_offset = torch.rand(
            size=(1, 1, 1, NUM_COORD_DIMENSIONS),
            device=self._device,
        )
        jitter_offset = (
            raw_jitter_offset
            * torch.tensor(
                [x_size, y_size, z_size], dtype=torch.float32, device=self._device
            )[None, None, None, :]
        )

        jittered_points = regular_points + jitter_offset

        point_features = grid_sample(
            features.permute(0, 1, 4, 3, 2),
            jittered_points[None, ...],
            align_corners=False,
        ).permute(0, 2, 3, 4, 1)
        flat_point_features = point_features.reshape(-1, num_channels)

        # create viewing directions
        view_dirs = torch.randn(
            (1, NUM_COORD_DIMENSIONS), dtype=torch.float32, device=self._device
        ).repeat(len(flat_point_features), 1)
        # make sure that the camera always look down:
        view_dirs[..., -1] = -torch.abs(view_dirs[..., -1])
        view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)

        rgba_values = batchify(
            processor_fn=render_mlp,
            collate_fn=partial(torch.cat, dim=0),
            chunk_size=self._render_params.num_points_chunk,
            verbose=False,
        )(torch.cat([flat_point_features, view_dirs], dim=-1))

        rgb_values, a_values = (
            rgba_values[:, :NUM_COLOUR_CHANNELS],
            rgba_values[:, NUM_COLOUR_CHANNELS:],
        )
        rgb_values = self._colour_producer(rgb_values).reshape(
            1, x_dim - 1, y_dim - 1, z_dim - 1, -1
        )
        a_values = self._transmittance_behaviour(
            a_values, torch.ones_like(a_values, device=self._device)
        ).reshape(1, x_dim - 1, y_dim - 1, z_dim - 1, -1)

        decoded_values = torch.cat([rgb_values, a_values], dim=-1)

        # also concatenate the jitter offset and the view_dirs to the decoded_values
        reshaped_jitter_offset = jitter_offset[None, ...].repeat(
            *decoded_values.shape[:-1], 1
        )
        reshaped_view_dirs = view_dirs.reshape(*decoded_values.shape[:-1], -1)

        decoded_values = torch.cat(
            [decoded_values, reshaped_jitter_offset, reshaped_view_dirs], dim=-1
        ).permute(0, 4, 1, 2, 3)

        return decoded_values

    def train(
        self,
        training_feature_grid: FeatureGrid,
        camera_hemisphere_radius: float,
        discriminator_receptive_field: Tuple[int, int, int] = (11, 11, 11),
        discriminator_num_layers: int = 5,
        num_iterations_per_stage: int = 2000,
        num_dis_steps: int = 3,
        num_gen_steps: int = 3,
        g_lrate: float = 0.003,
        d_lrate: float = 0.003,
        # Parameters related to Adversarial training:
        use_3d_discriminator: bool = True,
        use_2d_discriminator: bool = True,
        apply_3d_reconstruction_loss: bool = True,
        apply_2d_reconstruction_loss: bool = False,
        virtual_camera_size: Optional[int] = None,
        patch_size: int = 32,
        patch_batch_size: int = 8,
        adv_2d_loss_lambda: float = 1.0,
        # rest of the parameters
        threed_wgan_gp_lambda: float = 1.0,
        threed_wgan_gp_drift_penalty: float = 0.0,
        twod_wgan_gp_lambda: float = 1.0,
        twod_wgan_gp_drift_penalty: float = 0.0,
        lr_decay_steps: Optional[int] = None,
        lr_decay_gamma: float = 0.9,
        threed_recon_loss_alpha: float = 10.0,
        twod_recon_loss_alpha: float = 100.0,
        num_feedback_samples: int = 6,
        feedback_frequency: int = 500,
        loss_feedback_frequency: int = 100,
        save_frequency: int = 500,
        output_dir: Path = Path(__file__).parent.absolute() / "logs",
        fast_debug_mode: bool = False,
    ) -> None:

        if not (use_2d_discriminator or use_3d_discriminator):
            log.info(
                "Not using any discriminators, so the training would only perform reconstruction"
            )

        # setup the stage-wise training real feature-grids:
        stage_wise_reals = self._setup_stage_wise_training_reals(training_feature_grid)
        stage_wise_camera_intrinsics = self._setup_stage_wise_camera_intrinsics(
            virtual_camera_intrinsics=CameraIntrinsics(
                virtual_camera_size, virtual_camera_size, virtual_camera_size
            )
            if virtual_camera_size is not None
            else None
        )

        # create a render feedback pose
        render_feedback_pose = pose_spherical(45.0, -30.0, camera_hemisphere_radius)

        # save a sample for stage wise reals:
        if not fast_debug_mode:
            log.info("Rendering stage-wise real feature-grid samples")
            self._render_stage_wise_reals(
                stage_wise_reals,
                hem_rad=camera_hemisphere_radius,
                render_pose=render_feedback_pose,
                output_dir=output_dir,
            )

        # setup output directories:
        model_dir, logs_dir = output_dir / "saved_models", output_dir / "training_logs"
        tensorboard_dir = logs_dir / "tensorboard"
        render_dir = logs_dir / "rendered_output"
        for directory in (model_dir, logs_dir, tensorboard_dir, render_dir):
            directory.mkdir(exist_ok=True, parents=True)

        # tensorboard writer:
        tensorboard_writer = SummaryWriter(tensorboard_dir)

        # create fixed noise vectors for gan feedback:
        gan_feedback_noise = torch.randn(
            num_feedback_samples,
            *self._reconstruction_noise.shape[1:],
            device=self._device,
        )

        # Define a gan_loss:
        gan_loss_3d = WganGPGanLoss(
            gp_lambda=threed_wgan_gp_lambda, drift_lambda=threed_wgan_gp_drift_penalty
        )
        gan_loss_2d = WganGPGanLoss(
            gp_lambda=twod_wgan_gp_lambda, drift_lambda=twod_wgan_gp_drift_penalty
        )

        log.info("!! Beginning Training !!")
        global_step = 0

        prev_stage_2d_discriminator, prev_stage_3d_discriminator = None, None
        current_stage_2d_discriminator, current_stage_3d_discriminator = None, None
        prev_stage_gen_block = None
        for stage in range(1, self._num_stages + 1):
            log.info(f"Starting new stage: {stage}")

            # setup directories for the current stage:
            recon_logs_dir = render_dir / "recon" / str(stage)
            gan_logs_dir = render_dir / "gan" / str(stage)
            gan_2d_patches_logs_dir = gan_logs_dir / "patches"
            for directory in (recon_logs_dir, gan_logs_dir, gan_2d_patches_logs_dir):
                directory.mkdir(exist_ok=True, parents=True)

            # select the current_stage_num_intermediate_features:
            current_stage_num_intermediate_features = self._intermediate_features[
                stage - 1
            ]

            # setup current stage real feature-grid
            current_stage_real = stage_wise_reals[stage - 1]
            log.info(
                f"Current Stage's feature_grid resolution: {current_stage_real.features.shape}"
            )

            # obtain the current stage's camera intrinsics:
            current_stage_camera_intrinsics = stage_wise_camera_intrinsics[stage - 1]
            log.info(
                f"Current Stage's 2D camera intrinsics: {current_stage_camera_intrinsics}"
            )

            # load previous stage's gen_block to initialize current stage:
            if (
                prev_stage_gen_block is not None
                and self._intermediate_features[stage - 2]
                == current_stage_num_intermediate_features
            ):
                self._generator.load_block_at_stage(prev_stage_gen_block, stage)
            else:
                log.info(
                    f"Skipped loading of generator weights from previous stage due to "
                    f"size mismatch: previous stage size {self._intermediate_features[stage - 2]} "
                    f"current stage size: {current_stage_num_intermediate_features}"
                )

            # ----------------------------------------------------------------------------------------
            # create new discriminators (and their optimizers) for the current stage
            # Note that we can use ex-nor combinations of the 2d and 3d discriminators
            # ----------------------------------------------------------------------------------------
            if use_3d_discriminator:
                in_features = (
                    NUM_RGBA_CHANNELS + (2 * NUM_COORD_DIMENSIONS)
                    if self._render_mlp is not None
                    else self._feature_dims
                )
                current_stage_3d_discriminator = Thre3dSinGanDiscriminatorDS(
                    in_features=in_features,
                    required_receptive_field=discriminator_receptive_field,
                    num_layers=discriminator_num_layers,
                    intermediate_features=current_stage_num_intermediate_features,
                    use_eql=self._use_eql,
                ).to(self._device)
                if (
                    prev_stage_3d_discriminator is not None
                    and self._intermediate_features[stage - 2]
                    == current_stage_num_intermediate_features
                ):
                    # load the previous stage discriminator's weights into this stage's discriminator:
                    current_stage_3d_discriminator.load_state_dict(
                        prev_stage_3d_discriminator.state_dict()
                    )
            if use_2d_discriminator:
                # twod_discriminator_depth = int(np.log(patch_size) / np.log(2))
                # current_stage_2d_discriminator = get_convolutional_discriminator(
                #     depth=twod_discriminator_depth,
                #     latent_size=4 * current_stage_num_intermediate_features,
                #     fmap_max=4 * current_stage_num_intermediate_features,
                #     fmap_base=4 * current_stage_num_intermediate_features,
                #     fmap_min=current_stage_num_intermediate_features,
                #     use_eql=self._use_eql,
                # ).to(self._device)
                current_stage_2d_discriminator = TwodSinGanDiscriminator(
                    in_channels=NUM_COLOUR_CHANNELS,
                    intermediate_channels=current_stage_num_intermediate_features,
                    use_eql=self._use_eql,
                ).to(self._device)
                if (
                    prev_stage_2d_discriminator is not None
                    and self._intermediate_features[stage - 2]
                    == current_stage_num_intermediate_features
                ):
                    current_stage_2d_discriminator.load_state_dict(
                        prev_stage_2d_discriminator.state_dict()
                    )

            log.info(
                f"New 3D Discriminator Architecture: {current_stage_3d_discriminator}"
            )
            log.info(
                f"New 2D Discriminator Architecture: {current_stage_2d_discriminator}"
            )
            # ---------------------------------------------------------------------------------------

            # setup optimizers for generator and discriminator:
            # generator optimizer:
            optimizer_generator = torch.optim.Adam(
                params=[
                    {
                        "params": self._generator.get_block_at_stage(
                            stage
                        ).parameters(),
                        "lr": g_lrate,
                    },
                    # {
                    #     "params": self._trainable_render_mlp.parameters()
                    #     if self._trainable_render_mlp is not None
                    #     else [],
                    #     "lr": 0.1 * g_lrate,
                    # },
                ],
                lr=g_lrate,
                betas=(0.5, 0.999),
            )
            gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_generator, gamma=lr_decay_gamma
            )

            # discriminator optimizer:
            optimizer_discriminator = torch.optim.Adam(
                params=[
                    {
                        "params": current_stage_3d_discriminator.parameters()
                        if current_stage_3d_discriminator is not None
                        else [],
                        "lr": d_lrate,
                    },
                    {
                        "params": current_stage_2d_discriminator.parameters()
                        if current_stage_2d_discriminator is not None
                        else [],
                        "lr": d_lrate,
                    },
                ],
                betas=(0.5, 0.999),
            )
            dis_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_discriminator, gamma=lr_decay_gamma
            )

            # prepare the real feature grid for this stage's training:
            real_feature_grid = current_stage_real.features.permute(3, 0, 1, 2)[
                None, ...
            ]
            if self._render_mlp is None:
                with torch.no_grad():
                    # real_feature_grid = adjust_dynamic_range(
                    #     real_feature_grid, drange_in=(0, 1), drange_out=(-1, 1)
                    # )
                    pass

            rmse_recon_loss = None
            if stage > 1:
                # compute the noise_amp for the generator at the current stage:
                with torch.no_grad():
                    reconstructed_feature_grid = self._generator(
                        self._reconstruction_noise, stage - 1, apply_noise=False
                    )
                    if self._trainable_render_mlp is not None:
                        rmse_recon_loss = torch.sqrt(
                            mse_loss(
                                *self._decode_real_fake_at_random_locations_synchronously(
                                    real_feature_grid,
                                    reconstructed_feature_grid,
                                    self._render_mlp,
                                    self._trainable_render_mlp,
                                )
                            )
                        ).item()
                    else:
                        rmse_recon_loss = torch.sqrt(
                            mse_loss(
                                interpolate(
                                    reconstructed_feature_grid,
                                    size=real_feature_grid.shape[2:],
                                ),
                                real_feature_grid,
                            )
                        ).item()

            for stage_iter in range(1, num_iterations_per_stage + 1):
                # with profile(
                #     activities=[ProfilerActivity.CPU],
                #     record_shapes=True,
                # ) as prof:
                #     with record_function("single_training_step"):
                real_random_patches, fake_random_patches = None, None

                # Always ensure that both the RenderMLPs have gradients turned off
                if self._render_mlp is not None:
                    toggle_grad(self._render_mlp, False)
                if self._fine_render_mlp is not None:
                    toggle_grad(self._fine_render_mlp, False)

                # -----------------------------------------------------------------------
                # Discriminator steps:
                # -----------------------------------------------------------------------
                # This is done to speed up training
                toggle_grad(self._generator, False)
                if current_stage_3d_discriminator is not None:
                    toggle_grad(current_stage_3d_discriminator, True)
                if current_stage_2d_discriminator is not None:
                    toggle_grad(current_stage_2d_discriminator, True)

                dis_stats_list = []
                for _ in range(num_dis_steps):
                    fake_feature_grid = self._generator(
                        stage=stage,
                        apply_noise=True,
                        noise_multiplier=rmse_recon_loss,
                    )

                    # compute the 3D discriminator loss
                    dis_loss_3d, extra_info_3d = (
                        torch.tensor([0.0], device=self._device, requires_grad=True),
                        {},
                    )
                    dis_loss_2d, extra_info_2d = (
                        torch.tensor([0.0], device=self._device, requires_grad=True),
                        {},
                    )
                    if current_stage_3d_discriminator is not None:
                        decoded_real_feature_grid = real_feature_grid
                        decoded_fake_feature_grid = fake_feature_grid
                        if self._render_mlp is not None:
                            with torch.no_grad():
                                decoded_real_feature_grid = (
                                    self._decode_feature_grid_at_random_locations(
                                        real_feature_grid, self._render_mlp
                                    )
                                )

                            decoded_fake_feature_grid = (
                                self._decode_feature_grid_at_random_locations(
                                    fake_feature_grid,
                                    self._trainable_render_mlp,
                                )
                            )

                        dis_loss_3d, extra_info_3d = gan_loss_3d.dis_loss(
                            current_stage_3d_discriminator,
                            decoded_real_feature_grid,
                            decoded_fake_feature_grid,
                            suffix="3d",
                        )

                    # compute the 2D discriminator loss:
                    if current_stage_2d_discriminator is not None:
                        with torch.no_grad():
                            # the real patches don't need to create a pytorch graph
                            real_random_patches = self._render_random_patches(
                                real_feature_grid,
                                camera_intrinsics=current_stage_camera_intrinsics,
                                hem_rad=camera_hemisphere_radius,
                                scene_bounds=self._render_params.scene_bounds,
                                patch_size=patch_size,
                                patch_batch_size=patch_batch_size,
                                voxel_size=current_stage_real.voxel_size,
                            )

                            fake_random_patches = self._render_random_patches(
                                fake_feature_grid,
                                camera_intrinsics=current_stage_camera_intrinsics,
                                hem_rad=camera_hemisphere_radius,
                                scene_bounds=self._render_params.scene_bounds,
                                patch_size=patch_size,
                                patch_batch_size=patch_batch_size,
                                voxel_size=current_stage_real.voxel_size,
                            )

                        dis_loss_2d, extra_info_2d = gan_loss_2d.dis_loss(
                            current_stage_2d_discriminator,
                            real_random_patches,
                            fake_random_patches,
                            suffix="2d",
                        )

                    # compute the total loss:
                    dis_loss = dis_loss_3d + dis_loss_2d

                    # optimization step for the discriminator
                    optimizer_discriminator.zero_grad()
                    dis_loss.backward()
                    optimizer_discriminator.step()

                    # combine all the logs in a single stats_dict
                    extra_info = {}
                    extra_info.update(extra_info_3d)
                    extra_info.update(extra_info_2d)
                    extra_info.update({"dis_loss": dis_loss.item()})
                    dis_stats_list.append(extra_info)

                dis_stats = self._average_stats(dis_stats_list)

                # -----------------------------------------------------------------------
                # Generator steps:
                # -----------------------------------------------------------------------
                # This is done to speed up training
                toggle_grad(self._generator, True)
                if current_stage_3d_discriminator is not None:
                    toggle_grad(current_stage_3d_discriminator, False)
                if current_stage_2d_discriminator is not None:
                    toggle_grad(current_stage_2d_discriminator, False)

                gen_stats_list = []
                for _ in range(num_gen_steps):
                    # -----------------------------------------------------------------------
                    # Reconstruction step:
                    # -----------------------------------------------------------------------
                    reconstructed_feature_grid = self._generator(
                        self._reconstruction_noise, stage, apply_noise=False
                    )
                    real_grid = real_feature_grid

                    if self._render_mlp is not None:
                        # decode real and fake feature grids synchronously (at same jitter locations)
                        (
                            real_grid,
                            reconstructed_feature_grid,
                        ) = self._decode_real_fake_at_random_locations_synchronously(
                            real_grid,
                            reconstructed_feature_grid,
                            self._render_mlp,
                            self._trainable_render_mlp,
                        )

                    recon_loss_3d = mse_loss(reconstructed_feature_grid, real_grid)
                    mse_recon_loss = mse_loss(reconstructed_feature_grid, real_grid)

                    recon_loss_2d = torch.tensor([0.0], device=self._device)
                    if apply_2d_reconstruction_loss:
                        # please note that this loss is computed
                        # only when required in order to save computations
                        real_patches, fake_patches = self._render_synchronized_patches(
                            features_real=real_grid,
                            features_fake=reconstructed_feature_grid,
                            camera_intrinsics=current_stage_camera_intrinsics,
                            hem_rad=camera_hemisphere_radius,
                            scene_bounds=self._render_params.scene_bounds,
                            patch_size=patch_size,
                            patch_batch_size=patch_batch_size,
                            voxel_size=current_stage_real.voxel_size,
                        )
                        recon_loss_2d = mse_loss(real_patches, fake_patches)

                    # -----------------------------------------------------------------------
                    # Adversarial step:
                    # -----------------------------------------------------------------------

                    fake_feature_grid = self._generator(
                        stage=stage,
                        apply_noise=True,
                        noise_multiplier=rmse_recon_loss,
                    )

                    gen_loss_3d, extra_info_3d = (
                        torch.tensor([0.0], device=self._device),
                        {},
                    )
                    gen_loss_2d, extra_info_2d = (
                        torch.tensor([0.0], device=self._device),
                        {},
                    )
                    if current_stage_3d_discriminator is not None:
                        decoded_fake_feature_grid = fake_feature_grid
                        if self._render_mlp is not None:
                            decoded_fake_feature_grid = (
                                self._decode_feature_grid_at_random_locations(
                                    fake_feature_grid,
                                    self._trainable_render_mlp,
                                )
                            )
                        gen_loss_3d, extra_info_3d = gan_loss_3d.gen_loss(
                            current_stage_3d_discriminator,
                            _,
                            decoded_fake_feature_grid,
                        )

                    if current_stage_2d_discriminator is not None:
                        fake_random_patches = self._render_random_patches(
                            fake_feature_grid,
                            camera_intrinsics=current_stage_camera_intrinsics,
                            hem_rad=camera_hemisphere_radius,
                            scene_bounds=self._render_params.scene_bounds,
                            patch_size=patch_size,
                            patch_batch_size=patch_batch_size,
                            voxel_size=current_stage_real.voxel_size,
                        )

                        gen_loss_2d, extra_info_2d = gan_loss_2d.gen_loss(
                            current_stage_2d_discriminator,
                            _,
                            fake_random_patches,
                        )

                    # compute total generator loss
                    recon_loss_3d = (
                        torch.tensor([0.0], device=self._device)
                        if not apply_3d_reconstruction_loss
                        else recon_loss_3d
                    )
                    total_gen_loss = (
                        (threed_recon_loss_alpha * recon_loss_3d)
                        + (twod_recon_loss_alpha * recon_loss_2d)
                        + gen_loss_3d
                        + (adv_2d_loss_lambda * gen_loss_2d)
                    )

                    # optimization step for the generator
                    optimizer_generator.zero_grad()
                    total_gen_loss.backward()
                    optimizer_generator.step()

                    # combine all the logs in a single stats_dict
                    extra_info = {}
                    extra_info.update(extra_info_3d)
                    extra_info.update(extra_info_3d)
                    extra_info.update(
                        {
                            "gen_loss_3d": gen_loss_3d.item(),
                            "gen_loss_2d": gen_loss_2d.item(),
                            "total_gen_loss": total_gen_loss.item(),
                            "recon_loss_3d": recon_loss_3d.item(),
                            "recon_loss_2d": recon_loss_2d.item(),
                            "psnr": mse2psnr(mse_recon_loss.item()),
                        }
                    )
                    gen_stats_list.append(extra_info)
                gen_stats = self._average_stats(gen_stats_list)

                # -----------------------------------------------------------------------
                # Logging, saving, feedback, and other bells and whistles per iteration:
                # -----------------------------------------------------------------------
                # print(
                #     prof.key_averages(group_by_stack_n=10).table(
                #         sort_by="cpu_time_total", row_limit=20
                #     )
                # )
                # print(
                #     prof.key_averages().table(sort_by="cuda_time_total", row_limit=20)
                # )

                global_step += 1

                all_stats = {}
                all_stats.update(dis_stats)
                all_stats.update(gen_stats)

                # add all the scalar stats to tensorboard:
                for stat_name, stat_value in all_stats.items():
                    tensorboard_writer.add_scalar(
                        stat_name, stat_value, global_step=global_step
                    )

                # log loss values to the console
                if stage_iter % loss_feedback_frequency == 0 or stage_iter == 1:
                    log_info_string = f"Global Iteration: {global_step} Stage: {stage} Iteration: {stage_iter}"
                    for stat_name, stat_value in all_stats.items():
                        log_info_string += f" {stat_name}: {stat_value: .5f}"
                    log.info(log_info_string)

                # step the learning rate schedulers
                # note that we don't apply learning rate decay if lr_decay_steps are None
                if lr_decay_steps is not None and stage_iter % lr_decay_steps == 0:
                    gen_lr_scheduler.step()
                    dis_lr_scheduler.step()
                    new_gen_lrs = [
                        param_group["lr"]
                        for param_group in optimizer_generator.param_groups
                    ]
                    new_dis_lrs = [
                        param_group["lr"]
                        for param_group in optimizer_discriminator.param_groups
                    ]
                    log_string = (
                        f"Adjusted learning rate | "
                        f"Generator learning rates: {new_gen_lrs} | "
                        f"Discriminator learning rates: {new_dis_lrs} | "
                    )
                    log.info(log_string)

                # render feedback output
                if stage_iter % feedback_frequency == 0 or stage_iter == 1:
                    # obtain the reconstruction sample:
                    self._render_reconstruction_feedback(
                        stage=stage,
                        camera_hemisphere_radius=camera_hemisphere_radius,
                        render_feedback_pose=render_feedback_pose,
                        recon_logs_dir=recon_logs_dir,
                        global_step=global_step,
                    )
                    if use_2d_discriminator or use_3d_discriminator:
                        # no need to render the GAN feedback if neither discriminator is in use
                        self._render_gan_feedback(
                            gan_feedback_noise=gan_feedback_noise,
                            stage=stage,
                            camera_hemisphere_radius=camera_hemisphere_radius,
                            render_feedback_pose=render_feedback_pose,
                            gan_logs_dir=gan_logs_dir,
                            global_step=global_step,
                        )

                    if current_stage_2d_discriminator is not None:
                        # log the real and fake-rendered 2D patches if
                        # a 2d discriminator is being used
                        real_patches_grid = make_grid(
                            real_random_patches,
                            nrow=int(np.ceil(np.sqrt(len(real_random_patches)))),
                            padding=0,
                        )
                        fake_patches_grid = make_grid(
                            fake_random_patches,
                            nrow=int(np.ceil(np.sqrt(len(fake_random_patches)))),
                            padding=0,
                        )
                        all_patches = make_grid(
                            torch.stack([real_patches_grid, fake_patches_grid], dim=0),
                            nrow=2,
                            padding=5,
                        )
                        imageio.imwrite(
                            f"{gan_2d_patches_logs_dir}/patches_stage_{stage}_iter_{global_step}.png",
                            to8b(all_patches.permute(1, 2, 0).detach().cpu().numpy()),
                        )

                # save the current model:
                if (
                    stage_iter % save_frequency == 0
                    or stage_iter == num_iterations_per_stage
                ):
                    torch.save(
                        self._get_save_info(
                            discriminator_3d=current_stage_3d_discriminator,
                            discriminator_2d=current_stage_2d_discriminator,
                            extra_info={
                                "camera_hemisphere_radius": camera_hemisphere_radius
                            },
                        ),
                        model_dir / f"model_stage_{stage}_iter_{stage_iter}.pth",
                    )

            # replace the previous stage models with current stage
            prev_stage_3d_discriminator = current_stage_3d_discriminator
            prev_stage_2d_discriminator = current_stage_2d_discriminator
            prev_stage_gen_block = self._generator.get_block_at_stage(stage)

            log.info(
                f"Noise_amp for current stage after training: {self._generator.noise_amps[stage - 1]}"
            )
            log.info("!! Stage complete !!")
        log.info("!! Training complete !!")


# noinspection PyProtectedMember
def create_thre3d_singan_with_direct_supervision_from_saved_model(
    saved_model: Path,
) -> Tuple[Thre3dSinGanWithDirectSupervision, Dict[str, Any]]:
    loaded_model = torch.load(saved_model)
    noise_amps = loaded_model["generator"]["noise_amps"]
    if loaded_model["render_mlp"] is not None:
        if len(loaded_model["render_mlp"]["point_mlp"]["config"]["layer_depths"]) > 5:
            mlp_creator = get_big_render_mlp
        elif (
            len(loaded_model["render_mlp"]["point_mlp"]["config"]["layer_depths"]) <= 3
        ):
            mlp_creator = get_tiny_render_mlp
        else:
            mlp_creator = get_default_render_mlp

        render_mlp = mlp_creator(
            feature_size=loaded_model["generator"]["conf"]["num_features"],
            feature_embeddings_dims=loaded_model["render_mlp"]["point_embedder"][
                "emb_dims"
            ],
            dir_embedding_dims=loaded_model["render_mlp"]["dir_embedder"]["emb_dims"],
            normalize_features=loaded_model["render_mlp"]["normalize_features"],
        )
        render_mlp.load_weights(loaded_model["render_mlp"])
    else:
        render_mlp = None

    if loaded_model["fine_render_mlp"] is not None:
        if (
            len(loaded_model["fine_render_mlp"]["point_mlp"]["config"]["layer_depths"])
            > 5
        ):
            mlp_creator = get_big_render_mlp
        elif (
            len(loaded_model["fine_render_mlp"]["point_mlp"]["config"]["layer_depths"])
            <= 3
        ):
            mlp_creator = get_tiny_render_mlp
        else:
            mlp_creator = get_default_render_mlp

        fine_render_mlp = mlp_creator(
            feature_size=loaded_model["generator"]["conf"]["num_features"],
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

    if "trad_rgba" in loaded_model["conf"]:
        del loaded_model["conf"]["trad_rgba"]

    tsds = Thre3dSinGanWithDirectSupervision(
        render_params=Thre3dSinGanWithDirectSupervisionRenderingParameters(
            **loaded_model["render_params"]
        ),
        render_mlp=render_mlp,
        fine_render_mlp=fine_render_mlp,
        **loaded_model["conf"],
    )

    tsds.load_generator_weights(loaded_model["generator"]["state_dict"])
    tsds.reconstruction_noise = loaded_model["reconstruction_noise"]
    if (
        "trainable_render_mlp" in loaded_model
        and loaded_model["trainable_render_mlp"] is not None
    ):
        tsds._trainable_render_mlp.load_weights(loaded_model["trainable_render_mlp"])

    # noinspection PyProtectedMember
    tsds._generator.noise_amps = noise_amps
    return tsds, loaded_model["extra_info"]
