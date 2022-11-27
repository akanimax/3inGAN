import copy
import dataclasses
import time
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Iterator, Sequence, Tuple

import imageio
import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import mse_loss, binary_cross_entropy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from thre3d_atom.data.micro_loaders import PointOccupancyMeshDataset
from thre3d_atom.modules.volumetric_model.utils import render_image_in_chunks
from thre3d_atom.networks.dense_nets import SkipMLP, SkipMLPConfig
from thre3d_atom.networks.network_interface import Network
from thre3d_atom.networks.shared.layers import PositionalEncodingsEmbedder
from thre3d_atom.rendering.volumetric.implicit import (
    cast_rays,
    accumulate_processed_points_on_rays,
    simple_process_points_on_rays_with_network,
)
from thre3d_atom.rendering.volumetric.render_interface import Rays, RenderOut, render
from thre3d_atom.rendering.volumetric.sample import sample_uniform_points_on_rays
from thre3d_atom.rendering.volumetric.utils import (
    compute_grid_sizes,
    reshape_rendered_output,
    collate_rendered_output,
)
from thre3d_atom.rendering.volumetric.voxels import (
    FeatureGrid,
    scale_feature_grid_with_output_size,
)
from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS
from thre3d_atom.utils.imaging_utils import (
    CameraPose,
    SceneBounds,
    CameraIntrinsics,
    pose_spherical,
    mse2psnr,
    postprocess_depth_map,
)
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import batchify


def relu_field(x: Tensor) -> Tensor:
    return torch.relu(x)


@dataclasses.dataclass
class VolumetricOccupancyRenderParams:
    num_rays_chunk: int = 32768
    num_points_chunk: int = 512  # needed for MLP mode
    num_samples_per_ray: int = 256
    scene_bounds: SceneBounds = SceneBounds(1.35, 1.65)
    hemispherical_radius: float = 1.5
    camera_pitch: float = 0.0
    camera_intrinsics: CameraIntrinsics = CameraIntrinsics(
        height=1024, width=1024, focal=1024
    )
    num_poses: int = 42

    @property
    def camera_pose(self) -> CameraPose:
        return pose_spherical(
            yaw=0.0, pitch=self.camera_pitch, radius=self.hemispherical_radius
        )

    @property
    def render_poses(self) -> List[CameraPose]:
        def translate_z(z: float, device=torch.device("cpu")) -> Tensor:
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

        def rotate_y(angle: float, device=torch.device("cpu")) -> Tensor:
            return torch.tensor(
                [
                    [np.cos(angle), 0, -np.sin(angle), 0],
                    [0, 1, 0, 0],
                    [np.sin(angle), 0, np.cos(angle), 0],
                    [0, 0, 0, 1],
                ],
                dtype=torch.float32,
                device=device,
            )

        # noinspection DuplicatedCode
        def pose_spherical_specific(
            angle: float, radius: float, device=torch.device("cpu")
        ) -> CameraPose:
            c2w = translate_z(radius, device)
            c2w = rotate_y(angle / 180.0 * np.pi, device) @ c2w
            c2w = (
                torch.tensor(
                    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    dtype=torch.float32,
                    device=device,
                )
                @ c2w
            )
            return CameraPose(rotation=c2w[:3, :3], translation=c2w[:3, 3:])

        return [
            pose_spherical_specific(angle, self.hemispherical_radius)
            for (angle) in np.linspace(0, 360, self.num_poses)[:-1]
        ]


class OccupancyMLP(Network):
    def __init__(self, config: SkipMLPConfig, embedding_dims: int = 10):
        super().__init__()

        # configuration of the model:
        self._config = config
        self._embedding_dims = embedding_dims

        # Modules used by this MLP
        self._pe_mapper = PositionalEncodingsEmbedder(
            self._config.input_dims, self._embedding_dims
        )
        new_config = copy.deepcopy(config)
        new_config.input_dims = self._pe_mapper.output_shape[-1]
        self._mlp = SkipMLP(new_config)

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return self._pe_mapper.input_shape

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        return self._mlp.output_shape

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "config": dataclasses.asdict(self._config),
                "embedding_dims": self._embedding_dims,
            },
            "state_dict": self.state_dict(),
        }

    def forward(self, x: Tensor) -> Tensor:
        return self._mlp(self._pe_mapper(x))


class VolumetricOccupancyModel:
    def __init__(
        self,
        representation: Union[FeatureGrid, OccupancyMLP],
        render_params: Optional[
            VolumetricOccupancyRenderParams
        ] = VolumetricOccupancyRenderParams(),
        relu_field_mode: bool = True,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        # object state:
        self._repr = representation
        self._device = device
        self._render_params = render_params
        self._relu_field_mode = relu_field_mode

        log.info(f"Using underlying representation as: {self._repr}")

        # short-hands
        self._grid_mode = isinstance(self._repr, FeatureGrid)

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "grid_mode": self._grid_mode,
            "config": {"relu_field_mode": self._relu_field_mode},
            "render_params": dataclasses.asdict(self._render_params),
            "state_dict": self._repr.state_dict(),
        }

    def render(
        self,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        camera_pose: Optional[CameraPose] = None,
        scene_bounds: Optional[SceneBounds] = None,
        verbose: bool = False,
    ) -> np.array:
        camera_intrinsics = (
            camera_intrinsics
            if camera_intrinsics is not None
            else self._render_params.camera_intrinsics
        )
        camera_pose = (
            camera_pose if camera_pose is not None else self._render_params.camera_pose
        )
        scene_bounds = (
            scene_bounds
            if scene_bounds is not None
            else self._render_params.scene_bounds
        )

        raw2alpha = (
            (lambda x, y: relu_field(x)) if self._relu_field_mode else (lambda x, y: x)
        )

        if self._grid_mode:
            # We are in grid mode
            # noinspection PyTypeChecker
            rendered_output = render_image_in_chunks(
                cam_intrinsics=camera_intrinsics,
                camera_pose=camera_pose,
                num_rays_chunk=self._render_params.num_rays_chunk,
                num_samples_per_ray=self._render_params.num_samples_per_ray,
                feature_grid=self._repr,
                scene_bounds=scene_bounds,
                density_noise_std=0.0,
                perturb_sampled_points=True,
                use_dists_in_rendering=False,
                raw2alpha=raw2alpha,
                optimized_sampling_mode=True,
                gpu_render=True,
                device=self._device,
                verbose=verbose,
            )
        else:
            # We are in MLP mode:
            def _render_rays(rays_chunk: Rays) -> RenderOut:
                sampler_fn = partial(
                    sample_uniform_points_on_rays,
                    perturb=True,
                    linear_disparity_sampling=False,
                )
                point_processor_fn = partial(
                    simple_process_points_on_rays_with_network,
                    network=self._repr,
                    chunk_size=self._render_params.num_points_chunk,
                )
                accumulator_fn = partial(
                    accumulate_processed_points_on_rays,
                    density_noise_std=0.0,
                    raw2_alpha=raw2alpha,
                )

                # Render the rays with a batchified point processor for better efficiency
                # noinspection PyTypeChecker
                rendered_chunk, _ = render(
                    rays=rays_chunk,
                    scene_bounds=scene_bounds,
                    num_samples=self._render_params.num_samples_per_ray,
                    sampler_fn=sampler_fn,
                    point_processor_fn=point_processor_fn,
                    accumulator_fn=accumulator_fn,
                )

                return rendered_chunk

            rays = cast_rays(camera_intrinsics, camera_pose, device=self._device)

            # batchified rendering (aka. ray_processing):
            # noinspection PyTypeChecker
            batchified_render = batchify(
                _render_rays,
                collate_fn=collate_rendered_output,
                chunk_size=self._render_params.num_rays_chunk,
                verbose=verbose,
            )

            if verbose:
                log.info("Rendering the MLP output chunk-by-chunk")

            with torch.no_grad():
                rendered_output = reshape_rendered_output(
                    batchified_render(rays), camera_intrinsics
                )

        rendered_image = postprocess_depth_map(
            rendered_output.disparity.squeeze().cpu().numpy(),
            scene_bounds=scene_bounds,
        )
        return rendered_image

    def render_animation(
        self,
        output_path: Path,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        scene_bounds: Optional[SceneBounds] = None,
        render_poses: Optional[List[CameraPose]] = None,
    ) -> None:
        # make sure output path exists:
        if output_path.is_dir() and output_path.exists():
            output_path = output_path / "360_spinning_anim.mp4"
        elif output_path.is_dir():
            output_path.mkdir(parents=True, exist_ok=True)
            output_path = output_path / "360_spinning_anim.mp4"
        elif not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)

        render_poses = (
            render_poses
            if render_poses is not None
            else self._render_params.render_poses
        )
        rendered_buffer = []
        total_frames = len(render_poses)
        for frame_num, render_pose in enumerate(render_poses, start=1):
            log.info(f"rendering frame: {frame_num} / {total_frames}")
            rendered_buffer.append(
                self.render(camera_intrinsics, render_pose, scene_bounds, verbose=True)
            )
        rendered_animation = np.stack(rendered_buffer, axis=0)
        imageio.mimwrite(output_path, rendered_animation)
        log.info(f"Rendering animation complete! Please check: {output_path}")

    def test(
        self,
        data_loader: Iterator[Tensor],
        batch_size: int,
        num_points_estimate: int = 100_000,
        global_step: Optional[int] = None,
        tensorboard_writer: Optional[SummaryWriter] = None,
    ) -> float:
        # computes the Volumetric IoU between the original GT mesh and the learned
        # underlying representation.
        # Refer to the https://arxiv.org/pdf/1812.03828.pdf paper for more info
        num_points_in_intersection, num_points_in_union = (
            torch.zeros((1,), device=self._device),
            torch.zeros((1,), device=self._device),
        )
        log.info(f"Computing the Volumetric IoU between the gt and the _repr ...")
        with torch.no_grad():
            for _ in tqdm(range(0, num_points_estimate, batch_size)):
                datum = next(data_loader)
                points, occupancies = datum[:, :-1], datum[:, -1:]
                points, occupancies = points.to(self._device), occupancies.to(
                    self._device
                )
                preds = self._repr(points)
                if preds.shape[-1] != 1:
                    preds = preds[:, -1:]
                pred_occupancies = (
                    relu_field(preds) if self._relu_field_mode else preds
                ) > 0.0

                union = torch.clip(occupancies + pred_occupancies, 0.0, 1.0)
                intersection = occupancies * pred_occupancies
                num_points_in_union += union.sum()
                num_points_in_intersection += intersection.sum()
            vol_iou = num_points_in_intersection / num_points_in_union

        # Write tensorboard summaries if requested:
        if tensorboard_writer is not None and global_step is not None:
            tensorboard_writer.add_scalar(
                "VOLUMETRIC_IOU", vol_iou, global_step=global_step
            )

        return vol_iou.item()

    def train(
        self,
        dataset: PointOccupancyMeshDataset,
        batch_size: int,
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
        output_dir: Path = Path(__file__).parent.absolute() / "logs",
        render_feedback_pose: Optional[CameraPose] = None,
        loss_feedback_freq: Optional[int] = 100,
        num_workers: int = 4,
    ) -> None:
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
        stagewise_grid_sizes = None
        if self._grid_mode:
            stagewise_grid_sizes = compute_grid_sizes(
                self._repr.grid_dims, num_stages=num_stages, scale_factor=scale_factor
            )
            with torch.no_grad():
                # ---------------------------------------------------------------------
                # Note that this loop is unnecessarily required because of the buggy
                # interpolation implementation in PyTorch. Reference:
                # https://www.cs.cmu.edu/~clean-fid/
                # ---------------------------------------------------------------------
                for grid_size in reversed(stagewise_grid_sizes[:-1]):
                    self._repr = scale_feature_grid_with_output_size(
                        self._repr,
                        output_size=grid_size,
                    )

        # setup render_feedback_pose
        if render_feedback_pose is None:
            render_feedback_pose = self._render_params.camera_pose

        # setup certain frequencies
        loss_feedback_freq = (
            feedback_freq // 10 if loss_feedback_freq is None else loss_feedback_freq
        )

        # short-hands for better code readability
        scene_bounds = self._render_params.scene_bounds
        camera_intrinsics = self._render_params.camera_intrinsics

        # setup data loader:
        data_loader = iter(
            DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=num_workers > 0,
                persistent_workers=num_workers > 0,
            )
        )

        # setup output directories
        model_dir = output_dir / "saved_models"
        logs_dir = output_dir / "training_logs"
        tensorboard_dir = logs_dir / "tensorboard"
        render_dir = logs_dir / "rendered_output"
        for directory in (model_dir, logs_dir, tensorboard_dir, render_dir):
            directory.mkdir(exist_ok=True, parents=True)

        # setup tensorboard writer
        tensorboard_writer = SummaryWriter(str(tensorboard_dir))

        # start the training loop
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

            # setup optimizer:
            optimizer = torch.optim.Adam(
                params=[{"params": self._repr.parameters(), "lr": current_stage_lr}],
                betas=(0.9, 0.999),  # written here only for showing explicitly
            )

            # setup learning rate schedulers for the optimizer
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=current_stage_lr_decay_gamma
            )

            log.info(f"Training stage: {stage} ")
            if self._grid_mode:
                log.info(f"Feature grid resolution: {self._repr.grid_dims}")

            # log the current learning rates being used:
            current_stage_lrs = [
                param_group["lr"] for param_group in optimizer.param_groups
            ]
            log_string = f"Current stage learning rates: {current_stage_lrs} "
            log.info(log_string)

            num_iterations_in_current_stage = num_iterations_per_stage[stage - 1]
            stage_iteration = 0
            global_last_time = time.time()

            while stage_iteration < num_iterations_in_current_stage:
                # ----------------------------------------------------------------
                # Core training computation:
                # ----------------------------------------------------------------
                # get a batch of points and occupancies:
                datum = next(data_loader)
                points, occupancies = (
                    datum[:, :NUM_COORD_DIMENSIONS],
                    datum[:, NUM_COORD_DIMENSIONS:],
                )
                points, occupancies = points.to(self._device), occupancies.to(
                    self._device
                )

                # perform the forward pass:
                preds = self._repr(points)
                if preds.shape[-1] > 1:
                    preds = preds[:, -1:]

                # ReLUField:
                predicted_occupancies = (
                    relu_field(preds) if self._relu_field_mode else preds
                )

                # compute all the different losses:
                mse_score = mse_loss(predicted_occupancies, occupancies)
                bce_loss = binary_cross_entropy(predicted_occupancies, occupancies)

                # the expression for total loss is computed as:
                total_loss = bce_loss
                # ----------------------------------------------------------------

                # optimization steps:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # increment counters
                global_step += 1
                stage_iteration += 1

                time_spent_actually_training += time.time() - global_last_time

                # --------------------------------------------------------------------
                # Non-training block
                # --------------------------------------------------------------------
                # Compute training batch-wise psnrs:
                psnr_score = None
                if (
                    global_step % loss_feedback_freq == 0
                    or stage_iteration == 0
                    or stage_iteration == num_iterations_in_current_stage - 1
                ):
                    psnr_score = mse2psnr(mse_score)

                # tensorboard summaries feedback
                if (
                    global_step % loss_feedback_freq == 0
                    or stage_iteration == 0
                    or stage_iteration == num_iterations_in_current_stage - 1
                ):
                    for summary_name, summary_value in (("total_loss", total_loss),):
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
                    # loss and psnr values:
                    loss_info_string += (
                        f"Loss: {mse_score: .3f} " f"PSNR: {psnr_score: .3f} "
                    )
                    # Add total loss to the log by default:
                    loss_info_string += f"Total_loss: {total_loss: .5f} "

                    # log the loss to console
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
                    rendered_image = self.render(
                        camera_intrinsics,
                        render_feedback_pose,
                        scene_bounds,
                        verbose=True,
                    )
                    imageio.imwrite(
                        render_dir / f"render_{stage}_iter_{stage_iteration}.png",
                        rendered_image,
                    )

                # obtain and log the test metrics
                if (
                    global_step % testing_freq == 0
                    or stage_iteration == num_iterations_in_current_stage - 1
                ):
                    score = self.test(
                        data_loader,
                        batch_size,
                        global_step=global_step,
                        tensorboard_writer=tensorboard_writer,
                    )
                    log.info(f"Current Volumetric IoU score: {score: .3f}")

                # save the model
                if (
                    global_step % save_freq == 0
                    or stage_iteration == num_iterations_in_current_stage - 1
                ):
                    torch.save(
                        self.get_save_info(),
                        model_dir / f"model_stage_{stage}_iter_{global_step}.pth",
                    )
                # ====================================================================

                global_last_time = time.time()

            # upsample the feature-grid after the completion of the stage:
            # don't upsample the feature grid if the last stage is complete
            if self._grid_mode and stage != num_stages:
                with torch.no_grad():
                    self._repr = scale_feature_grid_with_output_size(
                        self._repr, output_size=stagewise_grid_sizes[stage]
                    )

        # save the final trained model
        torch.save(
            self.get_save_info(),
            model_dir / f"model_final.pth",
        )

        # generate the final animation:
        self.render_animation(output_path=render_dir / f"final_spin_anim.mp4")

        # training complete yay! :)
        log.info("Training complete")
        log.info(
            f"Total actual training time: {timedelta(seconds=time_spent_actually_training)}"
        )
