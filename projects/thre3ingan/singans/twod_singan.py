from pathlib import Path
from typing import Tuple, List, Any, Dict, Optional

import imageio
import numpy as np
import torch.cuda
from PIL import Image
from torch import Tensor
from torch.nn.functional import mse_loss
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from projects.thre3ingan.singans.networks import (
    TwodSinGanGenerator,
    TwodSinGanDiscriminator,
)
from thre3d_atom.training.adversarial.losses import WganGPGanLoss
from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS
from thre3d_atom.utils.imaging_utils import adjust_dynamic_range, mse2psnr, to8b
from thre3d_atom.utils.logging import log
from thre3d_atom.utils.misc import toggle_grad


class TwodSingan:
    def __init__(
        self,
        num_stages: int = 8,
        output_resolution: Tuple[int, int] = (256, 256),
        num_channels: int = NUM_COLOUR_CHANNELS,
        scale_factor: float = (1 / 0.75),
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        # state of the object:
        self._num_stages = num_stages
        self._output_resolution = output_resolution
        self._num_channels = num_channels
        self._scale_factor = scale_factor
        self._device = device

        # create a TwodSingan generator object:
        self._generator = TwodSinGanGenerator(
            output_resolution=self._output_resolution,
            num_channels=self._num_channels,
            num_stages=self._num_stages,
            scale_factor=self._scale_factor,
            device=self._device,
        )

        log.info(
            f"Created a 2D singan generator with stage-wise image_sizes: {self._generator.image_sizes}"
        )
        log.info(f"Generator Architecture: {self._generator}")

        # create a short-hand for the specific noise vector which is used for the reconstruction task:
        self._reconstruction_noise = self._generator.reconstruction_noise

    def generate_samples(
        self,
        num_samples: int = 50,
        stage: Optional[int] = None,
        input_noise_resolution: Optional[Tuple[int, int]] = None,
        batch_size: int = 16,
        reconstruction: bool = False,
    ) -> Tensor:
        input_noise_resolution = (
            self._reconstruction_noise.shape[2:]
            if input_noise_resolution is None
            else input_noise_resolution
        )
        input_noise = (
            self._reconstruction_noise
            if reconstruction
            else torch.randn(
                (num_samples, NUM_COLOUR_CHANNELS, *input_noise_resolution),
                dtype=torch.float32,
                device=self._device,
            )
        )

        with torch.no_grad():
            generated_samples_list = []
            for batch_index in range(
                0, min(num_samples, input_noise.shape[0]), batch_size
            ):
                input_noise_batch = input_noise[batch_index : batch_index + batch_size]
                generated_samples_list.append(
                    adjust_dynamic_range(
                        self._generator(input_noise_batch, stage=stage).cpu(),
                        (-1, 1),
                        (0, 1),
                        slack=True,
                    )
                )
        generated_samples = torch.cat(generated_samples_list, dim=0)

        if reconstruction:
            return generated_samples[0]
        return generated_samples

    def _get_save_info(self, discriminator: TwodSinGanDiscriminator) -> Dict[str, Any]:
        return {
            "conf": {
                "num_stages": self._num_stages,
                "output_resolution": self._output_resolution,
                "num_channels": self._num_channels,
                "scale_factor": self._scale_factor,
            },
            "reconstruction_noise": self._reconstruction_noise,
            "discriminator": discriminator.get_save_info(),
            "generator": self._generator.get_save_info(),
        }

    def _setup_stage_wise_training_images(
        self, training_image_path: Path
    ) -> List[Tensor]:
        # load the training image and create downsampled versions of it:
        image = Image.open(training_image_path)

        # note the swapping of size below to account for the PIL and Pytorch's image representation difference
        stage_wise_real_images = [
            image.resize((size[1], size[0]), Image.ANTIALIAS)
            for size in self._generator.image_sizes
        ]
        stage_wise_real_images = [
            torch.from_numpy(np.array(image).astype(np.float32)).to(self._device)
            for image in stage_wise_real_images
        ]
        stage_wise_real_images = [
            image[
                ..., :NUM_COLOUR_CHANNELS
            ]  # ignore additional channels in case image is RGBA
            for image in stage_wise_real_images
        ]
        stage_wise_real_images = [
            adjust_dynamic_range(image, (0, 255), (-1, 1))
            for image in stage_wise_real_images
        ]
        return stage_wise_real_images

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

    def train(
        self,
        training_image_path: Path,
        num_iterations_per_stage: int = 2000,
        num_dis_steps: int = 3,
        num_gen_steps: int = 3,
        g_lrate: float = 0.003,
        d_lrate: float = 0.003,
        lr_decay_steps: int = 1600,
        recon_loss_alpha: float = 10.0,
        num_feedback_samples: int = 6,
        feedback_frequency: int = 500,
        loss_feedback_frequency: int = 100,
        save_frequency: int = 500,
        output_dir: Path = Path(__file__).parent.absolute() / "logs",
    ) -> None:

        # setup the stage-wise training real images
        stage_wise_real_images = self._setup_stage_wise_training_images(
            training_image_path
        )

        # setup output directories:
        model_dir, logs_dir = output_dir / "saved_models", output_dir / "training_logs"
        tensorboard_dir = logs_dir / "tensorboard"
        render_dir = logs_dir / "rendered_output"
        for directory in (model_dir, logs_dir, tensorboard_dir, render_dir):
            directory.mkdir(exist_ok=True, parents=True)

        # tensorboard writer:
        tensorboard_writer = SummaryWriter(tensorboard_dir)

        # setup optimizers:
        optimizer_generator = torch.optim.Adam(
            self._generator.parameters(), lr=g_lrate, betas=(0, 0.99)
        )
        gen_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer_generator, gamma=0.1
        )

        # Define a gan_loss:
        gan_loss = WganGPGanLoss()

        log.info("!! Beginning Training !!")
        global_step = 0
        prev_stage_discriminator, prev_stage_gen_block = None, None
        dis_lr_scheduler = None
        for stage in range(1, self._num_stages + 1):
            log.info(f"Starting new stage: {stage}")

            # setup directories for the current stage:
            recon_logs_dir = render_dir / "recon" / str(stage)
            gan_logs_dir = render_dir / "gan" / str(stage)
            for directory in (recon_logs_dir, gan_logs_dir):
                directory.mkdir(exist_ok=True, parents=True)

            current_stage_real_image = (
                stage_wise_real_images[stage - 1].permute(2, 0, 1).unsqueeze(0)
            )

            # create a new discriminator (and it's optimizer) for the current stage
            current_stage_discriminator = TwodSinGanDiscriminator(
                in_channels=self._num_channels
            ).to(self._device)
            if prev_stage_discriminator is not None:
                # load the previous stage discriminator's weights into this stage's discriminator:
                current_stage_discriminator.load_state_dict(
                    prev_stage_discriminator.state_dict()
                )
            if prev_stage_gen_block is not None:
                self._generator.load_block_at_stage(prev_stage_gen_block, stage)

            optimizer_discriminator = torch.optim.Adam(
                current_stage_discriminator.parameters(),
                lr=d_lrate
                if dis_lr_scheduler is None
                else dis_lr_scheduler.get_last_lr()[0],
                betas=(0, 0.99),
            )
            dis_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer_discriminator, gamma=0.1
            )

            log.info(f"New Discriminator Architecture: {current_stage_discriminator}")
            for stage_iter in range(1, num_iterations_per_stage + 1):
                # -----------------------------------------------------------------------
                # Discriminator steps:
                # -----------------------------------------------------------------------
                # This is done to speed up training
                toggle_grad(self._generator, False)
                toggle_grad(current_stage_discriminator, True)

                dis_stats_list = []
                for _ in range(num_dis_steps):
                    fake_image = self._generator(stage=stage)
                    dis_loss, extra_info = gan_loss.dis_loss(
                        current_stage_discriminator,
                        current_stage_real_image,
                        fake_image,
                    )

                    optimizer_discriminator.zero_grad()
                    dis_loss.backward()
                    optimizer_discriminator.step()

                    extra_info.update({"dis_loss": dis_loss.item()})
                    dis_stats_list.append(extra_info)
                dis_stats = self._average_stats(dis_stats_list)

                # -----------------------------------------------------------------------
                # Generator steps:
                # -----------------------------------------------------------------------
                # This is done to speed up training
                toggle_grad(self._generator, True)
                toggle_grad(current_stage_discriminator, False)

                gen_stats_list = []
                for _ in range(num_gen_steps):
                    # -----------------------------------------------------------------------
                    # Reconstruction step:
                    # -----------------------------------------------------------------------

                    reconstructed_image = self._generator(
                        self._reconstruction_noise, stage
                    )
                    recon_loss = mse_loss(reconstructed_image, current_stage_real_image)

                    # -----------------------------------------------------------------------
                    # Adversarial step:
                    # -----------------------------------------------------------------------

                    fake_image = self._generator(stage=stage)
                    gen_loss, extra_info = gan_loss.gen_loss(
                        current_stage_discriminator,
                        _,
                        fake_image,
                    )

                    total_gen_loss = (recon_loss_alpha * recon_loss) + gen_loss

                    optimizer_generator.zero_grad()
                    total_gen_loss.backward()
                    optimizer_generator.step()

                    extra_info.update(
                        {
                            "gen_loss": gen_loss.item(),
                            "recon_loss": recon_loss.item(),
                            "psnr": mse2psnr(recon_loss.item()),
                        }
                    )
                    gen_stats_list.append(extra_info)
                gen_stats = self._average_stats(gen_stats_list)

                # -----------------------------------------------------------------------
                # Logging, saving, feedback, and other bells and whistles per iteration:
                # -----------------------------------------------------------------------
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
                    log_info_string = f"Stage: {stage} Iteration: {stage_iter}"
                    for stat_name, stat_value in all_stats.items():
                        log_info_string += f" {stat_name}: {stat_value}"
                    log.info(log_info_string)

                # step the learning rate schedulers
                if global_step % lr_decay_steps == 0:
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
                        f"Discriminator learning rates: {new_dis_lrs}"
                    )
                    log.info(log_string)

                # render feedback output
                if stage_iter % feedback_frequency == 0 or stage_iter == 1:
                    reconstructed_image = self.generate_samples(
                        stage=stage, reconstruction=True
                    )

                    generated_samples = self.generate_samples(
                        num_feedback_samples, stage=stage
                    )
                    generated_samples_sheet = make_grid(
                        generated_samples,
                        nrow=int(np.ceil(np.sqrt(num_feedback_samples))),
                        padding=0,
                    )

                    imageio.imwrite(
                        recon_logs_dir / f"iter_{stage_iter}.png",
                        to8b(reconstructed_image.permute(1, 2, 0).numpy()),
                    )

                    imageio.imwrite(
                        gan_logs_dir / f"iter_{stage_iter}.png",
                        to8b(generated_samples_sheet.permute(1, 2, 0).numpy()),
                    )

                # save the current model:
                if (
                    stage_iter % save_frequency == 0
                    or stage_iter == num_iterations_per_stage
                ):
                    torch.save(
                        self._get_save_info(current_stage_discriminator),
                        model_dir / f"model_stage_{stage}_iter_{stage_iter}.pth",
                    )
            # copy the current stage discriminator into prev_stage and the current stage generator_block
            prev_stage_discriminator = current_stage_discriminator
            prev_stage_gen_block = self._generator.get_block_at_stage(stage)
            log.info("!! Stage complete !!")
        log.info("!! Training complete !!")
