import argparse
import sys
from pathlib import Path
from typing import List

import torch.cuda
from torch.backends import cudnn

from projects.thre3ingan.singans.thre3d_singan_with_direct_supervision import (
    Thre3dSinGanWithDirectSupervisionRenderingParameters,
    Thre3dSinGanWithDirectSupervision,
)
from thre3d_atom.modules.volumetric_model.volumetric_model import (
    create_vol_mod_from_saved_model,
    process_rgba_model,
    process_hybrid_rgba_volumetric_model,
)
from thre3d_atom.utils.config_utils import (
    str2bool,
    log_args_config_to_disk,
    int_or_none,
    float_or_none,
)
from thre3d_atom.utils.imaging_utils import scale_camera_intrinsics
from thre3d_atom.utils.logging import log

# Turn on benchmarking for fast training
cudnn.benchmark = False


# noinspection DuplicatedCode
def parse_arguments(args: List[str]) -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("Thre3d SinGAN model trainer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument("-i", "-m", "--model_path",
                        action="store", type=Path, required=True, help="path to the trained volumetric_model")
    parser.add_argument("-o", "--output_dir",
                        action="store", type=Path, required=True, help="path to the output directory")

    # Optional arguments
    # Rendering related arguments
    parser.add_argument("--num_rays_chunk", action="store", type=int, required=False, default=65536,
                        help="number of rays in a single chunk (similar to batch size for rendering)")
    parser.add_argument("--density_noise_std", action="store", type=float, required=False, default=0.0,
                        help="std of the noise added to the network predicted density")
    parser.add_argument("--scale_camera_intrinsics", action="store", type=float_or_none, required=False, default=None,
                        help="scale the camera intrinsics for faster/slower rendered feedback during training")

    # SinGAN preprocessing / architecture related arguments:
    parser.add_argument("--use_eql", action="store", type=str2bool, required=False, default=False,
                        help="whether to use equalized learning rate based Modules (Convs and Linear).")
    parser.add_argument("--num_stages", action="store", type=int, required=False, default=7,
                        help="number of stages in the generator architecture.")
    parser.add_argument("--intermediate_channels", action="store", type=int, required=False,
                        default=[32, 32, 32, 64, 64, 64, 92], nargs="+",
                        help="number of intermediate channels in the blocks of the generator architecture.")
    parser.add_argument("--output_resolution", action="store", type=int, required=False, nargs=3,
                        default=None,
                        help="Resolution of the final stage generated image")
    parser.add_argument("--scale_factor", action="store", type=float, required=False, default=1 / 0.75,
                        help="scale by which generation is upsampled per stage")

    # Adversarial training-schedule based parameters:
    parser.add_argument("--use_3d_disc", action="store", type=str2bool, required=False, default=True,
                        help="whether to use a 3d discriminator in the singan training")
    parser.add_argument("--use_2d_disc", action="store", type=str2bool, required=False, default=True,
                        help="whether to use a 2d discriminator in the singan training")
    parser.add_argument("--apply_3d_reconstruction_loss", action="store", type=str2bool, required=False, default=True,
                        help="whether to apply 3D reconstruction loss for a single fixed seed.")
    parser.add_argument("--apply_2d_reconstruction_loss", action="store", type=str2bool, required=False, default=True,
                        help="whether to apply 2D reconstruction loss for a single fixed seed.")
    parser.add_argument("--virtual_camera_size", action="store", type=int_or_none, required=False, default=None,
                        help="size for the virtual camera used for the 2d discriminator. "
                             "Defaults to original size of the volmod")
    parser.add_argument("--patch_size", action="store", type=int, required=False, default=11,
                        help="patch size for 2d discriminator")
    parser.add_argument("--patch_batch_size", action="store", type=int, required=False, default=64,
                        help="patch batch size for 2d discriminator")
    parser.add_argument("--adv_2d_loss_lambda", action="store", type=float, required=False, default=0.3,
                        help="weight of 2D adversarial loss against the 3D adversarial loss")
    # training arguments
    parser.add_argument("--num_iterations_per_stage", action="store", type=int, required=False, default=2000,
                        help="number of iterations to train the model for per stage")
    parser.add_argument("--num_dis_steps", action="store", type=int, required=False, default=3,
                        help="number of discriminator steps performed per iteration")
    parser.add_argument("--num_gen_steps", action="store", type=int, required=False, default=3,
                        help="number of generator steps performed per iteration")
    parser.add_argument("--g_lrate", action="store", type=float, required=False, default=0.0005,
                        help="generator learning rate")
    parser.add_argument("--d_lrate", action="store", type=float, required=False, default=0.0005,
                        help="discriminator learning rate")
    parser.add_argument("--apply_learning_rate_decay", action="store", type=str2bool, required=False, default=True,
                        help="whether to apply learning rate decay while training")
    parser.add_argument("--lr_decay_steps", action="store", type=int, required=False, default=1600,
                        help="number of iterations after which learning rate is decayed")
    parser.add_argument("--lr_decay_gamma", action="store", type=float, required=False, default=0.1,
                        help="value of gamma for exponential learning rate decay")
    parser.add_argument("--threed_wgan_gp_lambda", action="store", type=float, required=False, default=0.1,
                        help="value of lambda used by the WGAN-GP gradient penalty for 3d discriminator")
    parser.add_argument("--threed_wgan_gp_drift_penalty", action="store", type=float, required=False, default=0.0,
                        help="value of lambda used by the WGAN-GP drift penalty for the 3d discriminator")
    parser.add_argument("--twod_wgan_gp_lambda", action="store", type=float, required=False, default=1.0,
                        help="value of lambda used by the WGAN-GP gradient penalty for 2d discriminator")
    parser.add_argument("--twod_wgan_gp_drift_penalty", action="store", type=float, required=False, default=0.0,
                        help="value of lambda used by the WGAN-GP drift penalty for the 2d discriminator")
    parser.add_argument("--noise_scale", action="store", type=float, required=False, default=0.001,
                        help="scalar multiplier value for adding intermediate noise to the generator")
    parser.add_argument("--threed_recon_loss_alpha", action="store", type=float, required=False, default=10.0,
                        help="reconstruction loss scale-factor compared to gan loss for the generator for 3D")
    parser.add_argument("--twod_recon_loss_alpha", action="store", type=float, required=False, default=10.0,
                        help="reconstruction loss scale-factor compared to gan loss for the generator for 2D."
                             "Please note the default value is 10.0 when both 2D and 3D discriminators are used, "
                             "but remember to change it to 100.0 when only 2D discriminator is used")
    parser.add_argument("--num_feedback_samples", action="store", type=int, required=False, default=6,
                        help="number of random gan samples generated for feedback")
    parser.add_argument("--feedback_frequency", action="store", type=int, required=False, default=500,
                        help="frequency of rendering feedback pose")
    parser.add_argument("--loss_feedback_frequency", action="store", type=int, required=False, default=100,
                        help="frequency of logging loss values to console")
    parser.add_argument("--save_frequency", action="store", type=int, required=False, default=500,
                        help="frequency of taking a snapshot")
    parser.add_argument("--fast_debug_mode", action="store", type=str2bool, required=False, default=False,
                        help="whether to run the training in fast_debug_mode."
                             "skips lengthy visualizations")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def train_thre3d_singan(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Welcome to Thre3d SinGan Training")

    vol_mod, extra_info = create_vol_mod_from_saved_model(args.model_path, device)

    # noinspection PyProtectedMember
    if vol_mod.render_mlp is None:
        vol_mod = process_rgba_model(vol_mod)
    elif vol_mod._hybrid_rgba_mode:
        vol_mod = process_hybrid_rgba_volumetric_model(vol_mod)

    # Create a Thre3dSinGanWithDirectSupervision object:
    # noinspection PyProtectedMember
    render_params = Thre3dSinGanWithDirectSupervisionRenderingParameters(
        num_rays_chunk=args.num_rays_chunk,
        num_samples_per_ray=vol_mod._render_params.num_samples_per_ray,
        num_fine_samples_per_ray=vol_mod._render_params.num_fine_samples_per_ray,
        perturb_sampled_points=vol_mod._render_params.perturb_sampled_points,
        density_noise_std=args.density_noise_std,
        camera_intrinsics=scale_camera_intrinsics(
            extra_info["camera_intrinsics"], args.scale_camera_intrinsics
        )
        if args.scale_camera_intrinsics is not None
        else extra_info["camera_intrinsics"],
        scene_bounds=extra_info["scene_bounds"],
    )
    thre3d_singan_with_ds = Thre3dSinGanWithDirectSupervision(
        render_params=render_params,
        render_mlp=vol_mod.render_mlp,
        fine_render_mlp=vol_mod.fine_render_mlp,
        num_stages=args.num_stages,
        output_grid_resolution=args.output_resolution
        if args.output_resolution is not None
        else vol_mod.feature_grid.features.shape[:-1],
        intermediate_features=args.intermediate_channels
        if len(args.intermediate_channels) > 1
        else args.intermediate_channels[0],
        scale_factor=args.scale_factor,
        use_eql=args.use_eql,
        noise_scale=args.noise_scale,
        device=device,
    )

    # log the configuration as a yaml:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Logging configuration file ...")
    log_args_config_to_disk(args, args.output_dir)

    # train the model
    thre3d_singan_with_ds.train(
        training_feature_grid=vol_mod.feature_grid,
        camera_hemisphere_radius=extra_info["hemispherical_radius"],
        discriminator_receptive_field=(11, 11, 11),
        discriminator_num_layers=5,
        num_iterations_per_stage=args.num_iterations_per_stage,
        num_dis_steps=args.num_dis_steps,
        num_gen_steps=args.num_gen_steps,
        g_lrate=args.g_lrate,
        d_lrate=args.d_lrate,
        use_3d_discriminator=args.use_3d_disc,
        use_2d_discriminator=args.use_2d_disc,
        apply_3d_reconstruction_loss=args.apply_3d_reconstruction_loss,
        apply_2d_reconstruction_loss=args.apply_2d_reconstruction_loss,
        virtual_camera_size=args.virtual_camera_size,
        patch_size=args.patch_size,
        patch_batch_size=args.patch_batch_size,
        adv_2d_loss_lambda=args.adv_2d_loss_lambda,
        threed_wgan_gp_lambda=args.threed_wgan_gp_lambda,
        threed_wgan_gp_drift_penalty=args.threed_wgan_gp_drift_penalty,
        twod_wgan_gp_lambda=args.twod_wgan_gp_lambda,
        twod_wgan_gp_drift_penalty=args.twod_wgan_gp_drift_penalty,
        lr_decay_steps=args.lr_decay_steps if args.apply_learning_rate_decay else None,
        lr_decay_gamma=args.lr_decay_gamma,
        threed_recon_loss_alpha=args.threed_recon_loss_alpha,
        twod_recon_loss_alpha=args.twod_recon_loss_alpha,
        num_feedback_samples=args.num_feedback_samples,
        feedback_frequency=args.feedback_frequency,
        loss_feedback_frequency=args.loss_feedback_frequency,
        save_frequency=args.save_frequency,
        output_dir=args.output_dir,
        fast_debug_mode=args.fast_debug_mode,
    )

    log.info("!See you next time!")


def main() -> None:
    train_thre3d_singan(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
