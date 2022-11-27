import argparse
from pathlib import Path

import torch
from torch.backends import cudnn

from projects.thre3ingan.singans.image_model import load_trained_image_model
from projects.thre3ingan.singans.twod_singan import TwodSingan
from projects.thre3ingan.singans.twod_singan_on_image_model import (
    TwodSinganWithImageModel,
)
from thre3d_atom.utils.config_utils import str2bool, log_args_config_to_disk
from thre3d_atom.utils.logging import log

cudnn.benchmark = True


def parse_arguments() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("2D singan training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument("-i", "--image_path", action="store", type=Path, required=True,
                        help="path to the training image")
    parser.add_argument("-o", "--output_dir", action="store", type=Path, required=True,
                        help="path to the output asset directory")

    # 2D Singan related arguments:
    parser.add_argument("--train_in_feature_grid_mode", action="store", type=str2bool, required=False, default=False,
                        help="Whether to train the 2D-Singan with learned image feature-grids")
    parser.add_argument("--use_decoder_mlp_in_training", action="store", type=str2bool, required=False, default=False,
                        help="Whether to use the decoder_mlp while training the "
                             "2D-Singan with learned image feature-grids")
    parser.add_argument("--use_eql", action="store", type=str2bool, required=False, default=True,
                        help="whether to use equalized learning rate based Modules (Convs and Linear)")
    parser.add_argument("--num_stages", action="store", type=int, required=False, default=8,
                        help="number of stages in the generator architecture.")
    parser.add_argument("--intermediate_channels", action="store", type=int, required=False, default=32,
                        help="number of intermediate channels in the blocks of the generator architecture.")
    parser.add_argument("--output_resolution", action="store", type=int, required=False, nargs=2,
                        default=(512, 512),
                        help="Resolution of the final stage generated image")
    parser.add_argument("--scale_factor", action="store", type=float, required=False, default=1 / 0.75,
                        help="scale by which generation is upsampled per stage")

    # training arguments
    parser.add_argument("--num_iterations_per_stage", action="store", type=int, required=False, default=2000,
                        help="number of iterations to train the model for per stage")
    parser.add_argument("--num_dis_steps", action="store", type=int, required=False, default=3,
                        help="number of discriminator steps performed per iteration")
    parser.add_argument("--num_gen_steps", action="store", type=int, required=False, default=3,
                        help="number of generator steps performed per iteration")
    parser.add_argument("--g_lrate", action="store", type=float, required=False, default=0.003,
                        help="generator learning rate")
    parser.add_argument("--d_lrate", action="store", type=float, required=False, default=0.003,
                        help="discriminator learning rate")
    parser.add_argument("--lr_decay_steps", action="store", type=int, required=False, default=1600,
                        help="number of iterations after which learning rate is decayed")
    parser.add_argument("--recon_loss_alpha", action="store", type=float, required=False, default=10.0,
                        help="reconstruction loss scale-factor compared to gan loss for the generator")
    parser.add_argument("--num_feedback_samples", action="store", type=int, required=False, default=6,
                        help="number of random gan samples generated for feedback")
    parser.add_argument("--feedback_frequency", action="store", type=int, required=False, default=500,
                        help="frequency of rendering feedback pose")
    parser.add_argument("--loss_feedback_frequency", action="store", type=int, required=False, default=100,
                        help="frequency of logging loss values to console")
    parser.add_argument("--save_frequency", action="store", type=int, required=False, default=500,
                        help="frequency of taking a snapshot")
    # fmt: on

    parsed_args = parser.parse_args()
    return parsed_args


def main(args: argparse.Namespace) -> None:
    # create a device to train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # log the configuration as a yaml:
    log.info("Logging configuration file ...")
    log_args_config_to_disk(args, args.output_dir)

    if args.train_in_feature_grid_mode:
        # load the trained image_model (feature_grid + decoder_mlp)
        loaded_image_model = load_trained_image_model(args.image_path, device)

        # create the SinganWithImageModel object
        tswim = TwodSinganWithImageModel(
            decoder_mlp=loaded_image_model.decoder_mlp,
            num_stages=args.num_stages,
            output_image_resolution=loaded_image_model.image_resolution,
            gen_inter_channels=args.intermediate_channels,
            use_eql=args.use_eql,
            device=device,
        )

        # train SinganWithImageModel
        tswim.train(
            training_image_model=loaded_image_model,
            use_decoder_mlp_in_training=args.use_decoder_mlp_in_training,
            num_iterations_per_stage=args.num_iterations_per_stage,
            num_dis_steps=args.num_dis_steps,
            num_gen_steps=args.num_gen_steps,
            g_lrate=args.g_lrate,
            d_lrate=args.d_lrate,
            lr_decay_steps=args.lr_decay_steps,
            recon_loss_alpha=args.recon_loss_alpha,
            num_feedback_samples=args.num_feedback_samples,
            feedback_frequency=args.feedback_frequency,
            loss_feedback_frequency=args.loss_feedback_frequency,
            save_frequency=args.save_frequency,
            output_dir=args.output_dir,
        )

    else:
        # create a twod_singan object:
        twod_singan = TwodSingan(
            num_stages=args.num_stages,
            output_resolution=args.output_resolution,
            scale_factor=args.scale_factor,
            device=device,
        )

        # train the model:
        twod_singan.train(
            training_image_path=args.image_path,
            num_iterations_per_stage=args.num_iterations_per_stage,
            num_dis_steps=args.num_dis_steps,
            num_gen_steps=args.num_gen_steps,
            g_lrate=args.g_lrate,
            d_lrate=args.d_lrate,
            lr_decay_steps=args.lr_decay_steps,
            recon_loss_alpha=args.recon_loss_alpha,
            num_feedback_samples=args.num_feedback_samples,
            feedback_frequency=args.feedback_frequency,
            loss_feedback_frequency=args.loss_feedback_frequency,
            save_frequency=args.save_frequency,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main(parse_arguments())
