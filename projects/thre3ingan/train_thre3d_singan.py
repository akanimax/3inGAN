import argparse
import sys
from pathlib import Path

import torch.cuda
from torch.backends import cudnn

from thre3d_atom.data.loaders import PosedImagesDataset
from projects.thre3ingan.singans.networks import (
    Thre3dGenerator,
)
from projects.thre3ingan.singans.thre3d_singan import (
    Thre3dSinGan,
    Thre3dSinGanRenderingParameters,
)
from thre3d_atom.rendering.volumetric.voxels import VoxelSize, GridLocation
from thre3d_atom.utils.config_utils import str2bool
from thre3d_atom.utils.imaging_utils import SceneBounds
from thre3d_atom.utils.logging import log

# Turn on benchmarking for fast training
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments(args: str) -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("Thre3d SinGAN model trainer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument("-d",
                        "--data_path", action="store", type=Path, required=True, help="path to the data directory")
    parser.add_argument("-o",
                        "--output_dir", action="store", type=Path, required=True, help="path to the output directory")

    # Optional arguments
    # Nerf network related arguments
    parser.add_argument("--num_rays_chunk", action="store", type=int, required=False, default=1024,
                        help="number of rays in a single chunk (similar to batch size)")
    parser.add_argument("--num_samples_per_ray", action="store", type=int, required=False, default=64,
                        help="number of samples along the casted rays")
    parser.add_argument("--density_noise_std", action="store", type=float, required=False, default=0.0,
                        help="std of the noise added to the network predicted density")
    parser.add_argument("--perturb_sampled_points", action="store", type=str2bool, required=False, default=True,
                        help="whether to perturb the coarse sampled points along the rays")

    # training arguments
    parser.add_argument("--train_reconstruction_only", action="store", type=str2bool, required=False, default=False,
                        help="whether to train only reconstruction for SinGAN")
    parser.add_argument("--num_recon_steps", action="store", type=int, required=False, default=2,
                        help="Number of reconstruction steps to be performed per training iteration")
    parser.add_argument("--num_gan_steps", action="store", type=int, required=False, default=1,
                        help="Number of gan steps to be performed per training iteration")
    parser.add_argument("--downsample_factor", action="store", type=int, required=False, default=1,
                        help="downsample factor for the images in the dataset")
    parser.add_argument("--network_depth", action="store", type=int, required=False, default=6,
                        help="depth (size) of the generator network")
    parser.add_argument("--use_trilinear_upsampling", action="store", type=str2bool, required=False, default=False,
                        help="Whether to use trilinear upsampling in the generator or nearest-neighbour")
    parser.add_argument("--use_dists_in_rendering", action="store", type=str2bool, required=False, default=True,
                        help="Whether to use delta-dists in rendering (both training and inference)")
    parser.add_argument("--use_emb_loss_for_recon", action="store", type=str2bool, required=False, default=False,
                        help="whether to use gan based reconstruction loss")
    parser.add_argument("--patch_size", action="store", type=int, required=False, default=32,
                        help="patch-size of the discriminator")
    parser.add_argument("--base_feature_grid_dims", action="store", type=int, required=False, nargs=3,
                        default=(4, 4, 4), help="resolution of the base feature grid, i.e. 1st layer.")
    parser.add_argument("--image_batch_cache_size", action="store", type=int, required=False, default=8,
                        help="number of images to keep in cache for faster training")
    parser.add_argument("--lr_decay_steps", action="store", type=int, required=False, default=1000,
                        help="number of training iterations after which learning rate is decayed exponentially")
    parser.add_argument("--num_iterations_per_stage", action="store", type=int, required=False, default=500,
                        help="number of iterations to train the model for")
    parser.add_argument("--save_frequency", action="store", type=int, required=False, default=1000,
                        help="frequency of saving the model")
    parser.add_argument("--testing_frequency", action="store", type=int, required=False, default=1000,
                        help="frequency of testing the reconstruction performance of the model")
    parser.add_argument("--feedback_frequency", action="store", type=int, required=False, default=50,
                        help="frequency of rendering the model for visual feedback")
    parser.add_argument("--scene_bounds", action="store", type=float, required=False, nargs=2, default=None,
                        help="manually overridden scene bounds for training the model")
    parser.add_argument("--voxel_size", action="store", type=float, required=False, nargs=3, default=None,
                        help="manually overridden voxel_size for training the model")
    parser.add_argument("--grid_location", action="store", type=float, required=False, nargs=3, default=None,
                        help="manually overridden GridLocation of the Feature Grid")
    parser.add_argument("--loss_feedback_frequency", action="store", type=int, required=False, default=100,
                        help="frequency of showing current loss values on the console")
    parser.add_argument("--verbose_rendering", action="store", type=str2bool, required=False, default=True,
                        help="whether to show verbose progress-bar during rendering")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def train_thre3d_singan(args: argparse.Namespace) -> None:
    log.info("Welcome to Thre3d SinGan Training")

    # create a Thre3dSinGAN model
    thre3d_singan = Thre3dSinGan(
        thre3d_gen=Thre3dGenerator(
            num_stages=args.network_depth,
            base_feature_grid_dims=args.base_feature_grid_dims,
            device=device,
            fmap_base=2048,
            fmap_min=128,
            output_feature_size=32,
            upsample_factor=1.3333,
            use_trilinear_upsampling=args.use_trilinear_upsampling,
            use_dists_in_rendering=args.use_dists_in_rendering,
        ),
        render_params=Thre3dSinGanRenderingParameters(
            num_rays_chunk=args.num_rays_chunk,
            num_samples_per_ray=args.num_samples_per_ray,
            perturb_sampled_points=args.perturb_sampled_points,
            density_noise_std=args.density_noise_std,
        ),
        device=device,
    )

    # create a scene's camera-rays visualization
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_reconstruction_only:
        # train the thre3ingan in on reconstruction mode if requested:
        train_dataset, test_dataset = [
            PosedImagesDataset(
                images_dir=args.data_path / "images",
                camera_params_json=args.data_path / "camera_params.json",
                test_percentage=10.0,
                test_mode=test_mode,
                image_data_range=(0, 1),
                downsample_factor=args.downsample_factor,
            )
            for test_mode in (False, True)
        ]

        thre3d_singan.train_reconstruction(
            train_dataset,
            test_dataset,
            image_batch_cache_size=args.image_batch_cache_size,
            num_iterations=args.num_iterations_per_stage,
            save_freq=args.save_frequency,
            feedback_freq=args.feedback_frequency,
            test_freq=args.testing_frequency,
            lr_decay_steps=args.lr_decay_steps,
            gen_learning_rate=0.003,
            gen_renderer_learning_rate=0.003,
            output_dir=args.output_dir,
            loss_feedback_freq=args.loss_feedback_frequency,
            verbose_rendering=args.verbose_rendering,
            num_workers=4,
            scene_bounds=SceneBounds(*args.scene_bounds)
            if args.scene_bounds is not None
            else None,
            voxel_size=VoxelSize(*args.voxel_size)
            if args.voxel_size is not None
            else None,
            grid_location=GridLocation(*args.grid_location)
            if args.grid_location is not None
            else GridLocation(),
        )
    else:
        # train the thre3ingan model
        thre3d_singan.train_singan(
            dataset_dir=args.data_path,
            original_dataset_downsample_factor=args.downsample_factor,
            num_iterations_per_stage=args.num_iterations_per_stage,
            num_feedback_images=6,
            patch_size=args.patch_size,
            image_batch_cache_size=args.image_batch_cache_size,
            save_freq=args.save_frequency,
            feedback_freq=args.feedback_frequency,
            test_freq=args.testing_frequency,
            gen_learning_rate=0.001,
            gen_renderer_learning_rate=0.001,
            dis_learning_rate=0.001,
            lr_decay_steps=args.lr_decay_steps,
            output_dir=args.output_dir,
            loss_feedback_freq=args.loss_feedback_frequency,
            verbose_rendering=args.verbose_rendering,
            num_workers=4,
            scene_bounds=SceneBounds(*args.scene_bounds)
            if args.scene_bounds is not None
            else None,
            voxel_size=VoxelSize(*args.voxel_size)
            if args.voxel_size is not None
            else None,
            grid_location=GridLocation(*args.grid_location)
            if args.grid_location is not None
            else GridLocation(),
            use_gan_based_perceptual_recon_loss=args.use_emb_loss_for_recon,
            num_gan_steps=args.num_gan_steps,
            num_recon_steps=args.num_recon_steps,
        )

    log.info("!See you next time!")


def main() -> None:
    train_thre3d_singan(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
