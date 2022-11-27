import argparse
import sys
from pathlib import Path
from typing import List

from torch.backends import cudnn

from thre3d_atom.data.loaders import PosedImagesDataset
from thre3d_atom.modules.nerf import Nerf, NerfRenderingParameters
from thre3d_atom.modules.nerf import get_sota_nerf_net
from thre3d_atom.utils.config_utils import str2bool, log_args_config_to_disk
from thre3d_atom.utils.logging import log

# Turn on benchmarking for fast training
cudnn.benchmark = True


def parse_arguments(args: List[str]) -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("NeRF based rendering model trainer",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument("-d", "--data_path", action="store", type=Path, required=True,
                        help="path to the data directory")
    parser.add_argument("-o", "--output_dir", action="store", type=Path, required=True,
                        help="path to the output directory")

    # Optional arguments
    # Nerf network related arguments
    parser.add_argument("--num_rays_chunk", action="store", type=int, required=False, default=1024,
                        help="number of rays in a single chunk (similar to batch size)")
    parser.add_argument("--num_points_chunk", action="store", type=int, required=False, default=64 * 1024,
                        help="number of points processed in a single chunk. Needed to avoid OOM and for better perf")
    parser.add_argument("--num_coarse_samples", action="store", type=int, required=False, default=64,
                        help="number of coarse samples along the casted rays")
    parser.add_argument("--num_fine_samples", action="store", type=int, required=False, default=128,
                        help="number of fine samples along the casted rays")
    parser.add_argument("--training_density_noise_std", action="store", type=float, required=False, default=1.0,
                        help="std of the noise added to the network predicted density")
    parser.add_argument("--perturb_coarse_sampled_points", action="store", type=str2bool, required=False, default=True,
                        help="whether to perturb the coarse sampled points along the rays")
    parser.add_argument("--linear_disparity_sampling", action="store", type=str2bool, required=False, default=False,
                        help="whether to sample points with linear disparity")
    parser.add_argument("--ndcize_rays", action="store", type=str2bool, required=False, default=False,
                        help="whether to ndcize the casted rays")
    parser.add_argument("--use_viewdirs", action="store", type=str2bool, required=False, default=False,
                        help="whether to model the view-dependent effects using view-directions")
    parser.add_argument("--use_fine_net", action="store", type=str2bool, required=False, default=True,
                        help="whether to use secondary fine network. "
                             "Note that all the params related to fine net "
                             "would be ignored if this is set to False")

    # training arguments
    parser.add_argument("--downsample_factor", action="store", type=int, required=False, default=1,
                        help="downsample factor for the images in the dataset")
    parser.add_argument("--unit_normalize_scene_scale", action="store", type=str2bool, required=False, default=False,
                        help="whether to normalize the scene's scale by it's max norm. "
                             "Thereby all the camera locations + scene contents fall in the [-1, 1] cube range")
    parser.add_argument("--image_batch_cache_size", action="store", type=int, required=False, default=8,
                        help="number of images to keep in cache for faster training")
    parser.add_argument("--learning_rate", action="store", type=float, required=False, default=0.003,
                        help="learning rate for the training of the network")
    parser.add_argument("--lr_decay_steps", action="store", type=int, required=False, default=250000,
                        help="number of training iterations after which learning rate is decayed exponentially",)
    parser.add_argument("--lr_decay_gamma", action="store", type=float, required=False, default=0.1,
                        help="value of gamma used for exponential learning rate decay", )
    parser.add_argument("--test_percentage", action="store", type=float, required=False, default=10.0,
                        help="percentage of the dataset to be used as a heldout set for testing")
    parser.add_argument("--num_iterations", action="store", type=int, required=False, default=3e5,
                        help="number of iterations to train the model for")
    parser.add_argument("--save_frequency", action="store", type=int, required=False, default=1000,
                        help="frequency of saving the model")
    parser.add_argument("--feedback_frequency", action="store", type=int, required=False, default=500,
                        help="frequency of rendering the model for visual feedback")
    parser.add_argument("--loss_feedback_frequency", action="store", type=int, required=False, default=100,
                        help="frequency of showing current loss values on the console")
    parser.add_argument("--testing_frequency", action="store", type=int, required=False, default=5000,
                        help="frequency of computing the test metrics on the heldout set during training")
    parser.add_argument("--num_workers", action="store", type=int, required=False, default=4,
                        help="num_workers (processes) used for loading the training data")
    parser.add_argument("--verbose_rendering", action="store", type=str2bool, required=False, default=True,
                        help="whether to show verbose progress-bar during rendering")
    parser.add_argument("--fast_debug_mode", action="store", type=str2bool, required=False, default=False,
                        help="whether to run the training in fast-debug mode (skips some visualizations)")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def train_nerf(args: argparse.Namespace) -> None:
    log.info("Welcome to NeRF rendering")

    # create a NeRF model
    nerf = Nerf(
        nerf_net_coarse=get_sota_nerf_net(),
        nerf_net_fine=get_sota_nerf_net() if args.use_fine_net else None,
        render_params=NerfRenderingParameters(
            num_rays_chunk=args.num_rays_chunk,
            num_points_chunk=args.num_points_chunk,
            num_coarse_samples=args.num_coarse_samples,
            num_fine_samples=args.num_fine_samples,
            perturb_coarse_sampled_points=args.perturb_coarse_sampled_points,
            linear_disparity_sampling=args.linear_disparity_sampling,
            training_density_noise_std=args.training_density_noise_std,
            ndcize_rays=args.ndcize_rays,
            use_viewdirs=args.use_viewdirs,
        ),
    )

    # load the data
    train_dataset = PosedImagesDataset(
        images_dir=args.data_path / "images",
        camera_params_json=args.data_path / "camera_params.json",
        image_data_range=(0, 1),
        downsample_factor=args.downsample_factor,
        test_percentage=args.test_percentage,
        test_mode=False,
        unit_normalize_scene_scale=args.unit_normalize_scene_scale,
    )
    test_dataset = PosedImagesDataset(
        images_dir=args.data_path / "images",
        camera_params_json=args.data_path / "camera_params.json",
        image_data_range=(0, 1),
        downsample_factor=args.downsample_factor,
        test_percentage=args.test_percentage,
        test_mode=True,
        unit_normalize_scene_scale=args.unit_normalize_scene_scale,
    )

    # log the complete configuration to a file
    log.info("logging config file to the disk ...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_args_config_to_disk(args, args.output_dir)

    # train the nerf model
    nerf.train(
        train_dataset,
        test_dataset,
        image_batch_cache_size=args.image_batch_cache_size,
        num_iterations=args.num_iterations,
        save_freq=args.save_frequency,
        feedback_freq=args.feedback_frequency,
        learning_rate=args.learning_rate,
        lr_decay_steps=args.lr_decay_steps,
        lr_decay_gamma=args.lr_decay_gamma,
        output_dir=args.output_dir,
        loss_feedback_freq=args.loss_feedback_frequency,
        num_workers=args.num_workers,
        test_freq=args.testing_frequency,
        verbose_rendering=args.verbose_rendering,
        fast_debug_mode=args.fast_debug_mode,
    )
    log.info("!See you next time!")


def main() -> None:
    train_nerf(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
