import argparse
from pathlib import Path

import numpy as np
import torch
from torch.backends import cudnn

from projects.thre3ingan.singans.networks import (
    get_tiny_render_mlp,
    get_default_render_mlp,
    get_big_render_mlp,
)
from thre3d_atom.data.loaders import PosedImagesDataset
from thre3d_atom.modules.volumetric_model.volumetric_model import (
    VolumetricModel,
    VolumetricModelRenderingParameters,
)
from thre3d_atom.rendering.volumetric.voxels import (
    GridLocation,
    get_voxel_size_from_scene_bounds_and_dataset,
)
from thre3d_atom.utils.config_utils import str2bool, log_args_config_to_disk
from thre3d_atom.utils.constants import NUM_COORD_DIMENSIONS
from thre3d_atom.utils.imaging_utils import SceneBounds
from thre3d_atom.utils.logging import log

# Turn on the fast gpu training mode
cudnn.benchmark = True


def parse_arguments() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("Volumetric model training + rendering",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument("-d", "--data_path", action="store", type=Path, required=True,
                        help="path to the data directory")
    parser.add_argument("-o", "--output_dir", action="store", type=Path, required=True,
                        help="path to the output asset directory")

    # dataset related arguments:
    parser.add_argument("--downsample_factor", action="store", type=int, required=False, default=2,
                        help="factor by which the training images are downsampled")

    # feature-grid related arguments
    parser.add_argument("--rgba_only", action="store", type=str2bool, required=False, default=True,
                        help="whether to train a direct rgba model")
    parser.add_argument("--grid_dims", action="store", type=int, required=False, nargs=3,
                        default=(128, 128, 128),
                        # first controls x dimension
                        # middle controls y dimension
                        # last controls z dimension
                        help="spatial dimensions of the feature grid")
    parser.add_argument("--background_msfeature_grid_dims", action="store", type=int, required=False, nargs=3,
                        default=None,  # (1024, 2048, 64),  # otherwise use (1024, 2048, 64)
                        help="spatial dimensions of the feature-grid used in the background. "
                             "Note that passing a None here disables the MSFG based background.")
    parser.add_argument("--grid_location", action="store", type=float, required=False, nargs=3,
                        default=(0.0, 0.0, 0.0),
                        # x (+ve) is oriented towards right
                        # y (+ve) is oriented towards up
                        # z (+ve) is oriented towards out
                        help="location of the center of the voxel grid")
    parser.add_argument("--feature_dims", action="store", type=int, required=False, default=32,
                        help="number of feature dimensions in the feature grid. (Uses 4 if rgba_only is true)")
    parser.add_argument("--feature_embedding_dims", action="store", type=int, required=False, default=0,
                        help="number of positional encoding frequencies applied to the featured of feature-grid")
    parser.add_argument("--dir_embedding_dims", action="store", type=int, required=False, default=0,
                        help="number of positional encoding frequencies applied to the direction vectors")
    parser.add_argument("--normalize_features", action="store", type=str2bool, required=False, default=False,
                        help="whether to norm-normalize the decoded feature vectors from the feature-grid")
    parser.add_argument("--use_hierarchical_sampling", action="store", type=str2bool, required=False, default=False,
                        help="whether to use a secondary render_mlp for performing hierarchical sampling")
    parser.add_argument("--unit_normalize_scene_scale", action="store", type=str2bool, required=False, default=False,
                        help="whether to normalize the scene to -1, 1 range")
    parser.add_argument("--use_background_mlp", action="store", type=str2bool, required=False, default=False,
                        help="whether to use a third (potential) render_mlp for modelling the scene background")
    parser.add_argument("--mlp_size", action="store", type=str, required=False, default="small",
                        help="which mlp to use: one of `small`, `medium` or `big`")
    parser.add_argument("--use_sparsity_regularizer", action="store", type=str2bool, required=False, default=False,
                        help="whether to use the sparsity-regularizer on density when training")

    # rendering arguments
    parser.add_argument("--num_rays_chunk", action="store", type=int, required=False, default=8192,
                        help="number of rays in a single chunk (for training + rendering)")
    parser.add_argument("--num_points_chunk", action="store", type=int, required=False, default=65536,
                        help="number of points processed in a single chunk (for training + rendering)"
                             "This is only used in the case of a feature-grid + MLP model")
    parser.add_argument("--num_samples_per_ray", action="store", type=int, required=False, default=256,
                        help="number of uniform points sampled per casted ray into the volume")
    parser.add_argument("--num_fine_samples_per_ray", action="store", type=int, required=False, default=128,
                        help="number of weighted points sampled on casted ray for secondary render_mlp")

    # training arguments
    parser.add_argument("--batch_size", action="store", type=int, required=False, default=4,
                        help="batch size used for training the volume model")
    parser.add_argument("--num_stages", action="store", type=int, required=False, default=4,
                        help="number of stages in the coarse-to-fine training scheme")
    parser.add_argument("--num_iterations_per_stage", action="store", type=int, required=False,
                        nargs="+", default=[2000, 2000, 2000, 2000],
                        help="number of iterations per stage to train the model. Note this "
                             "should have as many values as num_stages requested.")
    parser.add_argument("--scale_factor", action="store", type=float, required=False, default=2.0,
                        help="factor by which the feature grid is upsampled "
                             "after every stage in coarse-to-fine training scheme")
    # The feature_scale needs to be 100.0 for ReluField grid optimization
    parser.add_argument("--feature_scale", action="store", type=float, required=False, default=10.0,
                        help="scale value used for equalized learning rate training of the feature-grid")
    parser.add_argument("--lr_per_stage", action="store", type=float, required=False,
                        nargs="+", default=[0.03, 0.03, 0.03, 0.03],
                        help="learning rate used for stagewise training")
    parser.add_argument("--lr_decay_steps_per_stage", action="store", type=int, required=False,
                        nargs="+", default=[1000, 1000, 1000, 1000],
                        help="number of steps after which to decay the learning rate during each stage")
    parser.add_argument("--lr_decay_gamma_per_stage", action="store", type=float, required=False,
                        nargs="+", default=[0.1, 0.1, 0.1, 0.1],
                        help="value of gamma used for exponential learning rate decay per stage")
    parser.add_argument("--num_workers", action="store", type=int, required=False, default=4,
                        help="number of worker processes used for loading the data using the dataloader")
    parser.add_argument("--save_frequency", action="store", type=int, required=False, default=500,
                        help="frequency of taking a snapshot")
    parser.add_argument("--testing_frequency", action="store", type=int, required=False, default=500,
                        help="frequency of performing testing")
    parser.add_argument("--feedback_frequency", action="store", type=int, required=False, default=100,
                        help="frequency of rendering feedback pose")
    parser.add_argument("--loss_feedback_frequency", action="store", type=int, required=False, default=10,
                        help="frequency of logging loss values to console")
    parser.add_argument("--overridden_scene_bounds", action="store", type=float, required=False, default=None, nargs=2,
                        help="manually overridden scene bounds")
    # The manually overridden_grid_size is needed for the nerf-synthetic scenes to work correctly
    parser.add_argument("--overridden_grid_size", action="store", type=float, required=False,
                        default=None, nargs=3,
                        help="manually overridden grid_size")
    parser.add_argument("--white_bkgd", action="store", type=str2bool, required=False, default=False,
                        help="whether to use white background for training with synthetic scenes. "
                             "Can be used with real scenes as well, but god knows what you'll get :D")
    parser.add_argument("--slacken_scene_bounds", action="store", type=str2bool, required=False, default=False,
                        help="use a 1 unit margin for near and far rendering planes scene_bounds")
    parser.add_argument("--hybrid_rgba_feature_grid_mode", action="store", type=str2bool, required=False, default=False,
                        help="whether to use hybrid feature-grid (with rgba localization) mode")
    parser.add_argument("--separate_train_test_folders", action="store", type=str2bool, required=False, default=False,
                        help="whether the data is stored in separate train and test folders")
    parser.add_argument("--use_relu_field", action="store", type=str2bool, required=False, default=True,
                        help="whether to use relu_fields or revert to traditional grid model")
    parser.add_argument("--use_sh", action="store", type=str2bool, required=False, default=False,
                        help="whether to use spherical harmonics for modelling the view-dep-fx")
    parser.add_argument("--sh_degree", action="store", type=int, required=False, default=0,
                        help="degree of the spherical harmonics coefficients to be used. "
                             "Supported values: [0, 1, 2, 3]")
    parser.add_argument("--apply_diffuse_render_reg", action="store", type=str2bool, required=False, default=True,
                        help="whether to apply diffuse_render regularization")

    # Importance sampling modes and their options (All are disabled by default)
    parser.add_argument("--use_vcbis", action="store", type=str2bool, required=False, default=False,
                        help="whether to use voxel-crop based importance sampling during training")
    parser.add_argument("--patch_percentage_vcbis", action="store", type=float, required=False, default=2.5,
                        help="size of patches used for vcbis measured in percentage of Image resolution")
    parser.add_argument("--use_fris", action="store", type=str2bool, required=False, default=False,
                        help="whether to use fully random importance sampling during training")
    parser.add_argument("--use_mselis", action="store", type=str2bool, required=False, default=False,
                        help="whether to use mse-loss weighted importance sampling during training")
    parser.add_argument("--mselis_random_percentage", action="store", type=float, required=False, default=30.0,
                        help="percentage of random-rays used along with the is-rays per iteration")
    parser.add_argument("--mselis_loss_weights_gamma", action="store", type=float, required=False, default=1.0,
                        help="value of gamma used to scale the loss-weights distribution. "
                             "Note that this operation is applied on un-normalized weights, hence,"
                             "values > 1.0 make it peakier and < 1.0 make it flatter.")

    # Miscellaneous modes
    parser.add_argument("--fast_debug_mode", action="store", type=str2bool, required=False, default=False,
                        help="whether to use the fast debug mode while training "
                             "(skips testing and some lengthy visualizations)")
    parser.add_argument("--profiling_mode", action="store", type=str2bool, required=False, default=False,
                        help="whether to use the profiling mode while training "
                             "(logs runtime taken by different parts of the code)")
    # fmt: on

    parsed_args = parser.parse_args()
    return parsed_args


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the PosedImages dataset
    # noinspection PyPep8Naming
    if args.separate_train_test_folders:
        train_dataset = PosedImagesDataset(
            args.data_path / "train",
            args.data_path / "train_camera_params.json",
            image_data_range=(0, 1),
            test_percentage=0.0,
            test_mode=False,
            unit_normalize_scene_scale=args.unit_normalize_scene_scale,
            downsample_factor=args.downsample_factor,
            rgba_white_bkgd=args.white_bkgd,
        )
        test_dataset = PosedImagesDataset(
            args.data_path / "test",
            args.data_path / "test_camera_params.json",
            image_data_range=(0, 1),
            test_percentage=0.0,
            test_mode=False,
            unit_normalize_scene_scale=args.unit_normalize_scene_scale,
            downsample_factor=args.downsample_factor,
            rgba_white_bkgd=args.white_bkgd,
        )
    else:
        train_dataset, test_dataset = [
            PosedImagesDataset(
                args.data_path / "images",
                args.data_path / "camera_params.json",
                image_data_range=(0, 1),
                test_percentage=10,
                test_mode=test_mode,
                unit_normalize_scene_scale=args.unit_normalize_scene_scale,
                downsample_factor=args.downsample_factor,
                rgba_white_bkgd=args.white_bkgd,
            )
            for test_mode in (False, True)
        ]

    render_params = VolumetricModelRenderingParameters(
        num_rays_chunk=args.num_rays_chunk,
        num_points_chunk=args.num_points_chunk,
        num_samples_per_ray=args.num_samples_per_ray,
        num_fine_samples_per_ray=args.num_fine_samples_per_ray,
        perturb_sampled_points=True,
        density_noise_std=0.0,
        white_bkgd=args.white_bkgd,
    )

    if args.overridden_scene_bounds is not None:
        near, far = args.overridden_scene_bounds
        train_dataset.scene_bounds = SceneBounds(near, far)
        test_dataset.scene_bounds = SceneBounds(near, far)

    if args.slacken_scene_bounds:
        old_scene_bounds = train_dataset.scene_bounds
        train_dataset.scene_bounds = SceneBounds(
            old_scene_bounds.near - 1, old_scene_bounds.far + 1
        )
        test_dataset.scene_bounds = SceneBounds(
            old_scene_bounds.near - 1, old_scene_bounds.far + 1
        )

    # compute the grid size:
    voxel_size = get_voxel_size_from_scene_bounds_and_dataset(
        train_dataset, max(args.grid_dims), train_dataset.scene_bounds
    )
    grid_size = (
        voxel_size.x_size * args.grid_dims[0],
        voxel_size.y_size * args.grid_dims[1],
        voxel_size.z_size * args.grid_dims[2],
    )

    if args.overridden_grid_size is not None:
        grid_size = args.overridden_grid_size

    mlp_size = args.mlp_size.lower()
    if mlp_size == "small":
        render_mlp_creator = get_tiny_render_mlp
    elif mlp_size == "medium":
        render_mlp_creator = get_default_render_mlp
    elif mlp_size == "big":
        render_mlp_creator = get_big_render_mlp
    else:
        raise ValueError("Cannot interpret the requested MLP size")

    vol_mod = VolumetricModel(
        grid_dims=args.grid_dims,
        feature_dims=args.feature_dims,
        grid_size=grid_size,
        grid_center=GridLocation(*args.grid_location),
        hybrid_rgba_mode=args.hybrid_rgba_feature_grid_mode,
        render_params=render_params,
        render_mlp=render_mlp_creator(
            feature_size=args.feature_dims,
            feature_embeddings_dims=args.feature_embedding_dims,
            dir_embedding_dims=args.dir_embedding_dims,
            normalize_features=args.normalize_features,
        )
        if not args.rgba_only
        else None,
        fine_render_mlp=render_mlp_creator(
            feature_size=args.feature_dims,
            feature_embeddings_dims=args.feature_embedding_dims,
            dir_embedding_dims=args.dir_embedding_dims,
            normalize_features=args.normalize_features,
        )
        if not args.rgba_only and args.use_hierarchical_sampling
        else None,
        background_render_mlp=render_mlp_creator(
            feature_size=NUM_COORD_DIMENSIONS,
            feature_embeddings_dims=10,  # default for NeRF
            dir_embedding_dims=4,
            normalize_features=False,
        )
        if args.use_background_mlp
        else None,
        background_msfeature_grid_dims=args.background_msfeature_grid_dims,
        feature_scale=args.feature_scale,
        use_sh=args.use_sh,
        use_relu_field=args.use_relu_field,
        sh_degree=args.sh_degree,
        apply_diffuse_render_reg=args.apply_diffuse_render_reg,
        device=device,
    )

    # log the configuration as a yaml:
    log.info("Logging configuration file ...")
    log_args_config_to_disk(args, args.output_dir)

    num_points_vcbis = None
    if args.use_vcbis:
        patch_height = int(
            np.round(
                (args.patch_percentage_vcbis / 100)
                * train_dataset.camera_intrinsics.height
            )
        )
        patch_width = int(
            np.round(
                (args.patch_percentage_vcbis / 100)
                * train_dataset.camera_intrinsics.width
            )
        )
        num_points_vcbis = int(
            np.ceil(
                args.num_rays_chunk / (len(train_dataset) * patch_height * patch_width)
            )
        )

    # train the volume_model
    vol_mod.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        image_batch_cache_size=args.batch_size,
        num_stages=args.num_stages,
        lr_per_stage=args.lr_per_stage,
        lr_decay_gamma_per_stage=args.lr_decay_gamma_per_stage,
        lr_decay_steps_per_stage=args.lr_decay_steps_per_stage,
        num_iterations_per_stage=args.num_iterations_per_stage,
        scale_factor=args.scale_factor,
        save_freq=args.save_frequency,
        testing_freq=args.testing_frequency,
        feedback_freq=args.feedback_frequency,
        output_dir=args.output_dir,
        loss_feedback_freq=args.loss_feedback_frequency,
        num_workers=args.num_workers,
        use_voxel_crop_based_sampling=args.use_vcbis,
        num_points_vcbis=num_points_vcbis,
        patch_percentage_vcbis=args.patch_percentage_vcbis,
        use_fully_random_importance_sampling=args.use_fris,
        use_mse_loss_weighted_importance_sampling=args.use_mselis,
        mselis_random_percentage=args.mselis_random_percentage,
        mselis_loss_weights_gamma=args.mselis_loss_weights_gamma,
        fast_debug_mode=args.fast_debug_mode,
        profiling=args.profiling_mode,
    )

    # # encode the videos for the training-timelapses:
    # rendered_feedback_dir = (
    #     args.output_dir / "training_logs/rendered_output/rendered_feedback"
    # )
    #
    # # encode the diffuse and specular versions of the training-timelapse
    # modes = ("diffuse", "specular") if args.use_sh else ("specular",)
    # for mode in modes:
    #     log.info(f"Encoding the video for the {mode}-version of training timelapse")
    #     video_output_path = args.output_dir / f"{mode}_training_timelapse.mp4"
    #     write_video(
    #         frames_path=rendered_feedback_dir,
    #         output_path=video_output_path,
    #         frames_pattern=f"{mode}_*.png",
    #         sort_key=lambda x: int(x.name.split("_")[1].split(".")[0]),
    #         fps=6,
    #     )

    log.info("!See you next time!")


if __name__ == "__main__":
    main(parse_arguments())
