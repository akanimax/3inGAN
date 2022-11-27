import argparse
import sys
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
import torch
from pytorch_fid.fid_score import compute_statistics_of_path, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from tqdm import tqdm

from projects.thre3ingan.singans import (
    create_thre3d_singan_with_direct_supervision_from_saved_model,
)
from thre3d_atom.modules.volumetric_model.volumetric_model import (
    create_vol_mod_from_saved_model,
)
from thre3d_atom.utils.imaging_utils import pose_spherical, to8b, CameraIntrinsics
from thre3d_atom.utils.logging import log


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Thre3d SinGAN Reconstruction 360 render demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("--model_path", action="store", type=Path, required=False,
                        help="Path to the trained thre3ingan model")
    parser.add_argument("--fake_samples_path", action="store", type=Path, required=True,
                        help="Path to save the generated fake-samples using the thre3d_singan_model")

    # optional arguments
    parser.add_argument("--dataset_path", action="store", type=Path, required=False, default=None,
                        help="Path to the GT image set (used if provided)")
    parser.add_argument("--gt_model_path", action="store", type=Path, required=False, default=None,
                        help="Path to the reconstructed 3D scene vol-mod")
    parser.add_argument("--num_real_samples", action="store", type=int, required=False, default=2048,
                        help="number of real images rendered from the vol-mod."
                             "Used only if gt_model_path is not None")
    parser.add_argument("--stage", action="store", type=int, required=False, default=None,
                        help="stage from which random samples are to be drawn")
    parser.add_argument("--num_fake_samples", action="store", type=int, required=False, default=2048,
                        help="number of fake images generated using the thre3d-singan")
    parser.add_argument("--batch_size", action="store", type=int, required=False, default=16,
                        help="batch_size of images used for InceptionNet forward pass")
    parser.add_argument("--inception_dims", action="store", type=int, required=False, default=2048,
                        help="number of dimension of the InceptionNet extracted features")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def generate_real_samples(
    model_path: Path,
    gt_samples_path: Path,
    num_real_samples: int,
    device: torch.device = torch.device("cpu"),
) -> None:
    gt_samples_path.mkdir(parents=True, exist_ok=True)
    possible_images_num = len(list(gt_samples_path.iterdir()))

    if (gt_samples_path / "cache.npz").exists():
        log.info(
            f"cache for gt samples found at: {gt_samples_path / 'cache.npz'}, "
            f"so skipping real sample generation"
        )
        return

    if possible_images_num >= num_real_samples:
        log.info(
            f"Already found generated samples at: {gt_samples_path}, "
            f"so skipping real sample generation"
        )
        return

    vol_mod, extra_info = create_vol_mod_from_saved_model(model_path, device=device)
    # don't use the background mlp for rendering
    vol_mod.background_render_mlp = None

    # additional information required for rendering
    hem_rad = extra_info["hemispherical_radius"]
    scene_bounds = extra_info["scene_bounds"]
    camera_intrinsics = extra_info["camera_intrinsics"]

    # override the camera intrinsics to have a height and width of 128 x 128
    modified_camera_intrinsics = CameraIntrinsics(
        height=camera_intrinsics.height,
        width=camera_intrinsics.width,
        focal=camera_intrinsics.focal,
    )

    log.info("Generating random views for the given vol-mod")
    for sample_num in tqdm(range(1, num_real_samples + 1)):
        random_yaw = np.random.uniform(low=0, high=360)
        random_pitch = np.random.uniform(low=-90, high=0)
        pose = pose_spherical(random_yaw, random_pitch, hem_rad)
        render_out = vol_mod.render(
            camera_intrinsics=modified_camera_intrinsics,
            camera_pose=pose,
            scene_bounds=scene_bounds,
            verbose=False,
        )

        colour_sample = render_out.colour.numpy()
        imageio.imwrite(gt_samples_path / f"{sample_num}.png", to8b(colour_sample))


def generate_fake_samples(
    model_path: Path,
    fake_samples_path: Path,
    num_fake_samples: int,
    stage: Optional[int] = None,
) -> None:
    fake_samples_path.mkdir(parents=True, exist_ok=True)

    if (fake_samples_path / "cache.npz").exists():
        log.info(f"cache for fake samples found at: {fake_samples_path / 'cache.npz'}")
        return

    tsds, extra_info = create_thre3d_singan_with_direct_supervision_from_saved_model(
        model_path
    )
    tsds.render_params.num_rays_chunk = 32768
    hem_rad = extra_info["camera_hemisphere_radius"]

    # override the camera intrinsics to have a height and width of 128 x 128
    modified_camera_intrinsics = CameraIntrinsics(
        height=tsds.render_params.camera_intrinsics.height,
        width=tsds.render_params.camera_intrinsics.width,
        focal=tsds.render_params.camera_intrinsics.focal,
    )

    log.info("Generating random samples for the trained thre3ingan")
    for sample_num in tqdm(range(1, num_fake_samples + 1)):
        random_yaw = np.random.uniform(low=0, high=360)
        random_pitch = np.random.uniform(low=-90, high=0)
        pose = pose_spherical(random_yaw, random_pitch, hem_rad)
        render_out = tsds.render_random_sample(
            hem_rad=hem_rad,
            camera_pose=pose,
            camera_intrinsics=modified_camera_intrinsics,
            scene_bounds=tsds.render_params.scene_bounds,
            stage=stage,
        )

        colour_sample = render_out.colour.numpy()
        imageio.imwrite(fake_samples_path / f"{sample_num}.png", to8b(colour_sample))


# noinspection DuplicatedCode
def compute_fid(
    path_1: Path,
    path_2: Path,
    cache_p1_stats: bool = True,
    cache_p2_stats: bool = True,
    batch_size: int = 16,
    inception_dims: int = 2048,
    device: torch.device = torch.device("cpu"),
) -> float:

    # create the inception model object for fid calculation
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_dims]
    model = InceptionV3([block_idx]).to(device)

    # compute inception gaussian for path_1 and path_2:
    gaussians_list = []
    for path, cache_stats in (
        (path_1, cache_p1_stats),
        (path_2, cache_p2_stats),
    ):
        log.info(f"Computing the inception statistics for: {path}")

        if (path / "cache.npz").exists():
            path = path / "cache.npz"
            log.info(f"Stats cache found at: {path}")

        mu, sigma = compute_statistics_of_path(
            str(path), model, batch_size, inception_dims, device
        )

        if (
            cache_stats
            and not str(path).endswith(".npz")
            and not (path / "cache.npz").exists()
        ):
            # cache the computed mu_1 and sigma_1 stats:
            cache_file = path / "cache.npz"
            np.savez(
                str(cache_file),
                mu=mu,
                sigma=sigma,
            )
        gaussians_list.append((mu, sigma))

    # compute the fid obtained for the two paths:
    (mu1, sig1), (mu2, sig2) = gaussians_list
    fid_value = calculate_frechet_distance(mu1, sig1, mu2, sig2)

    # log the computed fid_value to the console:
    log.info(f"Computed FID score: {fid_value : .5f}")

    return fid_value


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments(sys.argv[1:])

    if args.dataset_path is None and args.gt_model_path is None:
        raise ValueError("Neither dataset_path nor gt_model_path is provided")
    elif args.dataset_path is not None and args.gt_model_path is not None:
        raise ValueError(
            "Both dataset_path and gt_model_path are provided. Please check"
        )

    if args.dataset_path is not None:
        gt_path = args.dataset_path
    else:
        gt_path = args.gt_model_path.parent / "generated_samples"
        gt_path.mkdir(parents=True, exist_ok=True)

        # render samples for the ground_truth
        generate_real_samples(
            args.gt_model_path,
            gt_path,
            num_real_samples=args.num_real_samples,
            device=device,
        )

    generate_fake_samples(
        model_path=args.model_path,
        fake_samples_path=args.fake_samples_path,
        num_fake_samples=args.num_fake_samples,
        stage=args.stage,
    )
    compute_fid(
        path_1=gt_path,
        path_2=args.fake_samples_path,
        cache_p1_stats=True,
        cache_p2_stats=True,
        batch_size=args.batch_size,
        inception_dims=args.inception_dims,
        device=device,
    )


if __name__ == "__main__":
    main()
