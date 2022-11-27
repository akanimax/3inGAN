""" General purpose fid_computation script which computes the FID between sets of images
 simply contained inside two folders.
"""

import argparse
import numpy as np
import sys
from pathlib import Path
from typing import List

import torch
from pytorch_fid.fid_score import compute_statistics_of_path, calculate_frechet_distance
from pytorch_fid.inception import InceptionV3

from thre3d_atom.utils.config_utils import str2bool
from thre3d_atom.utils.logging import log


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Thre3d SinGAN Reconstruction 360 render demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("--path_1", action="store", type=Path, required=True,
                        help="Path to image set / stats 1")
    parser.add_argument("--path_2", action="store", type=Path, required=True,
                        help="Path to image set / stats 2")

    # optional arguments
    parser.add_argument("--cache_p1_stats", action="store", type=str2bool, required=False, default=True,
                        help="whether to write the mus and sigmas of path_1 as cache file")
    parser.add_argument("--cache_p2_stats", action="store", type=str2bool, required=False, default=True,
                        help="whether to write the mus and sigmas of path_2 as cache file")
    parser.add_argument("--batch_size", action="store", type=int, required=False, default=16,
                        help="batch_size of images used for InceptionNet forward pass")
    parser.add_argument("--inception_dims", action="store", type=int, required=False, default=2048,
                        help="number of dimension of the InceptionNet extracted features")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def compute_fid(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path_1, path_2 = args.path_1, args.path_2

    # create the inception model object for fid calculation
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.inception_dims]
    model = InceptionV3([block_idx]).to(device)

    # compute inception gaussian for path_1 and path_2:
    gaussians_list = []
    for path, cache_stats in (
        (path_1, args.cache_p1_stats),
        (path_2, args.cache_p2_stats),
    ):
        log.info(f"Computing the inception statistics for: {path}")

        if (path / "cache.npz").exists():
            path = path / "cache.npz"
            log.info(f"Stats cache found at: {path}")

        mu, sigma = compute_statistics_of_path(
            str(path), model, args.batch_size, args.inception_dims, device
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


def main() -> None:
    compute_fid(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
