import argparse
import numpy as np
import sys
from pathlib import Path
from typing import List

import torch

from thre3d_atom.custom_metrics.sifid.inception import InceptionV3
from thre3d_atom.custom_metrics.sifid.sifid_score import (
    calculate_sifid_given_paths,
    get_activations,
)
from thre3d_atom.utils.logging import log


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Thre3d SinGAN metrics calculator tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("--metric_samples_path", action="store", type=Path, required=True,
                        help="Path to the trained thre3ingan model")

    # optional arguments
    parser.add_argument("--inception_dims", action="store", type=int, required=False, default=64,
                        help="number of dimensions of the InceptionNet extracted features. "
                             "Please note that this is different from the usual 2048 for computing spatial "
                             "features")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments(sys.argv[1:])

    # locate the real images:
    real_views = sorted(list((args.metric_samples_path / "gt_views").iterdir()))

    # locate the random samples per image
    random_samples_per_view = [
        list(random_samples_path.iterdir())
        for random_samples_path in sorted(
            (args.metric_samples_path / "random_samples").iterdir()
        )
    ]

    # =========================================================================================
    # compute the quality metric between each real_view and it's corresponding generated random
    # samples:
    # =========================================================================================
    sifid_list = []
    for real_view_path, random_samples_paths in zip(
        real_views, random_samples_per_view
    ):
        tiled_real_view_paths = [real_view_path] * len(random_samples_paths)
        sifid_list.append(
            calculate_sifid_given_paths(
                tiled_real_view_paths, random_samples_paths, 1, True, 64
            )
        )

    all_sifids = np.array(sifid_list)
    quality_metric_value = all_sifids.mean()

    log.info(f"Computed quality metric score: {quality_metric_value}")

    # =========================================================================================
    # compute the mean diversity over generated random samples per view
    # =========================================================================================
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)

    diversities_list = []
    for random_samples_paths in random_samples_per_view:
        activations = []
        for random_sample in random_samples_paths:
            activations.append(
                get_activations(
                    [random_sample], model, 1, 2048, True, resize_image=True
                )
            )
        all_activations = np.vstack(activations)  # 5 x 2048
        activation_mean = all_activations.mean(axis=0, keepdims=True)  # 1 x 2048
        variance = all_activations - activation_mean  # 5 x 2048
        cov_mat = variance.T @ variance  # 2048 x 2048
        view_diversity = ((cov_mat.sum() - np.trace(cov_mat)) / 2) / (
            ((2048 * 2048) - 2048) / 2
        )
        diversities_list.append(view_diversity)

    all_diversities = np.array(diversities_list)
    diversity_metric_value = all_diversities.mean()
    log.info(f"Computed diversity metric score: {diversity_metric_value}")


if __name__ == "__main__":
    main()
