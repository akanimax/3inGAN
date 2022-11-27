import argparse
import sys
from pathlib import Path
from typing import List

import imageio
import torch
from sklearn.decomposition import PCA
from torch import Tensor

from thre3d_atom.modules.volumetric_model.volumetric_model import (
    create_vol_mod_from_saved_model,
    process_hybrid_rgba_volumetric_model,
)
from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS
from thre3d_atom.utils.imaging_utils import to8b, adjust_dynamic_range
from thre3d_atom.utils.logging import log


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Thre3d SinGAN model trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("-i", "-m", "--model_path",
                        action="store", type=Path, required=True, help="path to the trained 3dSGDS model")
    parser.add_argument("-o", "--output_dir",
                        action="store", type=Path, required=True, help="path to the output directory")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def save_flattened_rgba_sheets(
    features: Tensor,
    output_path: Path,
    cutting_axis: int,
) -> None:
    assert (
        features.shape[-1] == NUM_COLOUR_CHANNELS
    ), f"Sorry, cannot flatten non-reduced rgb volumes into sheets :("

    features = torch.flip(features, dims=(2,))

    # the flatten_permutation is made based on the requested cutting_axis:
    if cutting_axis == 0:
        flatten_permutation = (0, 3, 2, 1)
    elif cutting_axis == 1:
        flatten_permutation = (1, 3, 2, 0)
    elif cutting_axis == 2:
        flatten_permutation = (2, 3, 0, 1)
    else:
        raise ValueError(
            f"Cannot cut across {cutting_axis} axis :(."
            f"Acceptable values are 0, 1 and 2 for a 3D volume"
        )

    flat_rgba_slices = features.permute(*flatten_permutation)
    mean, var = flat_rgba_slices.mean().item(), flat_rgba_slices.std().item()
    flat_rgba_slices = adjust_dynamic_range(
        flat_rgba_slices,
        drange_in=(mean - var, mean + var),
        drange_out=(0, 1),
        slack=True,
    )

    imageio.mimwrite(
        output_path,
        to8b(flat_rgba_slices.permute(0, 2, 3, 1).detach().cpu().numpy()),
    )


def visualize_feature_grid_pca(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vol_mod, _ = create_vol_mod_from_saved_model(args.model_path)

    # noinspection PyProtectedMember
    if vol_mod._hybrid_rgba_mode:
        vol_mod = process_hybrid_rgba_volumetric_model(vol_mod)

    features = vol_mod.feature_grid.features
    x_dim, y_dim, z_dim, feats_dim = features.shape

    flat_features = features.reshape(-1, feats_dim)
    pca_model = PCA(n_components=NUM_COLOUR_CHANNELS)

    # fit the pca_model on the flat features:
    log.info("Applying PCA to make the features visualizable ...")
    pca_model.fit(flat_features.cpu().detach().numpy())

    # obtain the dimensionality reduced features:
    dim_reduced_feats = pca_model.transform(
        flat_features.detach().cpu().numpy()
    ).reshape(x_dim, y_dim, z_dim, NUM_COLOUR_CHANNELS)
    dim_reduced_feats = torch.from_numpy(dim_reduced_feats).to(device)

    for cutting_axis in range(3):
        save_flattened_rgba_sheets(
            features=dim_reduced_feats,
            output_path=args.output_dir / f"feats_coloured_{cutting_axis}.gif",
            cutting_axis=cutting_axis,
        )


def main() -> None:
    visualize_feature_grid_pca(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
