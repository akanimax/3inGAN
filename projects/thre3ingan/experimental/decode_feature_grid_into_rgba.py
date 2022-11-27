import argparse
import sys
from functools import partial
from pathlib import Path
from typing import List, Dict, Any

import torch

from thre3d_atom.modules.volumetric_model.volumetric_model import (
    create_vol_mod_from_saved_model,
    VolumetricModel,
)
from thre3d_atom.rendering.volumetric.voxels import HybridRGBAFeatureGrid
from thre3d_atom.utils.constants import (
    NUM_COLOUR_CHANNELS,
    NUM_COORD_DIMENSIONS,
    NUM_RGBA_CHANNELS,
)
from thre3d_atom.utils.misc import batchify


def parse_arguments(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Converts Feature-Grid (+ mlp model) into an RGBA grid",
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


# noinspection PyProtectedMember
def decode_hybrid_feature_grid_to_rgba(
    args: argparse.Namespace,
    vol_mod: VolumetricModel,
    extra_info: Dict[str, Any],
    device: torch.device,
) -> None:
    features = vol_mod.feature_grid.features
    rgba_part = features[..., :NUM_RGBA_CHANNELS]
    rgba_features = torch.clip(rgba_part, 0.0, 1.0)
    new_vol_mod = VolumetricModel(
        vol_mod._render_params,
        grid_dims=vol_mod._grid_dims,
        feature_dims=NUM_RGBA_CHANNELS,
        grid_size=vol_mod._grid_size,
        grid_center=vol_mod._grid_center,
        render_mlp=None,
        fine_render_mlp=None,
        background_render_mlp=None,
        device=device,
    )
    new_vol_mod.feature_grid.features = rgba_features

    torch.save(new_vol_mod.get_save_info(extra_info), f"{args.output_dir}")


# noinspection PyProtectedMember
def decode_feature_grid_to_rgba(
    args: argparse.Namespace,
    vol_mod: VolumetricModel,
    extra_info: Dict[str, Any],
    device: torch.device,
) -> None:
    features = vol_mod.feature_grid.features
    x_dim, y_dim, z_dim, nc = features.shape

    flat_features = features.reshape(-1, nc)

    # decode the flat_features:
    # we use a fixed viewing direction for decoding
    view_dirs = (
        (-torch.ones((1, NUM_COORD_DIMENSIONS), dtype=torch.float32))
        .repeat(len(flat_features), 1)
        .to(device)
    )
    view_dirs = view_dirs / view_dirs.norm(dim=-1, keepdim=True)

    # decode the features and viewing directions using the render_mlp:
    batchified_processor_function = batchify(
        processor_fn=vol_mod.render_mlp,
        collate_fn=partial(torch.cat, dim=0),
        chunk_size=vol_mod._render_params.num_points_chunk,
        verbose=True,
    )
    with torch.no_grad():
        rgba_values = batchified_processor_function(
            torch.cat([flat_features, view_dirs], dim=-1)
        )
    rgb_values, a_values = (
        rgba_values[:, :NUM_COLOUR_CHANNELS],
        rgba_values[:, NUM_COLOUR_CHANNELS:],
    )
    rgb_values = vol_mod._colour_producer(rgb_values).reshape(x_dim, y_dim, z_dim, -1)[
        None, ...
    ]
    a_values = vol_mod._transmittance_behaviour(
        a_values, torch.ones_like(a_values, device=device)
    ).reshape(x_dim, y_dim, z_dim, -1)[None, ...]

    decoded_values = torch.cat([rgb_values, a_values], dim=-1).permute(0, 4, 1, 2, 3)

    new_vol_mod = VolumetricModel(
        vol_mod._render_params,
        grid_dims=vol_mod._grid_dims,
        feature_dims=NUM_RGBA_CHANNELS,
        grid_size=vol_mod._grid_size,
        grid_center=vol_mod._grid_center,
        render_mlp=None,
        fine_render_mlp=None,
        background_render_mlp=None,
        device=device,
    )
    new_vol_mod.feature_grid.features = decoded_values[0].permute(1, 2, 3, 0)

    torch.save(new_vol_mod.get_save_info(extra_info), f"{args.output_dir}")


def main() -> None:
    args = parse_arguments(sys.argv[1:])

    # load the vol_mod
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vol_mod, extra_info = create_vol_mod_from_saved_model(args.model_path, device)

    # invoke the extractor method depending on the type of vol_mod we have:
    if isinstance(vol_mod.feature_grid, HybridRGBAFeatureGrid):
        decode_hybrid_feature_grid_to_rgba(args, vol_mod, extra_info, device)
    else:
        decode_feature_grid_to_rgba(args, vol_mod, extra_info, device)


if __name__ == "__main__":
    main()
