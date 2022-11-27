import argparse
import sys
from pathlib import Path

import imageio
import numpy as np
import torch
from projects.thre3ingan.singans import (
    create_thre3d_singan_with_direct_supervision_from_saved_model,
)
from thre3d_atom.modules.volumetric_model.volumetric_model import (
    create_vol_mod_from_saved_model,
)
from thre3d_atom.rendering.volumetric.voxels import FeatureGrid
from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS
from thre3d_atom.utils.imaging_utils import to8b, adjust_dynamic_range
from thre3d_atom.utils.logging import log
from torchvision.utils import make_grid
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_arguments(args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Thre3d SinGAN model trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("-i", "-m", "--model_path",
                        action="store", type=Path, required=True, help="path to the trained 3dSGDS model")
    parser.add_argument("-mode",
                        action="store", type=str, required=True,
                        help="which mode to run the script in; options: [3dsingan | vol-mod]")
    parser.add_argument("-o", "--output_dir",
                        action="store", type=Path, required=True, help="path to the output directory")

    # non required arguments:
    parser.add_argument("--num_gan_samples", action="store",
                        type=int, required=False, default=30, help="number of gan samples needed for analysis")
    parser.add_argument("--render_stage", action="store",
                        type=int, required=False, default=None, help="which stage to render the sheet samples at")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def save_flattened_density_and_rgba_sheets(
    feature_grid: FeatureGrid,
    output_path_rgb: Path,
    output_path_density: Path,
) -> None:
    features = torch.flip(feature_grid.features, dims=(2,))
    rgba_features, density_features = (
        features[..., :-1].detach().cpu(),
        features[..., -1:].detach().cpu(),
    )
    rgba_features = adjust_dynamic_range(
        rgba_features,
        drange_in=(rgba_features.min(), rgba_features.max()),
        drange_out=(0, 1),
    )
    density_features = adjust_dynamic_range(
        density_features,
        drange_in=(density_features.min(), density_features.max()),
        drange_out=(0, 1),
    )

    assert (
        rgba_features.shape[-1] == NUM_COLOUR_CHANNELS
    ), f"Sorry, cannot flatten non-rgba volumes into sheets :("

    # write the reconstruction RGBA sheet
    flat_rgba_slices = rgba_features.permute(2, 3, 0, 1)
    flat_rgba_sheet = make_grid(
        flat_rgba_slices,
        nrow=int(np.ceil(np.sqrt(len(flat_rgba_slices)))),
    )
    imageio.mimwrite(
        str(output_path_rgb).replace(".png", ".gif"),
        to8b(flat_rgba_slices.permute(0, 2, 3, 1).detach().cpu().numpy()),
    )
    imageio.imwrite(
        output_path_rgb,
        to8b(flat_rgba_sheet.permute(1, 2, 0).detach().cpu().numpy()),
    )

    # write the reconstruction density_sheet
    density_slices = density_features.permute(2, 3, 0, 1)
    flat_density_sheet = make_grid(
        density_slices,
        nrow=int(np.ceil(np.sqrt(len(density_slices)))),
    )
    imageio.mimwrite(
        str(output_path_density).replace(".png", ".gif"),
        to8b(density_slices.permute(0, 2, 3, 1).detach().cpu().numpy()),
    )
    imageio.imwrite(
        output_path_density,
        to8b(flat_density_sheet.permute(1, 2, 0).detach().cpu().numpy()),
    )


def flatten_rgba_sheets(args: argparse.Namespace) -> None:
    if args.mode.lower() == "3dsingan":
        (
            tsds,
            extra_info,
        ) = create_thre3d_singan_with_direct_supervision_from_saved_model(
            args.model_path
        )

        # we use a dummy hemispherical radius value:
        dummy_hem_rad = 12.0

        # make sure that the output directory exists:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # save the reconstruction sheets:
        log.info("Saving the Flattened sheets for the reconstruction case")
        feature_grid = tsds.get_feature_grid(
            hem_rad=dummy_hem_rad,
            scene_bounds=tsds.render_params.scene_bounds,
            random_noise=tsds.reconstruction_noise,
            stage=args.render_stage,
        )
        save_flattened_density_and_rgba_sheets(
            feature_grid=feature_grid,
            output_path_rgb=args.output_dir / "recon_colour.png",
            output_path_density=args.output_dir / "recon_density.png",
        )

        log.info(f"Saving flattened gan samples ...")
        for sample_num in tqdm(range(1, args.num_gan_samples + 1)):
            feature_grid = tsds.get_feature_grid(
                hem_rad=dummy_hem_rad,
                scene_bounds=tsds.render_params.scene_bounds,
                stage=args.render_stage,
            )
            save_flattened_density_and_rgba_sheets(
                feature_grid=feature_grid,
                output_path_rgb=args.output_dir / f"{sample_num}_colour_sample.png",
                output_path_density=args.output_dir
                / f"{sample_num}_density_sample.png",
            )
    elif args.mode.lower() == "vol-mod":
        vol_mod, _ = create_vol_mod_from_saved_model(args.model_path, device=device)
        # vol_mod = process_rgba_model(vol_mod)
        log.info("Saving the Flattened sheets for the trained RGBA_model")
        save_flattened_density_and_rgba_sheets(
            feature_grid=vol_mod.feature_grid,
            output_path_rgb=args.output_dir / "recon_colour.png",
            output_path_density=args.output_dir / "recon_density.png",
        )
    else:
        raise ValueError(f"Unknown mode requested: {args.mode.lower()}")


def main() -> None:
    flatten_rgba_sheets(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
