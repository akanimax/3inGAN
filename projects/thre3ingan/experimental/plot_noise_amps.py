import argparse
import sys
import matplotlib.pyplot as plt
from pathlib import Path

from projects.thre3ingan.singans import (
    create_thre3d_singan_with_direct_supervision_from_saved_model,
)
from thre3d_atom.utils.logging import log


def parse_arguments(args: str) -> argparse.Namespace:
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


def plot_noise_controllers(args: argparse.Namespace) -> None:
    tsds, _ = create_thre3d_singan_with_direct_supervision_from_saved_model(
        args.model_path
    )

    # make sure that the output directory exists:
    log.info(f"creating the output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # noinspection PyProtectedMember
    all_noise_amp_values = tsds._generator.noise_amps

    _ = plt.figure()
    plt.title("noise-controller")
    plt.xlabel("training stage")
    plt.xlabel("noise_scale value")
    plt.plot(range(2, len(all_noise_amp_values) + 2), all_noise_amp_values)
    plt.savefig(f"{args.output_dir}/noise_controller_values.png", dpi=600)


def main() -> None:
    plot_noise_controllers(parse_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
