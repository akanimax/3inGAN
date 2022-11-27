import argparse
from pathlib import Path

import imageio
import torch

from projects.thre3ingan.singans.image_model import load_trained_image_model
from thre3d_atom.utils.imaging_utils import to8b
from thre3d_atom.utils.logging import log


def parse_arguments() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("Image model rendering endpoint-script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument("-i", "-m", "--model_path", action="store", type=Path, required=True,
                        help="path to the trained image model")
    parser.add_argument("-o", "--output_dir", action="store", type=Path, required=True,
                        help="path to the output asset directory")

    # optional arguments
    parser.add_argument("--render_resolution", action="store", type=int, nargs=2, required=False, default=None,
                        help="Resolution at which the image is to be rendered from the Feature-Grid")
    # fmt: on

    parsed_args = parser.parse_args()
    return parsed_args


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the image model from the given model path:
    img_mod = load_trained_image_model(args.model_path, device=device)
    rendered_image_path = args.output_dir / f"rendered_output.png"
    imageio.imwrite(
        rendered_image_path,
        to8b(img_mod.render(render_resolution=args.render_resolution).numpy()),
    )
    log.info(f"Saved the rendered image at: {rendered_image_path}")


if __name__ == "__main__":
    main(parse_arguments())
