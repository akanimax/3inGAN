import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.backends import cudnn
from torchvision.transforms import RandomCrop
from tqdm import tqdm

from projects.thre3ingan.singans.image_model import load_trained_image_model
from thre3d_atom.utils.config_utils import str2bool
from thre3d_atom.utils.imaging_utils import to8b
from thre3d_atom.utils.logging import log

cudnn.benchmark = True


def parse_arguments(args: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Swap random crops of the feature-grid of a trained ImageModel and render them",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # fmt: off
    # Required arguments
    parser.add_argument("-i", "-m", "--model_path", action="store", type=Path, required=False,
                        help="path to the optimized ImageModel")
    parser.add_argument("-o", "--output_dir", action="store", type=Path, required=False,
                        help="path to the output asset directory")

    # optional arguments
    parser.add_argument("--feature_crop_size", action="store", type=int, required=False, default=10,
                        help="size of crops to be swapped")
    parser.add_argument("--num_samples", action="store", type=int, required=False, default=50,
                        help="number of these crops-swapped examples to generate")
    parser.add_argument("--draw_crop_rectangles", action="store", type=str2bool, required=False, default=True,
                        help="whether to draw the red and green rectangles around the swapped crops")
    # fmt: on

    parsed_args = parser.parse_args(args)
    return parsed_args


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments(sys.argv[1:])

    log.info("Creating the feature-crop swapped ImageModels")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample_num in tqdm(range(1, args.num_samples + 1)):
        # load the image_model:
        image_model = load_trained_image_model(
            model_path=args.model_path, device=device, verbose_creation=False
        )

        # obtain random crop locations on the feature grid:
        r1, c1, r_w1, c_w1 = RandomCrop.get_params(
            image_model.feature_grid.features,
            output_size=(args.feature_crop_size, args.feature_crop_size),
        )
        r2, c2, r_w2, c_w2 = RandomCrop.get_params(
            image_model.feature_grid.features,
            output_size=(args.feature_crop_size, args.feature_crop_size),
        )

        # create the two rectangles for Plotting based on the two crop_locations
        coord_scale = (
            image_model.image_resolution[0]
            / image_model.feature_grid.features.shape[-2]
        )

        # note the convention for the rows and columns in image space below:
        rect1 = (
            c1 * coord_scale,
            r1 * coord_scale,
            (c1 * coord_scale) + (c_w1 * coord_scale),
            (r1 * coord_scale) + (r_w1 * coord_scale),
        )
        rect2 = (
            c2 * coord_scale,
            r2 * coord_scale,
            (c2 * coord_scale) + (c_w2 * coord_scale),
            (r2 * coord_scale) + (r_w2 * coord_scale),
        )

        # swap the features in the two crop_locations:
        with torch.no_grad():
            crop_buffer = image_model.feature_grid.features[
                ..., r1 : r1 + r_w1, c1 : c1 + c_w1
            ].clone()
            image_model.feature_grid.features[
                ..., r1 : r1 + r_w1, c1 : c1 + c_w1
            ] = image_model.feature_grid.features[..., r2 : r2 + r_w2, c2 : c2 + c_w2]
            image_model.feature_grid.features[
                ..., r2 : r2 + r_w2, c2 : c2 + c_w2
            ] = crop_buffer

        # render the crop-swapped feature grid:
        rendered_image = image_model.render()

        # draw the rectangles on the rendered image:
        image = Image.fromarray(to8b(rendered_image.numpy()))
        if args.draw_crop_rectangles:
            drawer = ImageDraw.Draw(image)
            drawer.rectangle(rect1, outline=(255, 0, 0))
            drawer.rectangle(rect2, outline=(0, 255, 0))

        # save the final image
        image.save(
            args.output_dir / f"feature-crop_swapped_image_model_{sample_num}.png"
        )
    log.info(f"Samples generated. Please check: {args.output_dir}")


if __name__ == "__main__":
    main()
