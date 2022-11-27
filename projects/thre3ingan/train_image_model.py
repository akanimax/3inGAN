import argparse
from functools import partial
from pathlib import Path

import torch
from PIL import Image
from torch.backends import cudnn

from projects.thre3ingan.singans.image_model import (
    ImageModel,
    get_default_image_decoder_mlp,
)
from thre3d_atom.utils.config_utils import log_args_config_to_disk, str2bool
from thre3d_atom.utils.logging import log

cudnn.benchmark = True


def parse_arguments() -> argparse.Namespace:
    # fmt: off
    parser = argparse.ArgumentParser("Image model training endpoint-script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Required arguments
    parser.add_argument("-i", "--image_path", action="store", type=Path, required=True,
                        help="path to the input training image")
    parser.add_argument("-o", "--output_dir", action="store", type=Path, required=True,
                        help="path to the output asset directory")

    # feature-grid related arguments
    parser.add_argument("--feature_dims", action="store", type=int, required=False, default=32,
                        help="number of feature dims used by the feature grid")
    parser.add_argument("--feature_embedding_dims", action="store", type=int, required=False, default=0,
                        help="number of positional embeddings applied to decoded features from the feature-grid")
    parser.add_argument("--use_local_coords", action="store", type=str2bool, required=False, default=False,
                        help="whether to use local point coordinates while decoding image from the feature-grid")
    parser.add_argument("--local_coords_embedding_dims", action="store", type=int, required=False, default=0,
                        help="number of positional encoding dimensions applied to the local coordinates")
    parser.add_argument("--normalize_features", action="store", type=str2bool, required=False, default=False,
                        help="whether to normalize decoded features from the feature grid to a hypersphere")

    # training arguments
    parser.add_argument("--batch_size", action="store", type=int, required=False, default=8192,
                        help="batch size used for training the model")
    parser.add_argument("--num_iterations", action="store", type=int, required=False, default=15000,
                        help="number of iterations to train the model for")
    parser.add_argument("--learning_rate", action="store", type=float, required=False, default=0.003,
                        help="learning rate used for training")
    parser.add_argument("--lr_decay_steps", action="store", type=int, required=False, default=5000,
                        help="number of steps after which to decay the learning rate")
    parser.add_argument("--save_frequency", action="store", type=int, required=False, default=2000,
                        help="frequency of taking a snapshot")
    parser.add_argument("--testing_frequency", action="store", type=int, required=False, default=500,
                        help="frequency of performing testing")
    parser.add_argument("--feedback_frequency", action="store", type=int, required=False, default=1000,
                        help="frequency of rendering feedback")
    parser.add_argument("--loss_feedback_frequency", action="store", type=int, required=False, default=10,
                        help="frequency of logging loss values to console")
    # fmt: on

    parsed_args = parser.parse_args()
    return parsed_args


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the image model:
    image = Image.open(args.image_path)
    image_size = image.size
    img_mod = ImageModel(  # note the image height and width below
        image_height=image_size[1],
        image_width=image_size[0],
        feature_dims=32,
        decoder_mlp_maker=partial(
            get_default_image_decoder_mlp,
            feature_dims=args.feature_dims,
            feature_embedding_dims=args.feature_embedding_dims,
            use_local_coords=args.use_local_coords,
            local_coords_embedding_dims=args.local_coords_embedding_dims,
            normalize_features=args.normalize_features,
        ),
        device=device,
    )

    # log the configuration as a yaml:
    log.info("Logging configuration file ...")
    log_args_config_to_disk(args, args.output_dir)

    # train the volume_model
    img_mod.train(
        training_image=image,
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay_steps=args.lr_decay_steps,
        feedback_frequency=args.feedback_frequency,
        loss_feedback_frequency=args.loss_feedback_frequency,
        save_frequency=args.save_frequency,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main(parse_arguments())
