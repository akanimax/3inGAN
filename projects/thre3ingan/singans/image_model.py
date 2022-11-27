from functools import partial
from pathlib import Path
from typing import Dict, Any, Callable, Tuple, Optional, Sequence

import PIL
import imageio
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import Module, Tanh, Parameter
from torch.nn.functional import grid_sample, l1_loss, mse_loss
from torch.utils.tensorboard import SummaryWriter

from thre3d_atom.networks.dense_nets import SkipMLP, SkipMLPConfig
from thre3d_atom.networks.network_interface import Network
from thre3d_atom.networks.shared.layers import (
    PositionalEncodingsEmbedder,
    PixelwiseNorm,
)
from thre3d_atom.utils.constants import NUM_COLOUR_CHANNELS
from thre3d_atom.utils.imaging_utils import (
    adjust_dynamic_range,
    to8b,
    mse2psnr,
    get_2d_coordinates,
)
from thre3d_atom.utils.logging import log


class FeatureGrid2D(Module):
    def __init__(
        self,
        height: int,
        width: int,
        feature_dims: int,
        tunable: bool = True,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        super().__init__()

        # state of the object:
        self._height = height
        self._width = width
        self._feature_dims = feature_dims
        self._tunable = tunable

        self.features = torch.empty(
            (1, feature_dims, height, width), device=device, requires_grad=True
        )
        torch.nn.init.xavier_uniform_(self.features)

        if self._tunable:
            self.features = Parameter(self.features)

    @classmethod
    def from_feature_tensor(cls, feature_tensor: Tensor) -> Any:
        _, feature_dims, height, width = feature_tensor.shape
        # initialize a random feature_grid
        feature_grid = cls(
            height=height,
            width=width,
            feature_dims=feature_dims,
            tunable=False,
            device=feature_tensor.device,
        )

        # use the given feature_tensor as it's features:
        feature_grid.features = feature_tensor

        return feature_grid

    def extra_repr(self) -> str:
        return (
            f"grid_dims: {self.features.shape[2:]}, "
            f"feature_dims: {self.features.shape[1]}, "
            f"tunable: {self._tunable}"
        )

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "height": self._height,
                "width": self._width,
                "feature_dims": self._feature_dims,
                "tunable": self._tunable,
            },
            "state_dict": self.state_dict(),
        }

    def forward(self, coords: Tensor) -> Tensor:
        """coords should be of shape => [N x 2], and be in the range [-1, 1]"""
        sam_vals = grid_sample(
            # note the convention difference between the image and sample coordinates
            self.features.permute(0, 1, 3, 2),
            coords[None, None, ...],
            mode="bilinear",
            align_corners=False,
        )
        return sam_vals.permute(0, 2, 3, 1)[0, 0, ...]


class ImageDecoderMLP(Network):
    # noinspection PyUnresolvedReferences
    def __init__(
        self,
        mlp: SkipMLP,
        feature_dims: int = 32,
        feature_embedding_dims: int = 0,
        use_local_coords: bool = False,
        local_coords_embedding_dims: int = 0,
        normalize_features: bool = False,
    ) -> None:
        super().__init__()
        self._mlp = mlp
        self._feature_dims = feature_dims
        self._feature_embedding_dims = feature_embedding_dims
        self._use_local_coords = use_local_coords
        self._local_coords_embedding_dims = local_coords_embedding_dims
        self._normalize_features = normalize_features

        # objects of modification:
        self._normalizer = PixelwiseNorm()
        self._feature_embedder = PositionalEncodingsEmbedder(
            input_dims=self._feature_dims, emb_dims=self._feature_embedding_dims
        )
        self._local_coords_embedder = PositionalEncodingsEmbedder(
            input_dims=2, emb_dims=self._local_coords_embedding_dims
        )

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return self._mlp.input_shape

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        return self._mlp.output_shape

    @property
    def feature_dims(self) -> int:
        return self._feature_dims

    @property
    def use_local_coords(self) -> bool:
        return self._use_local_coords

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "feature_dims": self._feature_dims,
                "feature_embedding_dims": self._feature_embedding_dims,
                "use_local_coords": self._use_local_coords,
                "local_coords_embedding_dims": self._local_coords_embedding_dims,
                "normalize_features": self._normalize_features,
            },
            "mlp": self._mlp.get_save_info(),
            "state_dict": self.state_dict(),
        }

    def load_weights(self, weights: Dict[str, Any]) -> None:
        self._mlp.load_state_dict(weights["mlp"]["state_dict"])

    def forward(self, x: Tensor) -> Tensor:
        if self._use_local_coords:
            features, local_coords = (
                x[..., : self._feature_dims],
                x[..., self._feature_dims :],
            )
        else:
            features, local_coords = x, torch.zeros(size=(x.shape[0], 0))

        embedded_features = self._feature_embedder(features)
        embedded_local_coords = self._local_coords_embedder(local_coords)
        normalized_features = self._normalizer(features)

        if self._use_local_coords:
            feats = (
                normalized_features if self._normalize_features else embedded_features
            )
            mlp_input = torch.cat([feats, embedded_local_coords], dim=-1)
        else:
            mlp_input = (
                normalized_features if self._normalize_features else embedded_features
            )
        return self._mlp(mlp_input)


def get_default_image_decoder_mlp(
    feature_dims: int = 32,
    feature_embedding_dims: int = 0,
    use_local_coords: bool = False,
    local_coords_embedding_dims: int = 0,
    normalize_features: bool = False,
) -> ImageDecoderMLP:
    feat_inp_dims = feature_dims + (2 * feature_dims * feature_embedding_dims)
    lc_inp_dims = 2 + (2 * 2 * local_coords_embedding_dims)
    if use_local_coords:
        mlp_input_dims = feat_inp_dims + lc_inp_dims
    elif normalize_features:
        mlp_input_dims = feature_dims
    else:
        mlp_input_dims = feat_inp_dims

    mlp_config = SkipMLPConfig(
        input_dims=mlp_input_dims,
        layer_depths=[256],
        output_dims=NUM_COLOUR_CHANNELS,
        skips=[False],
        use_equalized_learning_rate=True,
        out_activation_fn=Tanh(),
    )
    return ImageDecoderMLP(
        SkipMLP(mlp_config),
        feature_dims=feature_dims,
        feature_embedding_dims=feature_embedding_dims,
        use_local_coords=use_local_coords,
        local_coords_embedding_dims=local_coords_embedding_dims,
        normalize_features=normalize_features,
    )


def decode_coords_with_fg_and_mlp(
    coords: Tensor,
    feature_grid: FeatureGrid2D,
    decoder_mlp: ImageDecoderMLP,
    image_resolution: Tuple[int, int],
) -> Tensor:
    """decodes the coords tensor into RGB pixel values"""
    orig_shape = coords.shape
    coords = coords.reshape(-1, orig_shape[-1])
    image_height, image_width = image_resolution

    local_coords = adjust_dynamic_range(coords, drange_in=(-1, 1), drange_out=(0, 1))
    local_coords[..., 0] *= image_height
    local_coords[..., 1] *= image_width
    local_coords = local_coords - torch.floor(local_coords)

    decoded_features = feature_grid(coords)
    decoder_input = (
        torch.cat([decoded_features, local_coords], dim=-1)
        if decoder_mlp.use_local_coords
        else decoded_features
    )
    return decoder_mlp(decoder_input).reshape(*orig_shape[:-1], -1)


class ImageModel:
    def __init__(
        self,
        image_height: int,
        image_width: int,
        feature_dims: int = 32,
        decoder_mlp_maker: Callable[[], Network] = get_default_image_decoder_mlp,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        verbose_creation: bool = True,
    ) -> None:
        self._image_height = image_height
        self._image_width = image_width
        self._feature_dims = feature_dims
        self._device = device

        # compute the height and width of the feature-grid so as to keep the number of
        # parameters in the image and the parametric-model same:
        self._setup_feature_dims()

        # create a feature grid object (note that these are kept public):
        self.feature_grid = FeatureGrid2D(
            self._feature_height, self._feature_width, feature_dims, device=device
        )
        self.decoder_mlp = decoder_mlp_maker().to(self._device)

        # print info related to the Feature Grid and the Decoder MLP:
        if verbose_creation:
            log.info(f"Created Feature grid: {self.feature_grid}")
            log.info(f"Created Decoder MLP: {self.decoder_mlp}")

    @property
    def image_resolution(self) -> Tuple[int, int]:
        return self._image_height, self._image_width

    @staticmethod
    def compute_feature_grid_dims(
        image_resolution: Tuple[int, int], feature_dims: int
    ) -> Tuple[int, int]:
        image_height, image_width = image_resolution
        aspect_ratio = image_width / image_height
        total_image_params = image_width * image_height * NUM_COLOUR_CHANNELS
        needed_params = total_image_params / feature_dims
        feature_grid_height = int(np.ceil(np.sqrt(needed_params / aspect_ratio)))
        feature_grid_width = int(aspect_ratio * feature_grid_height)
        return feature_grid_height, feature_grid_width

    def _setup_feature_dims(self) -> None:
        self._feature_height, self._feature_width = self.compute_feature_grid_dims(
            image_resolution=(self._image_height, self._image_width),
            feature_dims=self._feature_dims,
        )

    @staticmethod
    def _shuffle_tensor_2d(tensor_2d: Tensor) -> Tensor:
        """ shuffles a 2D Tensor of shape [N x C]"""
        return tensor_2d[torch.randperm(len(tensor_2d))]

    def _infinite_data_loader(self, data: Tensor, batch_size: int) -> Tensor:
        while True:
            data = self._shuffle_tensor_2d(data)
            for batch_index in range(0, len(data), batch_size):
                data_batch = data[batch_index : batch_index + batch_size]
                if data_batch.shape[0] == batch_size:
                    yield data_batch
                else:
                    break

    @staticmethod
    def _check_log_condition(
        current_step: int, frequency_step: int, start_step: int, end_step: int
    ) -> bool:
        return (
            current_step % frequency_step == 0
            or current_step == start_step
            or current_step == end_step
        )

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "conf": {
                "image_height": self._image_height,
                "image_width": self._image_width,
                "feature_dims": self._feature_dims,
            },
            "feature_grid": self.feature_grid.get_save_info(),
            "decoder_mlp": self.decoder_mlp.get_save_info(),
        }

    def render(
        self,
        render_resolution: Optional[Tuple[int, int]] = None,
        chunk_size: int = 64 * 1024,
    ) -> Tensor:
        height, width = (
            (self._image_height, self._image_width)
            if render_resolution is None
            else render_resolution
        )

        # create a coordinates mesh-grid:
        coords = get_2d_coordinates(height, width)

        # flatten the coordinates and bring them on the GPU:
        flat_coords = coords.reshape(-1, coords.shape[-1]).to(self._device)

        # decode all the coordinates into pixel values chunk by chunk:
        decoded_image = []
        with torch.no_grad():
            for chunk_index in range(0, len(flat_coords), chunk_size):
                coord_chunk = flat_coords[chunk_index : chunk_index + chunk_size]
                decoded_image.append(
                    decode_coords_with_fg_and_mlp(
                        coord_chunk,
                        self.feature_grid,
                        self.decoder_mlp,
                        self.image_resolution,
                    )
                )
        decoded_image = torch.cat(decoded_image, dim=0)
        decoded_image = decoded_image.reshape(height, width, -1)
        decoded_image = adjust_dynamic_range(
            decoded_image.cpu(),
            drange_in=(-1, 1),
            drange_out=(0, 1),
            slack=True,
        )
        return decoded_image

    def train(
        self,
        training_image: PIL.Image.Image,
        num_iterations: int = 10000,
        batch_size: int = 8192,
        learning_rate: float = 0.003,
        lr_decay_steps: int = 5000,
        feedback_frequency: int = 1000,
        loss_feedback_frequency: int = 10,
        testing_frequency: int = 1000,
        save_frequency: int = 2000,
        output_dir: Path = Path(__file__).parent.absolute() / "logs",
    ) -> None:
        # load the training image and create a dataset of pixel_coordinates -> pixel RGB values:
        image_np = np.array(training_image).astype(np.float32) / 255
        if len(image_np.shape) < 3:
            image_np = np.tile(image_np[..., None], (1, 1, 3))
        image_np = image_np[..., :3]  # in case of > 3 channel images
        real_feedback_image = image_np
        # bring the pixel range to (-1, 1) for training
        image_np = adjust_dynamic_range(image_np, drange_in=(0, 1), drange_out=(-1, 1))

        # make sure the training image is compatible with the ImageModel
        assert (
            self._image_height == image_np.shape[0]
            and self._image_width == image_np.shape[1]
        ), (
            f"The provided training image with size ({image_np.shape[:-1]}) is incompatible with the Image-Model's"
            f"image size ({self._image_height, self._image_width})"
        )

        image_coords = get_2d_coordinates(self._image_height, self._image_width)
        coord_rgb_image = torch.cat(
            [
                image_coords.to(self._device),
                torch.from_numpy(image_np).to(self._device),
            ],
            dim=-1,
        )
        training_data = coord_rgb_image.reshape(-1, coord_rgb_image.shape[-1])
        training_data_loader = iter(
            self._infinite_data_loader(training_data, batch_size=batch_size)
        )

        # setup optimizer:
        optimizer = torch.optim.Adam(
            params=[
                {"params": self.feature_grid.parameters(), "lr": learning_rate},
                {"params": self.decoder_mlp.parameters(), "lr": learning_rate},
            ],
            betas=(0, 0.99),
        )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

        # setup output directories
        # fmt: off
        model_dir = output_dir / "saved_models"
        logs_dir = output_dir / "training_logs"
        tensorboard_dir = logs_dir / "tensorboard"
        render_dir = logs_dir / "rendered_output"
        for directory in (model_dir, logs_dir, tensorboard_dir,
                          render_dir):
            directory.mkdir(exist_ok=True, parents=True)
        # fmt: on

        # create the tensorboard directory:
        tensorboard_writer = SummaryWriter(tensorboard_dir)

        # log the real image for feedback:
        log.info(f"Logging real feedback image")
        imageio.imwrite(
            render_dir / f"1__real_log.png",
            to8b(real_feedback_image),
        )

        log.info(f"!! Beginning Training !!")
        for num_iter in range(1, num_iterations + 1):
            # load the next batch of data:
            data_batch = next(training_data_loader)
            coords, gt_rgb = (
                data_batch[..., :-NUM_COLOUR_CHANNELS],
                data_batch[..., -NUM_COLOUR_CHANNELS:],
            )

            # forward pass and compute the loss
            pred_rgb = decode_coords_with_fg_and_mlp(
                coords,
                self.feature_grid,
                self.decoder_mlp,
                self.image_resolution,
            )
            loss = l1_loss(pred_rgb, gt_rgb)

            # perform single step of optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # verbose logging per iteration:
            loss_value = loss.item()
            psnr_value = mse2psnr(mse_loss(pred_rgb, gt_rgb).item())

            # tensorboard summaries feedback (logged every iteration)
            for summary_name, summary_value in (
                ("loss", loss_value),
                ("psnr", psnr_value),
            ):
                if summary_value is not None:
                    tensorboard_writer.add_scalar(
                        summary_name, summary_value, global_step=num_iter
                    )

            # console loss feedback log
            if self._check_log_condition(
                num_iter, loss_feedback_frequency, 1, num_iterations
            ):
                loss_info_string = (
                    f"Global Iteration: {num_iter} "
                    f"Loss: {loss_value: .5f} "
                    f"PSNR: {psnr_value: .5f} "
                )
                log.info(loss_info_string)

            # step the learning rate schedulers
            if num_iter % lr_decay_steps == 0:
                lr_scheduler.step()
                new_lrs = [param_group["lr"] for param_group in optimizer.param_groups]
                log_string = f"Adjusted learning rate | learning rate: {new_lrs} "
                log.info(log_string)

            # save the rendered feedback
            if self._check_log_condition(
                num_iter, feedback_frequency, 1, num_iterations
            ):
                imageio.imwrite(
                    render_dir / f"render_log_{num_iter}.png",
                    to8b(self.render().numpy()),
                )

            # obtain and log test metrics
            if self._check_log_condition(
                num_iter, testing_frequency, 1, num_iterations
            ):
                log.info(f"Computing test score ...")
                test_psnr = mse2psnr(
                    mse_loss(
                        self.render(),
                        torch.from_numpy(real_feedback_image),
                    ).item()
                )
                log.info(f"Full image PSNR: {test_psnr: .5f}")
                tensorboard_writer.add_scalar(
                    "full_image_psnr", test_psnr, global_step=num_iter
                )

            # save the model
            if self._check_log_condition(num_iter, save_frequency, 1, num_iterations):
                torch.save(
                    self.get_save_info(),
                    model_dir / f"model_iter_{num_iter}.pth",
                )

        # save the final model
        torch.save(self.get_save_info(), model_dir / f"model_final.pth")
        log.info("!! Training complete !!")


def load_trained_image_model(
    model_path: Path, device: torch.device, verbose_creation: bool = True
) -> ImageModel:
    loaded_model = torch.load(model_path)
    if verbose_creation:
        log.info(f"loaded trained model from: {model_path}")
    img_mod = ImageModel(
        **loaded_model["conf"],
        device=device,
        verbose_creation=verbose_creation,
        decoder_mlp_maker=partial(
            get_default_image_decoder_mlp,
            **loaded_model["decoder_mlp"]["conf"],
        ),
    )
    img_mod.feature_grid.load_state_dict(loaded_model["feature_grid"]["state_dict"])
    img_mod.decoder_mlp.load_weights(loaded_model["decoder_mlp"])
    return img_mod
