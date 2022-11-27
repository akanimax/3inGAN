from typing import Sequence, Tuple, Dict, Any, List

from torch import Tensor
from torch.nn import Conv2d

from thre3d_atom.networks.conv_nets import ConvolutionalEncoder
from thre3d_atom.networks.network_interface import Network
from thre3d_atom.networks.shared.equalized_layers import EqualizedConv2d


class Discriminator(Network):
    def __init__(self, encoder: Network, use_eql: bool = False) -> None:
        super().__init__()
        self._encoder = encoder
        self._use_eql = use_eql
        self._critic = (
            EqualizedConv2d(self._encoder.output_shape[1], 1, kernel_size=1, stride=1)
            if self._use_eql
            else Conv2d(self._encoder.output_shape[1], 1, kernel_size=1, stride=1)
        )

    @property
    def output_shape(self) -> Sequence[Tuple[int, ...]]:
        return None, 1, None, None  # num_channels = 1 for real/fake score

    @property
    def input_shape(self) -> Sequence[Tuple[int, ...]]:
        return self._encoder.input_shape

    def get_save_info(self) -> Dict[str, Any]:
        return {
            "encoder": self._encoder.get_save_info(),
            "conf": {
                "use_eql": self._use_eql,
            },
            "state_dict": self.state_dict(),
        }

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        embeddings, features = self._encoder(
            x, normalize_embeddings=False, intermediate_embeddings=True
        )
        predicted_scores = self._critic(embeddings)
        return predicted_scores, features + [embeddings]


def get_convolutional_discriminator(
    depth: int = 3,
    latent_size: int = 512,
    fmap_max: int = 128,
    fmap_base: int = 2048,
    fmap_min: int = 32,
    use_eql: bool = False,
    use_minibatch_stddev: bool = False,
) -> Discriminator:
    return Discriminator(
        ConvolutionalEncoder(
            depth=depth,
            latent_size=latent_size,
            use_minibatch_stddev=use_minibatch_stddev,
            fmap_max=fmap_max,
            fmap_base=fmap_base,
            fmap_min=fmap_min,
            use_eql=use_eql,
        ),
        use_eql=use_eql,
    )
