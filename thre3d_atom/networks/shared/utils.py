import numpy as np
import torch
from torch import Tensor


def num_feature_maps_depth_decaying(
    stage: int,
    feature_maps_base: int,
    feature_maps_decay: float,
    feature_maps_min: int,
    feature_maps_max: int,
) -> int:
    """
    computes the number of fmaps present in each stage
    Args:
        stage: stage level
        feature_maps_base: base number of fmaps
        feature_maps_decay: decay rate for the fmaps in the network
        feature_maps_min: minimum number of fmaps
        feature_maps_max: maximum number of fmaps
    Returns: number of fmaps that should be present there
    """
    return int(
        np.clip(
            int(feature_maps_base / (2.0 ** (stage * feature_maps_decay))),
            feature_maps_min,
            feature_maps_max,
        ).item()
    )


def detach_tensor_from_graph(x: Tensor) -> Tensor:
    """
    gracefully detaches the given tensor `x` from the graph it is a part of
    (mainly needed for handling input to convolution layers with cudnn backend)
    Args:
        x: input tensor
    Returns: detached version of the input tensor
    """
    x_detached = torch.empty_like(x).to(x.device)
    with torch.no_grad():
        x_detached.set_(x)
    return x_detached
