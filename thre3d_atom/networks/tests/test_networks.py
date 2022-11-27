import logging as log
from typing import Callable, Tuple

import numpy as np
import pytest
import torch
from torch.nn.functional import mse_loss
from thre3d_atom.networks.dense_nets import SkipMLP, SkipMLPConfig
from thre3d_atom.networks.network_interface import Network

TensorShape = Tuple[int, ...]

batch_size: int = 3
dense_input_size: TensorShape = (batch_size, 64)
dense_output_size: TensorShape = (batch_size, 3)


def skip_mlp_network() -> SkipMLP:
    return SkipMLP(
        SkipMLPConfig(
            input_dims=dense_input_size[-1],
            output_dims=dense_output_size[-1],
            layer_depths=(256,) * 3,
            skips=(False, True, False),
        )
    )


@pytest.mark.parametrize(
    "network, expected_output, input_size",
    [
        (skip_mlp_network, -0.71330, dense_input_size),
    ],
)
def test_forward_values(
    network: Callable[[], Network],
    expected_output: float,
    device: torch.device,
    input_size: TensorShape,
) -> None:
    """
    Tests whether the network gives reproducible (deterministic)
    output for same inputs every time.
    """
    # GIVEN: Some network input image
    dummy_input = torch.randn(input_size).to(device)
    network = network().to(device)

    # WHEN: Running a forward pass with some weights
    output = network(dummy_input)
    if isinstance(output, tuple):
        mean_value = output[0].mean().item()
    else:
        mean_value = output.mean().item()

    # THEN: The result is deterministic
    np.testing.assert_almost_equal(mean_value, expected_output, decimal=5)


@pytest.mark.parametrize("network", [skip_mlp_network])
def test_forward_initialization(
    network: Callable[[], Network], device: torch.device
) -> None:
    """
    Tests whether the model is initialized correctly
    """
    # GIVEN: Some network
    network = network().to(device)

    # WHEN: Computing the standard deviation of the initialized weights
    #  and the mean of initialized biases
    weight_std_list, bias_mean_list = [], []
    for name, param in network.named_parameters():
        if "weight" in name:
            std = param.std().item()
            weight_std_list.append(std)
        log.info(f"param: {name, param.shape} std: {param.std()} mean: {param.mean()}")

    # THEN: Since the initialization mechanism uses kaiming_uniform by default
    # even for a `fan_in=1`, the std can never be >= 1.0
    assert all([std < 1.0 for std in weight_std_list])


@pytest.mark.parametrize(
    "network, input_size, output_size",
    [
        (skip_mlp_network, dense_input_size, dense_output_size),
    ],
)
def test_backward_numerical_stability(
    network: Callable[[], Network],
    input_size: TensorShape,
    output_size: TensorShape,
    device: torch.device,
) -> None:
    """
    Tests the numerical stability of backpropagation on the network. Performs a backward
    pass to obtain gradients and then checks for NaNs and infs.
    """
    # GIVEN: Some network, dummy input and dummy label
    network = network().to(device)
    dummy_input = torch.randn(input_size).to(device)
    dummy_label = torch.rand(output_size).to(device)

    # WHEN: Obtaining the gradients with respect to a dummy loss
    output = network(dummy_input)
    output = output[0] if isinstance(output, tuple) else output

    loss = mse_loss(output, dummy_label)
    loss.backward()

    # THEN: No gradients are NaN or inf
    assert all(
        [
            torch.isnan(param.grad).sum() == 0 and torch.isinf(param.grad).sum() == 0
            for param in network.parameters()
            if hasattr(param, "grad")
        ]
    )
