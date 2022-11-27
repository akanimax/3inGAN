import warnings
from typing import Any, Tuple

import numpy as np
import torch

_constant_cache = dict()

# Symbolic assert.
try:
    symbolic_assert = torch._assert  # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0


def constant(
    value: Any,
    shape: Tuple[int, ...] = None,
    dtype: Any = None,
    device: torch.device = None,
    memory_format: torch.memory_format = None,
):
    """
    Cached construction of constant tensors. Avoids CPU=>GPU copy when the
    same constant is used multiple times."""
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (
        value.shape,
        value.dtype,
        value.tobytes(),
        shape,
        dtype,
        device,
        memory_format,
    )
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


# ----------------------------------------------------------------------------
# Context manager to suppress known warnings in torch.jit.trace().


class suppress_tracer_warnings(warnings.catch_warnings):
    def __enter__(self):
        super().__enter__()
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        return self


# ----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().


def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(
            f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}"
        )
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(torch.as_tensor(size), ref_size),
                    f"Wrong size for dimension {idx}",
                )
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(size, torch.as_tensor(ref_size)),
                    f"Wrong size for dimension {idx}: expected {ref_size}",
                )
        elif size != ref_size:
            raise AssertionError(
                f"Wrong size for dimension {idx}: got {size}, expected {ref_size}"
            )
