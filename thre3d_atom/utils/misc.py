from typing import Callable, Sequence, Any, Optional

from tqdm import tqdm

from thre3d_atom.networks.network_interface import Network


def check_power_of_2(x: int) -> bool:
    return x & (x - 1) == 0


def toggle_grad(model: Network, requires_grad: bool) -> None:
    """Toggle the gradient requirement for a given model"""
    for p in model.parameters():
        p.requires_grad_(requires_grad)


def batchify(
    processor_fn: Callable[[Sequence[Any], Any], Sequence[Any]],
    collate_fn: Callable[[Sequence[Any]], Any],
    chunk_size: Optional[int] = None,
    verbose: bool = False,
) -> Callable[[Sequence[Any]], Any]:
    if chunk_size is None:
        return processor_fn

    def batchified_processor_fn(inputs: Sequence[Any], *args, **kwargs) -> Any:
        processed_inputs_batches = []
        progress_bar = tqdm if verbose else lambda _: _
        for chunk_index in progress_bar(range(0, len(inputs), chunk_size)):
            processed_inputs_batches.append(
                processor_fn(
                    inputs[chunk_index : chunk_index + chunk_size], *args, **kwargs
                )
            )

        return collate_fn(processed_inputs_batches)

    return batchified_processor_fn
