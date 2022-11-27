import time
from typing import NamedTuple, Any, Dict, Callable, Tuple

import torch
from torch import Tensor
from thre3d_atom.utils.imaging_utils import SceneBounds


ExtraInfo = Dict[str, Any]  # Type for extra information


class SampledPointsOnRays(NamedTuple):
    points: Tensor  # shape [N x num_samples x NUM_COORD_DIMENSIONS]
    depths: Tensor  # shape [N x num_samples]


class RenderOut(NamedTuple):
    colour: Tensor  # shape [N x NUM_COLOUR_CHANNELS]
    disparity: Tensor  # shape [N x 1]
    extra: ExtraInfo = {}  # extra information

    def detach(self) -> Any:
        return RenderOut(
            colour=self.colour.detach(),
            disparity=self.disparity.detach(),
            extra={key: value.detach() for key, value in self.extra.items()},
        )

    def to(self, device: torch.device) -> Any:
        return RenderOut(
            colour=self.colour.to(device),
            disparity=self.disparity.to(device),
            extra={key: value.to(device) for key, value in self.extra.items()},
        )


class Rays(NamedTuple):
    origins: Tensor  # shape [N x NUM_COORD_DIMENSIONS]
    directions: Tensor  # shape [N x NUM_COORD_DIMENSIONS]

    def __getitem__(self, item) -> Any:
        """This is overridden to allow indexing and slicing of rays"""
        return Rays(
            origins=self.origins[item, :],
            directions=self.directions[item, :],
        )

    def __len__(self) -> int:
        return len(self.origins)

    def to(self, device: torch.device) -> Any:
        """This is shorthand to transfer a bunch of rays to GPU"""
        return Rays(self.origins.to(device), self.directions.to(device))


# The dataType is exactly the same, but this renaming improves readability
ProcessedPointsOnRays = SampledPointsOnRays
# the int is the number of sampled points
RaySamplerFunction = Callable[[Rays, SceneBounds, int], SampledPointsOnRays]
PointProcessorFunction = Callable[[SampledPointsOnRays, Rays], ProcessedPointsOnRays]
AccumulatorFunction = Callable[[ProcessedPointsOnRays, Rays], RenderOut]
ProfilingInfo = Dict[str, float]


def render(
    rays: Rays,
    scene_bounds: SceneBounds,
    num_samples: int,
    sampler_fn: RaySamplerFunction,
    point_processor_fn: PointProcessorFunction,
    accumulator_fn: AccumulatorFunction,
) -> Tuple[RenderOut, ProfilingInfo]:
    """
    Defines the overall flow of execution of the differentiable
    volumetric rendering process. Please note that this interface has been
    designed to allow enough flexibility of the rendering process.
    Args:
        rays: virtual casted rays (origins and directions).
        scene_bounds: SceneBounds (near and far) of the scene being rendered
        num_samples: number of points sampled on the rays
        sampler_fn: function that maps from casted rays to sampled points on the rays.
        point_processor_fn: function to process the points on the rays.
        accumulator_fn: function that accumulates the processed points into rendered
                        output.
    Returns: rendered output (rgb, disparity and extra information)
    """
    profiling_info: ProfilingInfo = {}

    start_time = time.time()
    last_time = start_time

    sampled_points = sampler_fn(rays, scene_bounds, num_samples)

    time_taken = (time.time() - last_time) * 1000
    profiling_info["sampling"] = time_taken
    last_time = time.time()

    processed_points = point_processor_fn(sampled_points, rays)

    time_taken = (time.time() - last_time) * 1000
    profiling_info["processing"] = time_taken
    last_time = time.time()

    rendered_output = accumulator_fn(processed_points, rays)

    time_taken = (time.time() - last_time) * 1000
    profiling_info["accumulation"] = time_taken

    return rendered_output, profiling_info
