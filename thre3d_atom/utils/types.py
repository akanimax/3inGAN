from typing import NamedTuple, Tuple


class AxisAlignedBoundingBox(NamedTuple):
    # min-max range values for all three dimensions
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_range: Tuple[float, float]
