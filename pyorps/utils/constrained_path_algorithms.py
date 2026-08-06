"""Backward-compatibility shim — imports from new OO modules."""
from pyorps.utils._constrained_context import (
    pack_state, unpack_state,
    pack_state_h, unpack_state_h,
)
from pyorps.utils._constrained_dijkstra import constrained_dijkstra_2d
from pyorps.utils._constrained_delta import (
    constrained_delta_stepping_2d,
    constrained_delta_stepping_clearance_2d,
    constrained_delta_stepping_height_2d,
    constrained_delta_stepping_lazy,
)
