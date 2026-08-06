"""Infrastructure profile for constrained routing."""

import json
import math
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class InfrastructureProfile:
    """Configuration for infrastructure-specific routing constraints.

    Defines turn-angle limits, intermediate structure (tower) placement
    constraints, tower cost models, and gradient limits. Can be loaded
    from YAML/JSON config files or created programmatically.
    """

    name: str
    description: str

    # Turn angle constraints
    soft_angle_limit_deg: float
    hard_angle_limit_deg: float
    angle_cost_function: str  # "linear", "quadratic", "piecewise"
    angle_cost_params: dict = field(default_factory=dict)

    # Span constraints (intermediate structures)
    min_span_m: Optional[float] = None
    max_span_m: Optional[float] = None
    span_bin_size_m: Optional[float] = None

    # Tower/structure cost
    tower_cost_function: str = "terrain_plus_angle"
    tower_cost_params: dict = field(default_factory=dict)

    # Gradient constraints
    max_gradient_percent: Optional[float] = None
    gradient_cost_function: Optional[str] = None
    gradient_cost_params: Optional[dict] = None

    # Catenary / clearance constraints
    tower_height_m: Optional[float] = None
    conductor_weight_per_m: Optional[float] = None  # N/m
    conductor_tension_n: Optional[float] = None  # N
    min_clearance_m: Optional[float] = None  # meters above ground + obstacles

    # Variable tower heights (overrides tower_height_m when set with >1 entry)
    tower_heights_m: Optional[list[float]] = None
    tower_height_cost_per_increment: float = 0.0  # cost per height step above base

    # Terminal towers (start/end anchors connecting to substation/transformer)
    terminal_tower_height_m: Optional[float] = None  # height at substations
    terminal_tower_cost: float = 0.0  # total cost for terminal dead-end tower

    # Cost model
    cost_assumptions_path: Optional[str] = None

    # Tower ground area (foundation footprint)
    tower_ground_area_m2: float = 1.0  # square footprint area in m²
    tower_area_cost_mode: str = "uniform"  # "uniform" or "exact"

    # Corridor width (informational)
    recommended_buffer_m: Optional[float] = None

    @property
    def has_span_constraints(self) -> bool:
        return self.min_span_m is not None and self.max_span_m is not None

    @property
    def has_clearance_constraints(self) -> bool:
        has_height = (self.tower_height_m is not None or
                      (self.tower_heights_m is not None and len(self.tower_heights_m) > 0))
        return (has_height and
                self.conductor_weight_per_m is not None and
                self.conductor_tension_n is not None and
                self.min_clearance_m is not None)

    @property
    def has_variable_height(self) -> bool:
        """True when multiple tower heights are configured."""
        return (self.tower_heights_m is not None and
                len(self.tower_heights_m) > 1)

    @property
    def effective_tower_heights_m(self) -> list[float]:
        """Sorted list of tower heights (descending for early-exit clearance)."""
        if self.tower_heights_m is not None and len(self.tower_heights_m) > 0:
            return sorted(self.tower_heights_m, reverse=True)
        if self.tower_height_m is not None:
            return [self.tower_height_m]
        return []

    def precompute_height_premium(self) -> np.ndarray:
        """Compute cost premium per height class (sorted descending).

        Returns:
            (n_heights,) float32 array: cost premium for each height class.
            Index 0 = tallest (most expensive), last = shortest (base, 0 cost).
        """
        heights = self.effective_tower_heights_m  # already sorted descending
        if not heights:
            return np.zeros(1, dtype=np.float32)
        base_height = min(heights)
        premiums = np.array(
            [(h - base_height) / 3.0 * self.tower_height_cost_per_increment
             if self.tower_height_cost_per_increment > 0 else 0.0
             for h in heights],
            dtype=np.float32,
        )
        return premiums

    @property
    def n_span_bins(self) -> Optional[int]:
        if not self.has_span_constraints:
            return None
        return int(self.max_span_m / self.span_bin_size_m) + 1

    def __post_init__(self):
        """Validate profile configuration."""
        if self.has_span_constraints:
            if self.span_bin_size_m is None or self.span_bin_size_m <= 0:
                raise ValueError("span_bin_size_m must be positive when span constraints are set")
            if self.min_span_m > self.max_span_m:
                raise ValueError(f"min_span_m ({self.min_span_m}) must be <= max_span_m ({self.max_span_m})")
        if self.hard_angle_limit_deg < self.soft_angle_limit_deg:
            raise ValueError("hard_angle_limit_deg must be >= soft_angle_limit_deg")
        if self.tower_ground_area_m2 < 1.0:
            raise ValueError("tower_ground_area_m2 must be >= 1.0")
        if self.tower_area_cost_mode not in ("uniform", "exact"):
            raise ValueError(
                f"tower_area_cost_mode must be 'uniform' or 'exact', "
                f"got '{self.tower_area_cost_mode}'"
            )

    @classmethod
    def from_dict(cls, config: dict) -> "InfrastructureProfile":
        """Create profile from dictionary."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def load(cls, path: str) -> "InfrastructureProfile":
        """Load profile from YAML or JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Profile file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                config = yaml.safe_load(f)
            else:
                config = json.load(f)

        return cls.from_dict(config)

    def to_dict(self) -> dict:
        """Serialize profile to dictionary."""
        from dataclasses import asdict
        return asdict(self)

    def save(self, path: str) -> None:
        """Save profile to YAML or JSON file."""
        data = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            if path.endswith((".yaml", ".yml")):
                import yaml
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, f, indent=2)

    def _compute_angle_cost(self, angle_rad: float) -> float:
        """Compute angle penalty cost for a given turn angle."""
        angle_deg = math.degrees(angle_rad)
        params = self.angle_cost_params

        if self.angle_cost_function == "linear":
            return params.get("scale", 1.0) * angle_deg
        elif self.angle_cost_function == "quadratic":
            return params.get("scale", 1.0) * angle_deg ** 2
        elif self.angle_cost_function == "piecewise":
            breakpoints = params["breakpoints"]
            costs = params["costs"]
            return float(np.interp(angle_deg, breakpoints, costs))
        else:
            raise ValueError(f"Unknown angle cost function: {self.angle_cost_function}")

    def precompute_angle_lut(
        self, steps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Precompute angle cost and validity LUTs for given neighborhood steps.

        Returns:
            angle_cost_lut: (n_dirs, n_dirs) float64 array of angle penalty costs
            angle_valid_lut: (n_dirs, n_dirs) bool array, False where turn is forbidden
        """
        n_dirs = len(steps)
        hard_limit_rad = math.radians(self.hard_angle_limit_deg)

        # Compute direction angles
        angles = np.array([math.atan2(dc, dr) for dr, dc in steps])

        angle_cost_lut = np.zeros((n_dirs, n_dirs), dtype=np.float64)
        angle_valid_lut = np.ones((n_dirs, n_dirs), dtype=bool)

        for i in range(n_dirs):
            for j in range(n_dirs):
                # Turn angle = absolute difference, normalized to [0, pi]
                diff = angles[j] - angles[i]
                turn = abs(math.atan2(math.sin(diff), math.cos(diff)))

                if turn > hard_limit_rad:
                    angle_valid_lut[i, j] = False
                    angle_cost_lut[i, j] = float("inf")
                else:
                    angle_cost_lut[i, j] = self._compute_angle_cost(turn)

        return angle_cost_lut, angle_valid_lut

    def precompute_tower_terrain_costs(self) -> np.ndarray:
        """Precompute tower foundation cost by terrain raster value.

        When tower_ground_area_m2 > 1 and tower_area_cost_mode == "uniform",
        the per-pixel terrain cost is multiplied by the ground area, assuming
        all pixels under the tower footprint have the same value as the center.

        Returns:
            (65536,) float64 array: tower cost for each possible uint16 raster value
        """
        costs = np.zeros(65536, dtype=np.float64)

        if "terrain_cost_map" not in self.tower_cost_params:
            return costs

        terrain_map = self.tower_cost_params["terrain_cost_map"]
        # Parse keys as ints, sort
        points = sorted([(int(k), float(v)) for k, v in terrain_map.items()])

        if not points:
            return costs

        raster_vals = np.array([p[0] for p in points])
        cost_vals = np.array([p[1] for p in points])

        interp = self.tower_cost_params.get("terrain_interpolation", "linear")
        if interp == "linear":
            all_vals = np.arange(65536)
            costs = np.interp(all_vals, raster_vals, cost_vals).astype(np.float64)
        else:
            # Nearest-neighbor fallback
            for i in range(65536):
                idx = np.searchsorted(raster_vals, i)
                idx = min(idx, len(raster_vals) - 1)
                costs[i] = cost_vals[idx]

        if self.tower_area_cost_mode == "uniform" and self.tower_ground_area_m2 != 1.0:
            costs *= self.tower_ground_area_m2

        return costs

    def precompute_tower_angle_costs(self, steps: np.ndarray) -> np.ndarray:
        """Precompute tower type cost based on turn angle at tower location.

        Returns:
            (n_dirs, n_dirs) float64 array: tower-type cost for each (d_in, d_out) pair
        """
        n_dirs = len(steps)
        lut = np.zeros((n_dirs, n_dirs), dtype=np.float64)

        angle_types = self.tower_cost_params.get("angle_types", {})
        if not angle_types:
            return lut

        # Sort types by max_angle_deg ascending
        sorted_types = sorted(
            angle_types.values(), key=lambda t: t["max_angle_deg"]
        )

        angles = np.array([math.atan2(dc, dr) for dr, dc in steps])

        for i in range(n_dirs):
            for j in range(n_dirs):
                diff = angles[j] - angles[i]
                turn_deg = abs(math.degrees(math.atan2(math.sin(diff), math.cos(diff))))

                for tower_type in sorted_types:
                    if turn_deg <= tower_type["max_angle_deg"]:
                        lut[i, j] = tower_type["base_cost"]
                        break
                else:
                    # Exceeds all types — use the most expensive
                    if sorted_types:
                        lut[i, j] = sorted_types[-1]["base_cost"]

        return lut

    def compute_step_distances(
        self, steps: np.ndarray, cell_size: float
    ) -> np.ndarray:
        """Compute physical distance for each neighborhood step.

        Returns:
            (n_dirs,) float64 array: distance in meters for each step
        """
        return np.array(
            [math.sqrt(dr**2 + dc**2) * cell_size for dr, dc in steps],
            dtype=np.float64,
        )
