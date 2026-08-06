"""
PYORPS: An Open-Source Tool for Automated Power Line Routing

Feasibility objective for multi-criteria routing.

An :class:`Objective` describes *what the search minimizes*: a weighted sum of
named metric layers (cost, landscape, permit, ...) plus the reserved layers

- ``"length"``  — implicit constant layer (no raster behind it), and
- ``"gradient"`` — the along-route slope of each step, computed inside the
  search kernels from the DEM.

It also owns the gradient *response* configuration and the builder for the
slope-response lookup tables (``Γ_mult`` / ``Γ_add``) consumed by the kernels.
The 3D length stretch ``sqrt(1 + (s/100)^2)`` is unconditional geometry whenever
a DEM is present — it is always folded into ``Γ_mult`` and is not configurable.

See ``docs/superpowers/plans/2026-08-04-feasibility-multi-objective-routing.md``.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .exceptions import ObjectiveError

#: Layer names that never require a raster layer behind them.
#: "length" and "gradient" are consumed by every backend; "smoothness"
#: (scales the turn-angle penalty LUT) and "tower" (scales the tower cost
#: terms) are consumed by constrained routing only — their DEFAULT is 1.0
#: (the profile LUTs are already the steering magnitudes), an explicit 0
#: disables the term.
RESERVED_LAYERS: tuple[str, ...] = ("length", "gradient", "smoothness",
                                    "tower")

#: Cap applied by the legacy multiplicative penalty models
#: (mirrors ``_traversal_numba.calculate_gradient_penalty``).
_LEGACY_PENALTY_CAP = 10000.0


# ---------------------------------------------------------------------------
# Multiplicative response models (legacy parity)
#
# These replicate the formulas of ``calculate_gradient_penalty`` in
# ``pyorps/utils/_traversal_numba.py`` exactly, expressed over slope in
# percent (the legacy code uses the rise/run ratio: ratio = s_pct / 100).
# ---------------------------------------------------------------------------

def _mult_exponential(s_pct: np.ndarray, params: dict) -> np.ndarray:
    scale = params.get("scale", 2.0)
    ratio = s_pct / 100.0
    return np.minimum(np.exp(ratio * ratio * scale), _LEGACY_PENALTY_CAP)


def _mult_power(s_pct: np.ndarray, params: dict) -> np.ndarray:
    ratio = s_pct / 100.0
    normalized = np.abs(np.arctan(ratio)) / (np.pi / 2.0)
    exp_adjust = 1.0 + 4.0 * normalized
    return np.minimum(1.0 + (100.0 * normalized) ** exp_adjust,
                      _LEGACY_PENALTY_CAP)


def _mult_energy(s_pct: np.ndarray, params: dict) -> np.ndarray:
    scale = params.get("scale", 3.0)
    ratio = s_pct / 100.0
    stretch = np.sqrt(1.0 + ratio * ratio)
    grade = np.sin(np.arctan(ratio))
    return np.minimum(stretch * np.exp(scale * grade * grade),
                      _LEGACY_PENALTY_CAP)


def _mult_sigmoid(s_pct: np.ndarray, params: dict) -> np.ndarray:
    center_deg = params.get("center_deg", 15.0)
    steepness = params.get("steepness", 0.2)
    max_penalty = params.get("max_penalty", 1000.0)
    deg = np.degrees(np.arctan(s_pct / 100.0))
    x = (np.abs(deg) - center_deg) * steepness
    return 1.0 + (max_penalty - 1.0) / (1.0 + np.exp(-x))


def _mult_squared(s_pct: np.ndarray, params: dict) -> np.ndarray:
    scale = params.get("scale", 400.0)
    normalized = np.abs(np.arctan(s_pct / 100.0)) / (np.pi / 2.0)
    return 1.0 + scale * normalized * normalized


MULTIPLIER_MODELS: dict[str, Callable[[np.ndarray, dict], np.ndarray]] = {
    "exponential": _mult_exponential,
    "power": _mult_power,
    "energy": _mult_energy,
    "sigmoid": _mult_sigmoid,
    "squared": _mult_squared,
}


# ---------------------------------------------------------------------------
# Additive response shapes  g(s)  (weighted by objective["gradient"])
# ---------------------------------------------------------------------------

def _add_linear(s_pct: np.ndarray, params: dict) -> np.ndarray:
    return s_pct.copy()


def _add_quadratic(s_pct: np.ndarray, params: dict) -> np.ndarray:
    return s_pct * s_pct


def _add_exponential(s_pct: np.ndarray, params: dict) -> np.ndarray:
    scale = params.get("scale", 2.0)
    ratio = s_pct / 100.0
    return np.exp(scale * ratio * ratio) - 1.0


ADDITIVE_MODELS: dict[str, Callable[[np.ndarray, dict], np.ndarray]] = {
    "linear": _add_linear,
    "quadratic": _add_quadratic,
    "exponential": _add_exponential,
}


def _resolve_response(
        model: str | Callable | None,
        registry: dict[str, Callable],
        kind: str,
) -> Callable[[np.ndarray, dict], np.ndarray] | None:
    """Resolve a response spec to a callable ``f(s_pct_array, params)``."""
    if model is None:
        return None
    if callable(model):
        return lambda s, params: np.asarray(model(s), dtype=np.float64)
    if isinstance(model, str):
        if model in registry:
            return registry[model]
        raise ObjectiveError(
            f"Unknown {kind} gradient response '{model}'. "
            f"Available: {sorted(registry)} or a callable f(slope_percent)."
        )
    raise ObjectiveError(
        f"{kind} gradient response must be a model name, a callable or None, "
        f"got {type(model)}"
    )


@dataclass
class GradientOptions:
    """Configuration of the slope *response* (never the geometry).

    The 3D length stretch is applied unconditionally whenever a DEM is
    present; these options only control how slope additionally contributes
    to the feasibility.
    """

    #: Shape g(s) of the additive exposure term (weighted by
    #: ``objective["gradient"]``): "linear" | "quadratic" | "exponential"
    #: or a callable ``f(slope_percent_array) -> array``.
    additive: str | Callable | None = None
    additive_params: dict = field(default_factory=dict)

    #: Optional multiplicative penalty model applied to the whole per-meter
    #: feasibility: one of the five legacy models or a callable. None => x1.
    multiplier: str | Callable | None = None
    multiplier_params: dict = field(default_factory=dict)

    #: Hard grade limit: edges steeper than this are forbidden.
    max_gradient_pct: float | None = None

    #: Slope discretization of the response LUTs.
    bin_width_pct: float = 0.25

    #: Upper end of the represented slope domain; steeper slopes clamp to
    #: the last bin (which is forbidden anyway when max_gradient_pct is set
    #: below this value).
    s_max_pct: float = 200.0

    def __post_init__(self):
        if self.bin_width_pct <= 0:
            raise ObjectiveError(
                f"bin_width_pct must be > 0, got {self.bin_width_pct}")
        if self.s_max_pct < self.bin_width_pct:
            raise ObjectiveError(
                f"s_max_pct ({self.s_max_pct}) must be >= bin_width_pct "
                f"({self.bin_width_pct})")
        if self.max_gradient_pct is not None:
            if self.max_gradient_pct <= 0:
                raise ObjectiveError(
                    f"max_gradient_pct must be > 0, got "
                    f"{self.max_gradient_pct}")
            # The limit must be representable inside the LUT domain.
            if self.max_gradient_pct >= self.s_max_pct:
                self.s_max_pct = self.max_gradient_pct + 8 * self.bin_width_pct
        # Fail fast on unknown model names (callables validated on use).
        _resolve_response(self.additive, ADDITIVE_MODELS, "additive")
        _resolve_response(self.multiplier, MULTIPLIER_MODELS, "multiplicative")

    def to_dict(self) -> dict:
        def _serialize(model):
            if model is None or isinstance(model, str):
                return model
            return f"callable:{getattr(model, '__qualname__', repr(model))}"

        return {
            "additive": _serialize(self.additive),
            "additive_params": dict(self.additive_params),
            "multiplier": _serialize(self.multiplier),
            "multiplier_params": dict(self.multiplier_params),
            "max_gradient_pct": self.max_gradient_pct,
            "bin_width_pct": self.bin_width_pct,
            "s_max_pct": self.s_max_pct,
        }

    @classmethod
    def from_dict(cls, config: dict) -> "GradientOptions":
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in config.items() if k in known}
        for key in ("additive", "multiplier"):
            value = filtered.get(key)
            if isinstance(value, str) and value.startswith("callable:"):
                raise ObjectiveError(
                    f"Cannot restore callable gradient response {value!r} "
                    f"from a serialized objective — pass the callable again."
                )
        return cls(**filtered)


@dataclass
class GradientLUTs:
    """Slope-response lookup tables consumed by the search kernels.

    Kernel contract (see plan section 3.2)::

        s = |DEM[v] - DEM[u]| * inv_horiz_m[d]        # percent
        b = min(int(s * bin_inv), n_bins - 1)
        w = W_sum * cost_factor[d] * mult[b] + add[b] * dist_cells[d]

    ``mult[b] == inf`` marks the edge forbidden (hard grade limit).
    ``add`` is pre-scaled by the quantization scale so it is commensurable
    with the quantized terrain term without consuming uint16 resolution.
    """

    mult: np.ndarray          # float32 (n_bins,)
    add: np.ndarray           # float32 (n_bins,)
    packed: np.ndarray        # float32 (n_bins, 2) C-contiguous [mult, add]
    inv_horiz_m: np.ndarray   # float32 (n_dirs,) = 1 / (step_len_cells * cell_size)
    bin_inv: float            # 1 / bin_width_pct
    n_bins: int
    bin_width_pct: float
    max_gradient_pct: float | None
    quant_scale: float
    #: Per-direction factor mapping |Δh| in meters straight to a bin index:
    #: ``b = min(int(|Δh| * bin_factor[d]), n_bins - 1)`` — folds the
    #: percent conversion and bin width into one multiply.
    bin_factor: np.ndarray = None      # float32 (n_dirs,)
    #: Per-direction step length in CELL units (multiplies Γ_add).
    step_len_cells: np.ndarray = None  # float32 (n_dirs,)

    @property
    def max_finite_mult(self) -> float:
        """Largest finite multiplier (for delta/bucket sizing bounds)."""
        finite = self.mult[np.isfinite(self.mult)]
        return float(finite.max()) if finite.size else 1.0

    @property
    def max_add(self) -> float:
        finite = self.add[np.isfinite(self.add)]
        return float(finite.max()) if finite.size else 0.0


class Objective:
    """A feasibility objective: ``F(path) = sum_k w_k * M_k(path)``.

    Parameters:
        weights: Mapping of layer name -> non-negative weight. Defaults to
            ``{"cost": 1.0}`` (today's behavior). The reserved names
            ``"length"`` and ``"gradient"`` need no raster layer.
        gradient_options: :class:`GradientOptions`, a plain dict, or None.
            When ``weights["gradient"] > 0`` and no additive response is
            configured, ``additive="linear"`` is used.
    """

    def __init__(
            self,
            weights: dict[str, float] | None = None,
            gradient_options: GradientOptions | dict | None = None,
    ):
        if weights is None:
            weights = {"cost": 1.0}
        if not isinstance(weights, dict) or not weights:
            raise ObjectiveError(
                "Objective weights must be a non-empty dict of "
                "{layer_name: weight}")

        clean: dict[str, float] = {}
        for name, value in weights.items():
            if not isinstance(name, str) or not name:
                raise ObjectiveError(
                    f"Layer names must be non-empty strings, got {name!r}")
            try:
                weight = float(value)
            except (TypeError, ValueError):
                raise ObjectiveError(
                    f"Weight for layer '{name}' must be numeric, "
                    f"got {value!r}") from None
            if not math.isfinite(weight) or weight < 0:
                raise ObjectiveError(
                    f"Weight for layer '{name}' must be finite and >= 0, "
                    f"got {weight}")
            clean[name] = weight

        if not any(w > 0 for w in clean.values()):
            raise ObjectiveError(
                "At least one objective weight must be > 0 — an all-zero "
                "objective has no minimum.")

        if gradient_options is None:
            gradient_options = GradientOptions()
        elif isinstance(gradient_options, dict):
            gradient_options = GradientOptions.from_dict(gradient_options)
        elif not isinstance(gradient_options, GradientOptions):
            raise ObjectiveError(
                f"gradient_options must be GradientOptions, dict or None, "
                f"got {type(gradient_options)}")

        if clean.get("gradient", 0.0) > 0 and gradient_options.additive is None:
            gradient_options.additive = "linear"

        self._weights = clean
        self.gradient_options = gradient_options

    # ------------------------------------------------------------------ API

    @property
    def weights(self) -> dict[str, float]:
        """Copy of the weight mapping (mutate via a new Objective)."""
        return dict(self._weights)

    def __getitem__(self, layer: str) -> float:
        return self._weights.get(layer, 0.0)

    @property
    def layer_names(self) -> list[str]:
        """All weighted layer names, reserved ones included."""
        return list(self._weights)

    @property
    def required_layers(self) -> list[str]:
        """Positively weighted layers that need a raster layer behind them."""
        return [name for name, w in self._weights.items()
                if w > 0 and name not in RESERVED_LAYERS]

    @property
    def has_gradient_terms(self) -> bool:
        """True when slope influences the objective beyond pure geometry."""
        opts = self.gradient_options
        return (self._weights.get("gradient", 0.0) > 0
                or opts.multiplier is not None
                or opts.max_gradient_pct is not None)

    def validate_layers(self, available: set[str] | list[str]) -> None:
        """Raise if a weighted layer has no metric layer behind it."""
        known = set(available) | set(RESERVED_LAYERS)
        unknown = [name for name in self._weights if name not in known]
        if unknown:
            raise ObjectiveError(
                f"Objective references unknown metric layer(s) {unknown}. "
                f"Available layers: {sorted(set(available))}; reserved: "
                f"{list(RESERVED_LAYERS)}."
            )

    # ------------------------------------------------------------- presets

    @classmethod
    def cheapest(cls) -> "Objective":
        """Minimize monetary cost — the default, today's behavior."""
        return cls({"cost": 1.0})

    @classmethod
    def shortest(cls) -> "Objective":
        """Minimize route length (3D length when a DEM is present)."""
        return cls({"length": 1.0})

    @classmethod
    def gentlest(cls, length_eps: float = 0.01) -> "Objective":
        """Minimize gradient exposure, with a small length regularization.

        The length term prevents degenerate wandering across flat terrain
        where the gradient contribution is zero.
        """
        if length_eps <= 0:
            raise ObjectiveError(
                f"length_eps must be > 0 (it prevents zero-weight "
                f"degeneracy), got {length_eps}")
        return cls({"gradient": 1.0, "length": length_eps})

    @classmethod
    def from_priorities(
            cls,
            priorities: dict[str, float],
            scales: dict[str, float],
            gradient_options: GradientOptions | dict | None = None,
    ) -> "Objective":
        """Build weights from relative priorities and per-layer scales.

        ``weight_k = priority_k / scale_k`` — with ``scale_k`` a typical
        magnitude of layer k (e.g. its p95 over traversable cells), the
        priorities express *relative influence*. Scales are window-dependent:
        persist the RESOLVED weights (``.weights``) with results, never
        re-derive them silently.

        Reserved layers default to scale 1.0 when not provided.
        """
        weights = {}
        for name, priority in priorities.items():
            if name in scales:
                scale = float(scales[name])
            elif name in RESERVED_LAYERS:
                scale = 1.0
            else:
                raise ObjectiveError(
                    f"from_priorities: no scale provided for layer '{name}'. "
                    f"Pass scales={{'{name}': <typical magnitude>}} (e.g. the "
                    f"layer's p95); MetricStack will supply these "
                    f"automatically in a later phase.")
            if scale <= 0 or not math.isfinite(scale):
                raise ObjectiveError(
                    f"Scale for layer '{name}' must be finite and > 0, "
                    f"got {scale}")
            weights[name] = float(priority) / scale
        return cls(weights, gradient_options)

    # ------------------------------------------------------ LUT construction

    def build_gradient_luts(
            self,
            steps: np.ndarray,
            cell_size: float,
            quant_scale: float = 1.0,
    ) -> GradientLUTs:
        """Build the slope-response LUT pair for the search kernels.

        The 3D length stretch ``sqrt(1 + (s/100)^2)`` is ALWAYS folded into
        both tables (geometry — not configurable). The configured responses
        ride on top of it.

        Parameters:
            steps: (n_dirs, 2) array of (dr, dc) neighborhood steps.
            cell_size: Raster cell size in meters (> 0).
            quant_scale: The feasibility-quantization scale (section 5.2 of
                the plan); pre-multiplied into the additive table so it is
                commensurable with the quantized terrain term.
        """
        if cell_size <= 0 or not math.isfinite(cell_size):
            raise ObjectiveError(
                f"cell_size must be finite and > 0, got {cell_size}")
        if quant_scale <= 0 or not math.isfinite(quant_scale):
            raise ObjectiveError(
                f"quant_scale must be finite and > 0, got {quant_scale}")

        steps = np.asarray(steps)
        if steps.ndim != 2 or steps.shape[1] != 2 or steps.shape[0] == 0:
            raise ObjectiveError(
                f"steps must be a (n_dirs, 2) array, got shape {steps.shape}")

        opts = self.gradient_options
        n_bins = int(math.ceil(opts.s_max_pct / opts.bin_width_pct))
        # Evaluate responses at bin centers (kernel bins by floor()).
        s_centers = (np.arange(n_bins, dtype=np.float64) + 0.5) \
            * opts.bin_width_pct

        ratio = s_centers / 100.0
        stretch = np.sqrt(1.0 + ratio * ratio)

        mult_response = _resolve_response(
            opts.multiplier, MULTIPLIER_MODELS, "multiplicative")
        if mult_response is not None:
            response = np.asarray(
                mult_response(s_centers, opts.multiplier_params),
                dtype=np.float64)
            self._check_response(response, "multiplicative", strict_pos=True)
            mult = stretch * response
        else:
            mult = stretch.copy()

        w_gradient = self._weights.get("gradient", 0.0)
        if w_gradient > 0:
            add_response = _resolve_response(
                opts.additive, ADDITIVE_MODELS, "additive")
            g_values = np.asarray(
                add_response(s_centers, opts.additive_params),
                dtype=np.float64)
            self._check_response(g_values, "additive", strict_pos=False)
            add = w_gradient * g_values * stretch * quant_scale
        else:
            add = np.zeros(n_bins, dtype=np.float64)

        if opts.max_gradient_pct is not None:
            forbidden = s_centers > opts.max_gradient_pct
            mult[forbidden] = np.inf
            add[forbidden] = 0.0

        lengths = np.sqrt((steps[:, 0].astype(np.float64)) ** 2
                          + (steps[:, 1].astype(np.float64)) ** 2)
        if np.any(lengths <= 0):
            raise ObjectiveError("steps must not contain the (0, 0) step")
        inv_horiz_m = (1.0 / (lengths * cell_size)).astype(np.float32)
        bin_inv = 1.0 / opts.bin_width_pct
        bin_factor = (100.0 * inv_horiz_m.astype(np.float64)
                      * bin_inv).astype(np.float32)

        mult32 = mult.astype(np.float32)
        add32 = add.astype(np.float32)
        packed = np.ascontiguousarray(
            np.stack([mult32, add32], axis=1))

        return GradientLUTs(
            mult=mult32,
            add=add32,
            packed=packed,
            inv_horiz_m=inv_horiz_m,
            bin_inv=bin_inv,
            n_bins=n_bins,
            bin_width_pct=opts.bin_width_pct,
            max_gradient_pct=opts.max_gradient_pct,
            quant_scale=quant_scale,
            bin_factor=bin_factor,
            step_len_cells=lengths.astype(np.float32),
        )

    @staticmethod
    def _check_response(values: np.ndarray, kind: str,
                        strict_pos: bool) -> None:
        if not np.all(np.isfinite(values)):
            raise ObjectiveError(
                f"The {kind} gradient response produced non-finite values.")
        if strict_pos:
            if np.any(values <= 0):
                raise ObjectiveError(
                    f"The {kind} gradient response must be strictly "
                    f"positive (zero would create free edges; negative "
                    f"weights break Dijkstra).")
        elif np.any(values < 0):
            raise ObjectiveError(
                f"The {kind} gradient response must be >= 0 (negative "
                f"edge weights break Dijkstra).")

    # -------------------------------------------------------- serialization

    def to_dict(self) -> dict:
        return {
            "weights": dict(self._weights),
            "gradient_options": self.gradient_options.to_dict(),
        }

    @classmethod
    def from_dict(cls, config: dict) -> "Objective":
        if "weights" not in config:
            raise ObjectiveError(
                "Serialized objective must contain a 'weights' entry")
        return cls(config["weights"],
                   config.get("gradient_options"))

    def fingerprint(self) -> str:
        """Stable content hash of the objective (weights + responses).

        Callable responses hash by qualified name only — prefer named
        models when reproducibility across sessions matters.
        """
        canonical = json.dumps(self.to_dict(), sort_keys=True,
                               separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    # -------------------------------------------------------------- dunders

    def __repr__(self) -> str:
        parts = ", ".join(f"{k}={v:g}" for k, v in self._weights.items())
        extra = ""
        if self.has_gradient_terms:
            opts = self.gradient_options
            details = []
            if opts.additive is not None and self._weights.get(
                    "gradient", 0.0) > 0:
                details.append(f"additive={opts.additive}")
            if opts.multiplier is not None:
                details.append(f"multiplier={opts.multiplier}")
            if opts.max_gradient_pct is not None:
                details.append(f"max_gradient={opts.max_gradient_pct:g}%")
            if details:
                extra = "; " + ", ".join(str(d) for d in details)
        return f"Objective({parts}{extra})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Objective):
            return NotImplemented
        return self.to_dict() == other.to_dict()
