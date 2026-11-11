"""Verified 110 kV towers from OpenStreetMap lines (owner rule, 2026-10-06).

OpenStreetMap tower data is not reliable (a tower often carries no voltage tag, only 5 of 4 977 points do). The line
data is better. A support point (tower or pole) that lies on a line whose ``voltage`` field contains ``110000`` is taken
as a true 110 kV tower. A tower node is a vertex of its line, so it lies on the line to the digit; the tolerance only
absorbs digitising noise.

Railway traction lines (operator DB Energie, or frequency 16.7 Hz) also carry the 110000 tag but cannot take a wind-farm
tap. Owner decision D-e (2026-10-06): drop them. ``exclude_traction=False`` keeps them, and the ``traction`` column
lets a caller report the difference either way.
"""

from __future__ import annotations

from pyorps.core.exceptions import InvalidInputError

__all__ = ["verified_110kv_towers"]

SUPPORT_TYPES = ("tower", "pole")
#: ``operator`` substring (case-insensitive) that marks a railway traction line: the German railway's energy company.
TRACTION_OPERATOR = "DB Energie"
#: ``frequency`` substring that marks a railway traction line: the German 16.7 Hz traction supply.
TRACTION_FREQUENCY = "16.7"


def verified_110kv_towers(lines, towers, *, tol_m: float = 1.0, voltage_token: str = "110000",
                          line_power=("line",), tower_power=SUPPORT_TYPES, exclude_traction: bool = True,
                          traction_operator: str = TRACTION_OPERATOR, traction_frequency: str = TRACTION_FREQUENCY):
    """The towers that lie on a line whose ``voltage`` contains ``voltage_token``.

    Args:
        lines: GeoDataFrame of LineStrings with ``voltage`` and ``power`` (and optionally ``operator``, ``frequency``,
            ``osm_id``), in a metric CRS.
        towers: GeoDataFrame of Points with ``power``, in the same CRS.
        tol_m: Largest distance from the line in metres.
        voltage_token: Substring of the ``voltage`` field that marks a 110 kV line (also matches ``380000;110000``).
        line_power, tower_power: ``power`` values accepted for lines and for support points.
        exclude_traction: Drop railway traction lines and their towers (default, decision D-e).
        traction_operator: Substring of the ``operator`` field (case-insensitive) that marks a traction line.
            Default: the German railway's energy company (:data:`TRACTION_OPERATOR`). An empty string matches every
            line that has an ``operator`` field, so pass a real name or disable the filter with
            ``exclude_traction=False``.
        traction_frequency: Substring of the ``frequency`` field that marks a traction line. Default: the German
            railway supply, 16.7 Hz (:data:`TRACTION_FREQUENCY`).

    The defaults of ``voltage_token``, ``tol_m``, ``traction_operator`` and ``traction_frequency`` are the German
    110 kV and railway values; pass others for another country.

    Returns:
        The matching towers with the added columns ``line_osm_id``, ``line_voltage``, ``traction`` and ``line_dist_m``.
    """
    import geopandas as gpd

    if lines.crs is None or towers.crs is None or lines.crs != towers.crs or not lines.crs.is_projected:
        raise InvalidInputError("lines and towers must share one projected CRS")
    sel = (lines["voltage"].fillna("").astype(str).str.contains(voltage_token, regex=False)
           & lines["power"].isin(line_power))
    ln = lines[sel].copy()
    operator = ln["operator"].fillna("").astype(str) if "operator" in ln else ln["power"].astype(str).str.slice(0, 0)
    freq = ln["frequency"].fillna("").astype(str) if "frequency" in ln else operator.str.slice(0, 0)
    ln["traction"] = operator.str.contains(traction_operator, case=False, regex=False) \
        | freq.str.contains(traction_frequency, regex=False)
    if exclude_traction:
        ln = ln[~ln["traction"]]
    tw = towers[towers["power"].isin(tower_power)]
    if ln.empty or tw.empty:
        return tw.iloc[0:0].assign(line_osm_id=None, line_voltage=None, traction=False, line_dist_m=0.0)
    right = ln.assign(line_osm_id=ln["osm_id"] if "osm_id" in ln else ln.index.astype(str),
                      line_voltage=ln["voltage"])[["line_osm_id", "line_voltage", "traction", "geometry"]]
    joined = gpd.sjoin_nearest(tw, right, how="inner", max_distance=float(tol_m), distance_col="line_dist_m")
    joined = joined[~joined.index.duplicated(keep="first")].drop(columns="index_right")
    return joined
