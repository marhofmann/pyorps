"""Static configuration for the MV-Oberrhein history benchmark.

Everything that defines *what* is measured lives here so that the runner,
the worker and the scorer cannot drift apart.

The case being reproduced is the CIRED 2025 case study
(``case_studies/mv_oberrhein/case_study_mv_oberrhein.py``): connect one PV
plant to the MV-Oberrhein grid, evaluating the 8 candidate points of common
coupling with neighbourhoods R0..R3 on a 1 m ALKIS cost raster.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Everything the benchmark writes lives OUTSIDE the repository, so the 1.2 GB
#: reference raster and the per-version checkouts can never reach git.
DATA_ROOT = Path(
    os.environ.get("PYORPS_BENCH_ROOT", Path.home() / "Documents" / "pyorps_benchmarks")
).resolve()

INPUTS = DATA_ROOT / "inputs"
CHECKOUTS = DATA_ROOT / "checkouts"
RUNS = DATA_ROOT / "runs"

ALKIS = INPUTS / "alkis_mvo.gpkg"
BBOX = INPUTS / "bbox.geojson"
SOURCE = INPUTS / "source.geojson"
TARGETS = INPUTS / "targets.geojson"
REF_RASTER = INPUTS / "ref_raster.tiff"

#: The shared search window, exported once so that the 2021 scikit-image
#: prototype solves EXACTLY the problem pyorps solves -- same cells, same
#: costs, same source/target pixels.
WINDOW_COST = INPUTS / "window_cost.npy"
WINDOW_META = INPUTS / "window_meta.json"

# --------------------------------------------------------------------------
# The case study
# --------------------------------------------------------------------------

#: ALKIS "Tatsaechliche Nutzung" -> construction cost, verbatim from the case
#: study. 65535 marks the maximum-cost class; the case study runs with
#: ``ignore_max_cost=False``, i.e. those cells stay traversable but ruinous.
COST_ASSUMPTIONS = {
    'objektname': {
        'Wohnbaufläche': 65535, 'Industrie- und Gewerbefläche': 65535,
        'Fläche besonderer funktionaler Prägung': 65535,
        'Tagebau/Grube/Steinbruch': 65535, 'Friedhof': 65535, 'Halde': 65535,
        'Sumpf': 65535, 'Flugverkehr': 65535,
        'Straßenverkehr': 178, 'Sport-, Freizeit- und Erholungsfläche': 107,
        'Weg': 97, 'Landwirtschaft': 285, 'Wald': 365, 'Fließgewässer': 155,
        'Gehölz': 365, 'Fläche gemischter Nutzung': 107, 'Platz': 152,
        'Unland/Vegetationslose Fläche': 92, 'Stehendes Gewässer': 155,
        'Bahnverkehr': 415,
    }
}

WFS_REQUEST = {
    'url': "https://owsproxy.lgl-bw.de/owsproxy/wfs/WFS_LGL-BW_ALKIS?version=2.0.0",
    'layer': "Tatsächliche Nutzung",
}

CRS = "EPSG:25832"

#: The 8 candidate PCC buses, in the order the case study lists them.
CASE_BUS_ORDER = [1, 2, 133, 141, 138, 170, 101, 106]

#: R3 is run in clusters of two targets in the case study, "due to memory
#: constraints". Reproduced verbatim, by bus id.
R3_CLUSTERS = [[133, 141], [138, 170], [1, 2], [101, 106]]

IGNORE_MAX_COST = False

# --------------------------------------------------------------------------
# Versions under test
# --------------------------------------------------------------------------
# ``ref`` is resolved with ``git archive`` (read-only; no worktree, no
# checkout, no index mutation). ``worktree`` means "use the working tree as it
# is right now", which is the only way to measure uncommitted work.

VERSIONS: list[dict] = [
    dict(label="neis2021", ref=None, kind="prototype", date="2021-09",
         note="pre-pyorps scikit-image MCP prototype (NEIS 2021 paper)"),
    dict(label="v0.1.0", ref="v0.1.0", kind="tag", date="2025-06-03",
         default_api="networkit",
         note="first public release; networkit default; materialised graph"),
    dict(label="v0.1.4", ref="v0.1.4", kind="tag", date="2025-06-03",
         default_api="networkit", note="end of the 0.1 line"),
    dict(label="v0.2.1", ref="v0.2.1", kind="tag", date="2025-09-03",
         default_api="cython",
         note="cython default; raster-direct search, no graph materialisation"),
    dict(label="v0.2.3", ref="v0.2.3", kind="tag", date="2025-09-08",
         default_api="cython", note="end of the 0.2 line"),
    dict(label="v0.3.0", ref="v0.3.0", kind="tag", date="2026-03-23",
         default_api="cython", note="GPU backend + DEM / 3D routing"),
    dict(label="v0.3.2", ref="v0.3.2", kind="tag", date="2026-03-23",
         default_api="cython", note="latest tagged release"),
    dict(label="HEAD", ref=None, kind="worktree", date="today",
         default_api="cython",
         note="current working tree (includes uncommitted work)"),
]

#: Versions skipped unless --profile full: same-day patch releases whose
#: routing behaviour is not expected to differ from their neighbours.
EXTRA_VERSIONS: list[dict] = [
    dict(label="v0.1.1", ref="v0.1.1", kind="tag", date="2025-06-03", note=""),
    dict(label="v0.1.2", ref="v0.1.2", kind="tag", date="2025-06-03", note=""),
    dict(label="v0.1.3", ref="v0.1.3", kind="tag", date="2025-06-03", note=""),
    dict(label="v0.2.2", ref="v0.2.2", kind="tag", date="2025-09-05", note=""),
    dict(label="v0.3.1", ref="v0.3.1", kind="tag", date="2026-03-23", note=""),
]

NEIGHBORHOODS = ["r0", "r1", "r2", "r3"]

#: Backends swept on HEAD. Availability is probed at runtime; a missing
#: package is RECORDED as unavailable, never silently dropped.
HEAD_BACKENDS = [
    dict(graph_api="cython", use_gpu=False),
    dict(graph_api="networkit", use_gpu=False),
    dict(graph_api="networkx", use_gpu=False),
    dict(graph_api="igraph", use_gpu=False),
    dict(graph_api="rustworkx", use_gpu=False),
    dict(graph_api="raster_gpu", use_gpu=False),
    dict(graph_api="raster_fim", use_gpu=False),
]

#: The backend every version has in common, used for the controlled
#: apples-to-apples track.
CONTROL_BACKEND = "networkit"

# --------------------------------------------------------------------------
# Resource model
# --------------------------------------------------------------------------
#: Rough peak-RSS estimate per job, used ONLY to decide how many jobs may run
#: concurrently. Measured peaks are recorded per job and are what the report
#: uses; these numbers just keep the parallel scheduler from thrashing.
NEIGHBORS = {"r0": 4, "r1": 8, "r2": 16, "r3": 32}

RASTER_DIRECT_BACKENDS = {"cython", "raster_gpu", "raster_fim"}

#: Bytes of resident memory per stored edge, by backend. networkx keeps a dict
#: of dicts per node and is an order of magnitude heavier than the compiled
#: libraries; on a 38 Mcell window at R2 that is the difference between a job
#: that runs and one that swaps the machine to a standstill.
#: networkit calibrated against a measured 12.75 GiB peak for v0.1.0 R2 on the
#: 38 Mcell window (~305 M edges), which the original 20 B/edge guess
#: underestimated by 40 %.
BYTES_PER_EDGE = {
    "networkx": 120,
    "networkit": 40,
    "igraph": 48,
    "rustworkx": 48,
}
DEFAULT_BYTES_PER_EDGE = 48


def estimate_mem_gb(graph_api: str, neighborhood: str, cells: float,
                    n_targets: int = 8) -> float:
    """Estimate peak RSS in GiB for one routing job."""
    base = 1.2  # interpreter + raster window + geopandas
    if graph_api in RASTER_DIRECT_BACKENDS:
        # no edge list: a handful of per-cell arrays
        return base + cells * 24 / 2**30
    # graph libraries materialise the edge list
    deg = NEIGHBORS.get(neighborhood, 8)
    edges = cells * deg / 2
    per_edge = BYTES_PER_EDGE.get(graph_api, DEFAULT_BYTES_PER_EDGE)
    return base + edges * per_edge / 2**30 + cells * 24 / 2**30
