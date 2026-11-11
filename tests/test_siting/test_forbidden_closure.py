"""Routes must never traverse forbidden land-use features (owner's requirement, 2026-10-05).

The cost map is closed: every cell a forbidden polygon touches is forbidden. These tests check the
chord helper, then the guarantee on a synthetic raster with a THIN forbidden sliver (the plain
pixel-centre burn leaves holes in it, so routes slip through; the closed burn does not), and on a small
window of the real cost map when the data is on disk.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import Polygon

CASE = Path(__file__).resolve().parents[2] / "case_studies" / "runkel_free_siting"
MANIFEST = CASE / "data" / "data_manifest.yaml"

if not (CASE / "scripts" / "verify_forbidden_closure.py").exists():
    # the wheel-test job runs the tests without the repository around them
    pytest.skip("case-study files not present", allow_module_level=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, CASE / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.path.insert(0, str(CASE / "scripts"))
    spec.loader.exec_module(mod)
    return mod


vf = _load("verify_forbidden_closure")
rec = _load("cost_recipe")


def test_chord_cells_are_the_positive_length_cells():
    assert vf.chord_cells(0, 0, 0, 3) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    # a diagonal touches the corner cells (1,0) and (0,1) only at a point: not chord cells
    assert vf.chord_cells(0, 0, 1, 1) == [(0, 0), (1, 1)]
    # (1,3): exact chords in (0,0), (0,1), (1,2), (1,3); the corner cells (1,1), (0,2) are not crossed
    assert sorted(vf.chord_cells(0, 0, 1, 3)) == [(0, 0), (0, 1), (1, 2), (1, 3)]


def test_class_cost_and_forbidden_rows_follow_the_recipe():
    import pandas as pd

    assert rec.class_cost("Landwirtschaft", "Ackerland") == 437
    assert rec.class_cost("Landwirtschaft", None) == 437            # empty sub-class: the default
    assert rec.class_cost("Wohnbaufläche", None) == rec.FORBIDDEN
    assert rec.class_cost("a class the recipe lacks", None) == rec.FORBIDDEN
    frame = pd.DataFrame({"nutzart": ["Wald", "Wohnbaufläche"], "bez": ["Laubholz", None]})
    assert rec.forbidden_mask(frame).tolist() == [False, True]


def _thin_world(size=120):
    """A passable raster with a thin diagonal forbidden sliver (0.4 m wide) across it."""
    tr = from_origin(0.0, float(size), 1.0, 1.0)
    sliver = Polygon([(10, 5), (11, 5), (size - 5 + 0.4, size - 10), (size - 5, size - 10)]).buffer(0.2)
    return tr, sliver


def _burn(tr, geom, size, all_touched):
    mask = features.rasterize([(geom, 1)], out_shape=(size, size), transform=tr, fill=0,
                              dtype="uint8", all_touched=all_touched).astype(bool)
    values = np.full((size, size), 100, dtype=np.uint16)
    values[mask] = rec.FORBIDDEN
    return values


def _violations(values, tr, geom, seed=3):
    rng = np.random.default_rng(seed)
    return vf.check_window(values, tr, geom, n_seeds=4, n_targets=40, rng=rng)


def test_the_plain_burn_lets_routes_through_a_thin_sliver_and_the_closed_burn_does_not():
    size = 120
    tr, sliver = _thin_world(size)
    plain = _burn(tr, sliver, size, all_touched=False)
    closed = _burn(tr, sliver, size, all_touched=True)
    # every cell the sliver touches is forbidden in the closed burn, not in the plain one
    touched = features.rasterize([(sliver, 1)], out_shape=(size, size), transform=tr, fill=0,
                                 dtype="uint8", all_touched=True).astype(bool)
    assert (closed[touched] == rec.FORBIDDEN).all()
    assert (plain[touched] != rec.FORBIDDEN).any()
    bad = _violations(plain, tr, sliver)
    ok = _violations(closed, tr, sliver)
    assert bad["vector_violations"] > 0            # negative control: the check can see a crossing
    assert ok["vector_violations"] == 0 and ok["cell_violations"] == 0
    assert ok["routes"] >= 100 and ok["steps"] > 1000


def test_a_barrier_made_of_corner_touching_polygons_is_closed_too():
    size = 80
    tr = from_origin(0.0, float(size), 1.0, 1.0)
    # staircase of unit squares touching only at corners, running across the window
    squares = [Polygon([(i, i), (i + 1, i), (i + 1, i + 1), (i, i + 1)]).buffer(-0.05)
               for i in range(5, size - 5)]
    from shapely.ops import unary_union
    geom = unary_union(squares)
    closed = _burn(tr, geom, size, all_touched=True)
    r = vf.check_window(closed, tr, geom, n_seeds=3, n_targets=40, rng=np.random.default_rng(5))
    assert r["vector_violations"] == 0 and r["cell_violations"] == 0


@pytest.mark.skipif(not MANIFEST.exists(), reason="case-study manifest not present")
def test_the_registered_closed_cost_map_has_no_forbidden_traversal_on_a_real_window():
    pytest.importorskip("yaml")
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window

    from pyorps.siting.data_manifest import DataManifest

    m = DataManifest.load(MANIFEST)
    if "cost_base_1m_closed" not in m.layers or m.layer("cost_base_1m_closed").status != "present":
        pytest.skip("closed cost map not built")
    path, alk = m.file_of("cost_base_1m_closed"), m.file_of("alkis_land_use")
    if not (path.is_file() and alk.is_file()):
        pytest.skip("data files not on disk")
    with rasterio.open(path) as src:
        win = Window(9000, 5000, 700, 700)            # inside the farm area
        values, tr = src.read(1, window=win), src.window_transform(win)
    frame = gpd.read_file(alk)
    xs, ys = tr * (0, 700), tr * (700, 0)
    union = vf.forbidden_union_in(frame, (xs[0], xs[1], ys[0], ys[1]))
    r = vf.check_window(values, tr, union, n_seeds=2, n_targets=25, rng=np.random.default_rng(11))
    assert r["routes"] >= 40
    assert r["cell_violations"] == 0 and r["vector_violations"] == 0
