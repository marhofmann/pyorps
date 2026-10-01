"""The whole free-siting chain, on one small synthetic study.

This is the "real consumer and real fixture" the free-siting plan's
section 4 asked for. The unit tests check each piece in isolation; this
one checks that they compose into the workflow the plan describes, and
that the properties survive the composition:

    cost raster
      -> PathFinder.cost_field                (the cable leg)
      -> save(codec="fixed-quantum")          (the cache)
      -> SavedCostField intervals             (lower / upper, by name)
      -> footprint screen                     (where a facility fits)
      -> candidate lattice                    (+ the stride's Lipschitz gap)
      -> tower_field_bounds                   (the overhead leg, LB and UB)
      -> SitingExport                         (what the MILP reads)

Deliberately NOT here: any selection among the candidates. That is a
third-party MILP's job and is out of PYORPS' scope by decision -- which
is what retired the chain fold, the ``a = b`` collapse and the seeded
multi-source settle.
"""

import os

import numpy as np
import pytest

from pyorps.core.infrastructure_profile import InfrastructureProfile
from pyorps.graph.path_finder import PathFinder
from pyorps.graph.search_session import SavedCostField, full_window_buffer_m
from pyorps.graph.tower_field import tower_field_bounds
from pyorps.raster.handler import create_test_tiff
from pyorps.siting import (
    Footprint,
    candidate_lattice,
    export_tower_fields,
    lipschitz_stride_gap,
    local_lipschitz_gap,
    min_tower_cost_eur,
    overhead_screen_lower_bound,
    sample_field,
    screen_footprints,
)
from pyorps.siting.export import LinkCosts
from pyorps.utils.directional import primitive_directions

pytest.importorskip("rasterio")
pytest.importorskip("scipy")

SIZE = 96
CELL = 10.0


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    """A 96 x 96 raster at 10 m, a terminal, and the profile."""
    from rasterio.transform import from_origin

    d = tmp_path_factory.mktemp("siting")
    path = str(d / "cost.tif")
    transform = from_origin(500000.0, 5600000.0, CELL, CELL)
    create_test_tiff(path, width=SIZE, height=SIZE, transform=transform,
                     pattern="gradient")
    import rasterio
    with rasterio.open(path) as src:
        raster = src.read(1)
        transform = src.transform
        crs = src.crs
    raster = np.clip(raster, 1, 500).astype(np.uint16)
    raster[30:60, 48] = 65535                      # an exclusion, with a gap
    raster[45, 48] = 40
    with rasterio.open(path, "w", driver="GTiff", height=SIZE, width=SIZE,
                       count=1, dtype="uint16", crs=crs,
                       transform=transform) as dst:
        dst.write(raster, 1)

    terminal = (transform.c + 5.5 * CELL, transform.f - 48.5 * CELL)
    profile = InfrastructureProfile.load("profiles/overhead_line_110kv.yaml")
    return {"path": path, "raster": raster, "transform": transform,
            "crs": crs, "terminal": terminal, "profile": profile}


@pytest.fixture(scope="module")
def screen(study):
    """Where a 40 x 20 m facility fits, and what it costs there."""
    build_cost = study["raster"].astype(np.float32)
    blocked = (study["raster"] >= 65535).astype(np.float32)
    return screen_footprints(
        build_cost, blocked,
        footprint=Footprint(40.0, 20.0, rotation_step_deg=30.0),
        transform=study["transform"], resolution_m=CELL)


@pytest.fixture(scope="module")
def candidates(screen, study):
    return candidate_lattice(screen, stride_m=50.0, crs=study["crs"])


class TestCableLegThroughTheCodec:
    """A cost field, cached lossily, read back as an interval."""

    def test_saved_interval_brackets_the_live_field(self, study, tmp_path):
        finder = PathFinder(study["path"], study["terminal"],
                            study["terminal"], graph_api="cython",
                            neighborhood_str="r2",
                            search_space_buffer_m=None)
        with finder.cost_field(study["terminal"],
                               algorithm="dijkstra") as field:
            field.settle_all()
            truth = field.field_array(dtype=np.float64)
            exact = field.save(tmp_path / "exact.npz", with_paths=False)
            lossy = field.save(tmp_path / "lossy.npz", with_paths=False,
                               codec="fixed-quantum", error_bound=25.0)

        assert os.path.getsize(lossy) < os.path.getsize(exact)

        with SavedCostField(lossy) as saved:
            assert saved.codec == "fixed-quantum"
            assert saved.error_bound == 25.0
            arr = saved.field_array(dtype=np.float64)
            ok = np.isfinite(truth) & np.isfinite(arr)
            assert ok.sum() > 1000
            under = truth[ok] - arr[ok]
            assert np.all(under >= 0.0)
            assert under.max() <= 25.0 * (1 + 1e-9)

            pts = [(study["terminal"][0] + 300.0,
                    study["terminal"][1] - 200.0)]
            lo = saved.costs_to(pts, bound="lower")
            hi = saved.costs_to(pts, bound="upper")
            assert lo[0] <= hi[0] <= lo[0] + 25.0 * (1 + 1e-6)

    def test_reachability_survives_the_cache(self, study, tmp_path):
        """Unreachable must stay unreachable: a finite filler there makes
        every bound over those cells false."""
        finder = PathFinder(study["path"], study["terminal"],
                            study["terminal"], graph_api="cython",
                            neighborhood_str="r2",
                            search_space_buffer_m=None)
        with finder.cost_field(study["terminal"],
                               algorithm="dijkstra") as field:
            field.settle_all()
            truth = field.field_array(dtype=np.float64)
            dest = field.save(tmp_path / "f.npz", with_paths=False,
                              codec="fixed-quantum", error_bound=5.0)
        with SavedCostField(dest) as saved:
            got = saved.field_array(dtype=np.float64)
        assert np.array_equal(np.isfinite(truth), np.isfinite(got))


class TestFootprintAndCandidates:
    def test_screen_and_lattice_agree_on_their_cells(self, screen,
                                                     candidates):
        assert len(candidates) > 10
        assert screen.feasible[candidates.rows, candidates.cols].all()
        probe = np.arange(screen.shape[0] * screen.shape[1],
                          dtype=np.float64).reshape(screen.shape)
        assert np.array_equal(
            sample_field(probe, screen.transform, candidates.xs,
                         candidates.ys),
            probe[candidates.rows, candidates.cols])

    def test_the_stride_carries_its_own_uncertainty(self, study,
                                                    candidates):
        """A stride certifies the lattice, not the space -- so the gap
        travels with the candidates instead of being forgotten."""
        gap = local_lipschitz_gap(
            study["raster"].astype(np.float64), candidates.rows,
            candidates.cols, stride_m=candidates.stride_m,
            resolution_m=CELL)
        assert np.all(gap > 0)
        traversable = study["raster"][study["raster"] < 65535].max()
        assert np.all(gap <= lipschitz_stride_gap(
            candidates.stride_m, float(traversable)) + 1e-9)


class TestOverheadLeg:
    def test_bounds_nest_and_export(self, study, candidates, tmp_path):
        lower, upper = tower_field_bounds(
            study["raster"], cell_size_m=CELL, profile=study["profile"],
            source_xy=study["terminal"], transform=study["transform"],
            crs=study["crs"], factor=1,
            directions=primitive_directions(2), record_pred=True)

        export = export_tower_fields(candidates, {"T0": (lower, upper)},
                                     with_geometry=True)
        report = export.check()
        assert report["T0::overhead"]["reachable"] > 0
        assert report["T0::overhead"]["gap_mean_eur"] >= 0

        dest = export.write_npz(tmp_path / "candidates.npz")
        data = np.load(dest, allow_pickle=False)
        lb = data["T0::overhead::cost_lb"]
        ub = data["T0::overhead::cost_ub"]
        both = np.isfinite(lb) & np.isfinite(ub)
        assert both.any()
        assert np.all(ub[both] >= lb[both])

    def test_the_cheap_screen_stays_below_the_real_field(self, study,
                                                         candidates):
        """The section 3.3 bound against an actual constrained field.

        Both price the KERNEL quantity -- terminals uncharged -- so they
        are commensurable, and the screen has to sit underneath.
        """
        lower, upper = tower_field_bounds(
            study["raster"], cell_size_m=CELL, profile=study["profile"],
            source_xy=study["terminal"], transform=study["transform"],
            crs=study["crs"], factor=1,
            directions=primitive_directions(2), record_pred=False)

        # The cheapest terrain any route to the candidate could have.
        cheapest = float(study["raster"][study["raster"] < 65535].min())
        dist = np.hypot(candidates.xs - study["terminal"][0],
                        candidates.ys - study["terminal"][1])
        screen_bound = overhead_screen_lower_bound(
            cheapest * dist, dist, profile=study["profile"],
            quantity="kernel", ignore_max_cost=True,
            field_steps=primitive_directions(2),
            constrained_steps=primitive_directions(2))

        field_cost = upper.costs_to(candidates.points)
        ok = np.isfinite(field_cost)
        assert ok.sum() > 5
        assert np.all(screen_bound.values[ok] <= field_cost[ok] + 1e-6)
        assert min_tower_cost_eur(study["profile"]) == 65_000.0

    def test_a_reconstructed_line_is_buildable(self, study):
        """Every span inside the profile, both ends on the line."""
        _lower, upper = tower_field_bounds(
            study["raster"], cell_size_m=CELL, profile=study["profile"],
            source_xy=study["terminal"], transform=study["transform"],
            crs=study["crs"], factor=1,
            directions=primitive_directions(2), record_pred=True)
        reach = np.argwhere(np.isfinite(upper.arrival))
        assert len(reach) > 10
        checked = 0
        for r, c in reach[::max(1, len(reach) // 5)]:
            seq = upper.tower_sequence(int(r), int(c))
            if len(seq) < 2:
                continue
            checked += 1
            assert (seq[0].row, seq[0].col) == upper.source
            assert (seq[-1].row, seq[-1].col) == (int(r), int(c))
            spans = [t.span_to_next_m for t in seq[:-1]]
            assert all(s < study["profile"].max_span_m + 1e-9 for s in spans)
            assert all(s >= study["profile"].min_span_m - 1e-9
                       for s in spans[:-1])
            line = upper.route_geometry(int(r), int(c))
            assert line is not None and line.length > 0
        assert checked >= 3


class TestTheExportContract:
    def test_a_cable_and_an_overhead_block_coexist(self, study, candidates,
                                                   tmp_path):
        """Per-candidate costs per terminal per link type -- no pairing,
        no selection, nothing composed."""
        lower, upper = tower_field_bounds(
            study["raster"], cell_size_m=CELL, profile=study["profile"],
            source_xy=study["terminal"], transform=study["transform"],
            crs=study["crs"], factor=1,
            directions=primitive_directions(2), record_pred=True)
        export = export_tower_fields(candidates, {"T0": (lower, upper)},
                                     with_geometry=True)

        # search_space_buffer_m=None sizes a window for ONE pair, which
        # is the wrong shape for candidates scattered over the raster.
        finder = PathFinder(
            study["path"], study["terminal"], study["terminal"],
            graph_api="cython", neighborhood_str="r2",
            search_space_buffer_m=full_window_buffer_m(study["path"]))
        with finder.cost_field(study["terminal"]) as field:
            cable = field.costs_to(candidates.points)
        export.add_link(LinkCosts("T0", "cable", cost_lb=cable,
                                  cost_ub=cable))
        assert set(export.links) == {"T0::overhead", "T0::cable"}
        export.check()
        dest = export.write_npz(tmp_path / "both.npz")
        data = np.load(dest, allow_pickle=False)
        assert "T0::cable::cost_lb" in data.files
        assert "T0::overhead::n_towers" in data.files
        assert "T0::cable::n_towers" not in data.files
