"""Window certificates (plan rev. 5, section 3.5, D2) against full fields.

Plan section 6, oracle 4: walls that force an exit, seeds outside ``W``.
On random rasters and windows, against the field drained on the whole
raster:

* ``lower <= full`` everywhere inside the window (the bound is sound);
* ``field >= full`` (a window can only lose walks);
* ``field == full`` wherever the certificate says exact;
* a wall that forces the optimum out of the window is caught.
"""
import numpy as np
import pytest

from pyorps.certify import boundary_cells, certify_path_field, drain
from pyorps.utils.neighborhood import get_neighborhood_steps

TOL = 1e-9


def _steps(k):
    return np.asarray(get_neighborhood_steps(k, directed=True), dtype=np.int8)


def _setup(seed, rows=16, cols=19):
    rng = np.random.default_rng(seed)
    vals = rng.integers(125, 900, size=(rows, cols)).astype(np.uint16)
    if seed % 2:
        vals[rows // 2, 2:cols - 4] = 65535
    inside = np.zeros((rows, cols), dtype=bool)
    r0, c0 = int(rng.integers(0, 4)), int(rng.integers(0, 4))
    inside[r0:r0 + rows - 5, c0:c0 + cols - 6] = True
    if seed % 3 == 0:                              # a non-rectangular window
        inside[r0 + 3:r0 + 6, c0 + 4:c0 + 9] = False
    return rng, vals, inside


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("k", [1, 2])
def test_certificate_is_sound_and_exact_where_it_says(seed, k):
    rng, vals, inside = _setup(seed)
    steps = _steps(k)
    n = vals.size
    seeds = rng.choice(n, size=4, replace=False)
    labels = rng.uniform(0.0, 4000.0, size=4)
    full = drain(vals, steps, seeds, labels, engine="python")
    cert = certify_path_field(vals, steps, inside, seeds, labels,
                              engine="python")
    fin = inside & np.isfinite(full)
    assert np.all(cert.lower[fin] <= full[fin] + TOL * full[fin])
    ok = inside & np.isfinite(cert.field)
    assert np.all(cert.field[ok] >= full[ok] - TOL * full[ok])
    ex = cert.exact & np.isfinite(full)
    np.testing.assert_allclose(cert.field[ex], full[ex], rtol=1e-12)
    assert 0.0 <= cert.exact_fraction <= 1.0


def test_a_wall_that_forces_an_exit_is_not_certified():
    """Source left of a wall with one gap, the gap outside the window: the
    windowed field is too high right of the wall, and the certificate must
    not call those cells exact."""
    vals = np.full((9, 15), 200, dtype=np.uint16)
    vals[:, 7] = 65535
    vals[0, 7] = 200                     # the only gap, in row 0
    inside = np.zeros_like(vals, dtype=bool)
    inside[2:, :] = True                 # the window excludes the gap row
    steps = _steps(1)
    src = np.array([4 * 15 + 2])
    full = drain(vals, steps, src, [0.0], engine="python")
    cert = certify_path_field(vals, steps, inside, src, [0.0],
                              engine="python")
    right = inside.copy()
    right[:, :8] = False
    assert np.all(~np.isfinite(cert.field[right]))       # unreachable in W
    assert not cert.exact[right].any()
    assert np.all(cert.lower[right] <= full[right])


def test_boundary_cells_of_a_rectangle():
    inside = np.zeros((8, 8), dtype=bool)
    inside[2:6, 2:6] = True
    dW = boundary_cells(inside, _steps(1))
    ring = inside.copy()
    ring[3:5, 3:5] = False
    np.testing.assert_array_equal(dW, ring)
