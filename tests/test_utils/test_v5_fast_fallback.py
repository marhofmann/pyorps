"""Fallback of the step-set-specialised V5 relax kernel (GpuSsspSession._kernel_fast).

Solves without predecessors (solve_seeded, field-only solve) run a relax body
generated for the session's step set. Any failure to build it (generate,
compile, size its grid) or a refused launch must fall back to the generic
kernel, and the generic kernel gives the same field bit for bit.
"""
import numpy as np
import pytest

cupy = pytest.importorskip("cupy")

import pyorps.utils.sssp_gpu as g  # noqa: E402
from pyorps.utils.neighborhood import get_neighborhood_steps  # noqa: E402

pytestmark = pytest.mark.skipif(
    not g.GPU_AVAILABLE or not g._check_v5_available(),  # pylint: disable=protected-access
    reason="V5 kernel unavailable")


def _case(k=1, n=160, seed=3):
    r = np.random.default_rng(seed).integers(1, 300, (n, n)).astype(np.uint16)
    r[40:120, 80] = 65535                      # a wall with gaps at both ends
    r[60:90, 20:50] = 0                        # a zero-cost plateau
    steps = np.asarray(get_neighborhood_steps(k, directed=True), dtype=np.int8)
    seeds = np.array([0, r.size - 1, (n // 2) * n + 5])
    labels = np.array([0.0, 7.5, 120.25], dtype=np.float32)
    return r, steps, seeds, labels


@pytest.mark.parametrize("k", [1, 3])
def test_build_failure_falls_back_with_identical_field(monkeypatch, k):
    r, steps, seeds, labels = _case(k)
    with g.GpuSsspSession(r, steps, arena_factor=1.0) as s:
        assert s._kernel_fast is not s._kernel          # specialised kernel in use
        fast = s.solve_seeded(seeds, labels)

    orig = g._v5_kernel_source

    def broken(steps_, float_raster, specialize=True):
        src, variant = orig(steps_, float_raster, specialize)
        src = src.replace("extern \"C\" __global__ __launch_bounds__",
                          "#error forced build failure\nextern \"C\" __global__ __launch_bounds__")
        return src, variant + "-forced-failure"

    monkeypatch.setattr(g, "_v5_kernel_source", broken)
    with g.GpuSsspSession(r, steps, arena_factor=1.0) as s:
        assert s._kernel_fast is s._kernel               # fell back at build time
        assert s._max_blocks_fast == s._max_blocks
        generic = s.solve_seeded(seeds, labels)
    assert np.array_equal(fast.view(np.uint32), generic.view(np.uint32))


class _RefusingKernel:
    """Stands in for a specialised kernel whose launch the driver refuses.
    (An oversized cooperative grid cannot be used for this: CuPy itself
    reduces it to the co-resident maximum, with a warning.)"""
    calls = 0

    def __call__(self, *args, **kwargs):
        _RefusingKernel.calls += 1
        raise RuntimeError("forced launch refusal")


def test_refused_launch_falls_back_with_identical_field():
    r, steps, seeds, labels = _case(1)
    with g.GpuSsspSession(r, steps, arena_factor=1.0) as s:
        fast = s.solve_seeded(seeds, labels)
        s._kernel_fast = _RefusingKernel()
        refused = s.solve_seeded(seeds, labels)          # launch refused -> generic kernel
        assert _RefusingKernel.calls == 1
        assert s._kernel_fast is s._kernel and s._max_blocks_fast == s._max_blocks
        again = s.solve_seeded(seeds, labels)            # stays on the generic kernel
        assert _RefusingKernel.calls == 1
    assert np.array_equal(fast.view(np.uint32), refused.view(np.uint32))
    assert np.array_equal(fast.view(np.uint32), again.view(np.uint32))


def test_missing_cuda_source_means_v5_unavailable(monkeypatch, tmp_path):
    """A missing sssp_async_v5.cu must not raise at import or probe time."""
    monkeypatch.setattr(g, "_ASYNC_SSSP_KERNEL", None)
    monkeypatch.setattr(g, "_v5_available", None)
    monkeypatch.setattr(g, "_load_cuda_source",
                        lambda name: open(tmp_path / name, encoding="utf-8").read())
    assert g._check_v5_available() is False
