"""Plan A5: cap = min(9 GB, free - 3 GB), P = floor(cap / peak), P <= 2."""
import pytest

from pyorps.utils.resources import GIB, resource_budget


def test_cap_is_the_smaller_of_the_ceiling_and_free_minus_reserve():
    b = resource_budget(2 * GIB, free_bytes=40 * GIB)
    assert b.cap_bytes == 9 * GIB
    assert b.processes == 2                     # 4 would fit, 2 is the limit
    b = resource_budget(2 * GIB, free_bytes=8 * GIB)
    assert b.cap_bytes == 5 * GIB
    assert b.processes == 2
    b = resource_budget(3 * GIB, free_bytes=8 * GIB)
    assert b.processes == 1


def test_no_room_for_one_worker_raises():
    with pytest.raises(MemoryError, match="shrink"):
        resource_budget(6 * GIB, free_bytes=8 * GIB)
    with pytest.raises(MemoryError):
        resource_budget(1 * GIB, free_bytes=2 * GIB)   # free < reserve


def test_record_is_plain_json():
    import json
    b = resource_budget(GIB, free_bytes=20 * GIB)
    assert json.loads(json.dumps(b.as_record()))["processes"] == 2


def test_bad_arguments():
    with pytest.raises(ValueError):
        resource_budget(0, free_bytes=20 * GIB)
    with pytest.raises(ValueError):
        resource_budget(GIB, free_bytes=20 * GIB, max_processes=0)


def test_reads_the_os_when_not_told():
    pytest.importorskip("psutil")
    b = resource_budget(1, max_gb=0.001, reserve_gb=0.0)
    assert b.processes >= 1 and b.free_at_start_bytes > 0
