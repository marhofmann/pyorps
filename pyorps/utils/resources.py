"""Resource budget for the free-siting runs (plan rev. 5, item A5).

The machine is shared: runs must leave room for the user's other work.

    cap = min(max_gb, free RAM at start - reserve_gb)
    P   = floor(cap / measured per-process peak)

``P`` is the number of worker processes; it is 2 on today's machine with
the measured drain peaks. Both numbers are taken ONCE, at the start of a
run, and recorded with its results -- a run that re-reads free memory
mid-way would change its own parallelism as it goes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = ["ResourceBudget", "resource_budget"]

GIB = 1024 ** 3


@dataclass(frozen=True)
class ResourceBudget:
    """The budget of one run, in bytes.

    Attributes:
        cap_bytes: Memory the run may use in total.
        per_process_bytes: Measured peak of one worker.
        processes: ``floor(cap / per_process)``, at least 1 and at most
            ``max_processes``.
        free_at_start_bytes: Free memory when the budget was taken.
    """
    cap_bytes: int
    per_process_bytes: int
    processes: int
    free_at_start_bytes: int

    def as_record(self) -> dict:
        """JSON-able form for a result manifest."""
        return asdict(self)


def _free_bytes() -> int:
    import psutil
    return int(psutil.virtual_memory().available)


def resource_budget(per_process_bytes: int, *, max_gb: float = 9.0,
                    reserve_gb: float = 3.0, max_processes: int = 2,
                    free_bytes: int | None = None) -> ResourceBudget:
    """The plan-A5 budget for workers that each peak at ``per_process_bytes``.

    Parameters:
        per_process_bytes: The measured peak of one worker (measure it on
            a small window and scale; never guess low).
        max_gb: The hard ceiling (9 GB by the user's rule).
        reserve_gb: What stays free for everything else.
        max_processes: Upper limit on ``P`` (2 by the user's rule).
        free_bytes: Free memory now; read from the OS when ``None``.

    Raises:
        MemoryError: not even one worker fits under the cap. Starting
            anyway would push the machine into swap; the caller must
            shrink the window instead.
    """
    if per_process_bytes <= 0:
        raise ValueError("per_process_bytes must be positive")
    if max_processes < 1:
        raise ValueError("max_processes must be at least 1")
    free = _free_bytes() if free_bytes is None else int(free_bytes)
    cap = int(min(max_gb * GIB, free - reserve_gb * GIB))
    processes = min(int(max_processes), cap // int(per_process_bytes)
                    if cap > 0 else 0)
    if processes < 1:
        raise MemoryError(
            f"one worker needs {per_process_bytes / GIB:.2f} GiB but the "
            f"cap is {max(cap, 0) / GIB:.2f} GiB (min({max_gb} GiB, free "
            f"{free / GIB:.2f} GiB - {reserve_gb} GiB reserve)); shrink "
            f"the window")
    return ResourceBudget(cap_bytes=cap,
                          per_process_bytes=int(per_process_bytes),
                          processes=int(processes),
                          free_at_start_bytes=free)
