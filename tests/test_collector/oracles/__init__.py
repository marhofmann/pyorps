"""Oracles for the trench-sharing collector engine (plan rev. 5, Phase D4).

Frozen ports of the 2026-09-23 prototypes in
``docs/superpowers/prototypes/2026-09-23-trench-sharing-csdw/``: two
independent exact DPs for model D_share (design A, set-valued labels;
design B, integer flows), two exhaustive enumerators of D_share (``a_brute``,
``b_brute``), the verifier's physical-model oracle (``v_strict``: one trench
per used raster step, cycles allowed) and a strict-model MILP
(``b_strict_milp``, HiGHS through scipy).

Plan section 13 asks for these to be ported and to pass on their own before
any engine code is written; ``tests/test_collector/test_oracles_*.py`` are
those checks. Nothing here imports pyorps.
"""
