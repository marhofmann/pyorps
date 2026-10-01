"""Compiled voltage check for the batched C5 check (one fused kernel).

Per design, in one parallel iteration: the radial tree is oriented by a
breadth-first search from the UW slot, the backward--forward sweep of
:func:`pyorps.collector.voltage.load_flow` runs at every corner (same
arithmetic, same order), and the turbine LV voltage of
:func:`pyorps.collector.voltage.evaluate_limits` is evaluated at every tap.
Only the tap decision stays in Python (``decide_taps``), so every backend
applies one rule. No ``fastmath``: the check enters a certificate, so IEEE
arithmetic is kept.

Error codes per design: 0 ok, 1 the branches form a cycle, 2 a turbine is
not connected, 3 a branch is not connected to the UW.
"""

import numba as nb
import numpy as np


@nb.njit(cache=True, parallel=True)
def solve_batch(ptr, eu, ev, zser, ysh, S, n, s_slot, v0, tol, max_iter,
                zt, ratio, lo, hi, out_v, out_conv, out_it, out_vmax,
                out_margin, out_err):
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    """Solve and evaluate every design (see the module docstring).

    Parameters:
        ptr: ``(B + 1,)`` design ``b`` owns branches ``ptr[b]:ptr[b+1]``.
        eu, ev: end slots of every branch.
        zser: series impedance (p.u.); ysh: total shunt admittance (p.u.,
            half at each end).
        S, n: slots, turbines (turbine ``t`` is slot ``1 + t``).
        s_slot: ``(C, S)`` injections (p.u.); v0: ``(C,)`` busbar voltage.
        tol, max_iter: As :func:`~pyorps.collector.voltage.load_flow`.
        zt: ``(K,)`` turbine transformer impedance per tap (p.u.);
            ratio: ``(K,)`` ``u_kv / tap_kv``; lo, hi: the LV band.
        out_v: ``(C, B, S)`` voltages by slot (the busbar voltage on unused
            slots); out_conv, out_it: ``(C, B)``; out_vmax: ``(C, B)`` the
            highest ``|V|`` over used slots; out_margin: ``(B, n, K)`` band
            margin, min over corners; out_err: ``(B,)`` error code.
    """
    # lizard forgives: inherent complexity of this numerical routine; behaviour is pinned by the test suite
    B = ptr.shape[0] - 1
    C = s_slot.shape[0]
    K = zt.shape[0]
    for b in nb.prange(B):
        e0 = ptr[b]
        m = ptr[b + 1] - e0
        # ---- adjacency and breadth-first orientation from slot 0
        deg = np.zeros(S + 1, dtype=np.int64)
        for i in range(m):
            deg[eu[e0 + i] + 1] += 1
            deg[ev[e0 + i] + 1] += 1
        for k in range(S):
            deg[k + 1] += deg[k]
        fill = deg[:S].copy()
        nbr = np.empty(2 * m, dtype=np.int64)
        via = np.empty(2 * m, dtype=np.int64)
        for i in range(m):
            a = eu[e0 + i]
            c = ev[e0 + i]
            nbr[fill[a]] = c
            via[fill[a]] = i
            fill[a] += 1
            nbr[fill[c]] = a
            via[fill[c]] = i
            fill[c] += 1
        order = np.empty(S, dtype=np.int64)
        parent = np.full(S, -1, dtype=np.int64)
        seen = np.zeros(S, dtype=np.bool_)
        z = np.zeros(S, dtype=np.complex128)
        y = np.zeros(S, dtype=np.complex128)
        order[0] = 0
        seen[0] = True
        head = 0
        tail = 1
        err = 0
        while head < tail:
            u = order[head]
            head += 1
            for q in range(deg[u], deg[u + 1]):
                w = nbr[q]
                i = via[q]
                if seen[w]:
                    if w != parent[u]:
                        err = 1                      # a second path: cycle
                    continue
                seen[w] = True
                parent[w] = u
                z[w] = zser[e0 + i]
                order[tail] = w
                tail += 1
        if err == 0 and tail - 1 != m:
            err = 3 if tail - 1 < m else 1
        for t in range(n):
            if not seen[1 + t]:
                err = 2
        out_err[b] = err
        if err != 0:
            for c in range(C):
                out_conv[c, b] = False
                out_it[c, b] = 0
            continue
        for i in range(m):
            half = 0.5 * ysh[e0 + i]
            y[eu[e0 + i]] += half
            y[ev[e0 + i]] += half
        used = tail
        # ---- sweep at every corner
        V = np.empty(S, dtype=np.complex128)
        Vn = np.empty(S, dtype=np.complex128)
        J = np.zeros(S, dtype=np.complex128)
        acc = np.zeros(S, dtype=np.complex128)
        for t in range(n):
            for k in range(K):
                out_margin[b, t, k] = np.inf
        for c in range(C):
            for k in range(S):
                V[k] = v0[c]
            conv = False
            it = 0
            while it < max_iter:
                it += 1
                for k in range(S):
                    acc[k] = 0.0
                for j in range(used - 1, 0, -1):
                    w = order[j]
                    jk = ((s_slot[c, w] / V[w]).conjugate() - y[w] * V[w]
                          + acc[w])
                    J[w] = jk
                    acc[parent[w]] += jk
                Vn[0] = v0[c]
                delta = 0.0
                finite = True
                for j in range(1, used):
                    w = order[j]
                    Vn[w] = Vn[parent[w]] + z[w] * J[w]
                    d = abs(Vn[w] - V[w])
                    if not np.isfinite(d):
                        finite = False
                    if d > delta:  # pylint: disable=consider-using-max-builtin  # numba kernel, left untouched
                        delta = d
                for j in range(1, used):
                    w = order[j]
                    V[w] = Vn[w]
                if not finite:
                    break
                if delta <= tol:
                    conv = True
                    break
            out_conv[c, b] = conv
            out_it[c, b] = it
            vmax = 0.0
            for k in range(S):
                out_v[c, b, k] = V[k]
            for j in range(used):
                a = abs(V[order[j]])
                if a > vmax:  # pylint: disable=consider-using-max-builtin  # numba kernel, left untouched
                    vmax = a
            out_vmax[c, b] = vmax
            # ---- turbine LV voltage at every tap (evaluate_limits)
            for t in range(n):
                vt = V[1 + t]
                st = s_slot[c, 1 + t]
                cur = 0.0j
                if st != 0:
                    cur = (st / vt).conjugate()
                for k in range(K):
                    vl = abs(vt + zt[k] * cur) * ratio[k]
                    mg = min(vl - lo, hi - vl)
                    if mg < out_margin[b, t, k]:
                        out_margin[b, t, k] = mg
