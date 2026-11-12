# SPDX-License-Identifier: Apache-2.0
"""Timing rules of the TTPU. Each documented bound is tested twice: at the bound (must pass) and one
clock inside it (must fail), so the tests are known to be sensitive to the rule they claim to check."""
import cocotb

import ttpu_model as m
from harness import (CFG, QUICK, check, fresh, play, rand_xv, rng, run_prog, save_result)


def _flip_tiles(n, d):
    """Two tiles whose products differ everywhere: +1 weights then -1 weights, same maximal digits."""
    big = m.digit_range(d)
    return [([[1] * n for _ in range(n)], [big] * n, d),
            ([[-1] * n for _ in range(n)], [big] * n, d)]


# ------------------------------------------------------------------------------------------------
# weights vs. the wave of the previous tile (lower bound of a W8 command)
# ------------------------------------------------------------------------------------------------
async def _wgt_margin_case(dut, margin, d):
    """Tile 1 (+1 weights) is sent back to back (its W8 commands, then its planes), so that the bus is free the moment its last
    plane is out and the next tile's W8 commands are limited by the wave rule alone; tile 2 (-1 weights) is placed by the
    scheduler with the rule shifted by `margin` clocks."""
    n = CFG.n
    big = m.digit_range(d)
    p = m.make_prog(CFG)
    p.wrst()
    (w1, x1, _), (w2, x2, _) = _flip_tiles(n, d)
    p.load_tile(w1)
    last1 = p.vector(x1, d, clear=True)
    ws, xs = m.schedule_tile(p, CFG, d, wmargin=margin)
    p.sched_w = ws
    for grp, t in zip(m.pack_tile(w2, n), ws):
        p.w8(grp, t)
    planes = [m.plane_word([m.to_digits(v, d)[k] for v in x2]) for k in range(d)]
    for k, (word, t) in enumerate(zip(planes, xs)):
        p.plane(word, m.CMD_XPF if k == 0 else m.CMD_XP, t)
    acc = m.block_result([(w1, x1), (w2, x2)], CFG)
    m.read_block(p, CFG, xs[-1], acc, tau=None, label=f"wgt_margin{margin}")
    return p, await run_prog(dut, p)


@cocotb.test()
async def test_wgt_overlap_bound(dut):
    """A W8 command may be issued from last_plane + delta + 1 + rcmax(group) (the previous wave has left the group's PEs);
    one clock earlier the weights change under the wave and corrupt the result. D = 1, 2 and 4 digit planes."""
    await fresh(dut)
    for d in ((1, 2) if QUICK else (1, 2, 4)):
        p, out = await _wgt_margin_case(dut, 0, d)
        check(p, out, f"(at the bound, D={d})")
        ws0 = p.sched_w
        p, out = await _wgt_margin_case(dut, -1, d)
        if p.sched_w == ws0:           # (the schedule did not move, so there is nothing to check here; the per-group test
            continue                   # shows that each bound is tight)
        try:
            check(p, out)
        except AssertionError:
            continue
        raise AssertionError(f"issuing a W8 command one clock early did NOT corrupt the result (D={d}): the bound is not tight")


def early_groups(j, margin):
    """Number of groups before group j that cannot follow the plane: group j sits at its bound plus `margin`, the groups before it in the clocks
    just before, and a clock at or before the plane's is not available (one command per clock). Those groups rewrite weights that tile 1 already holds, so they are written in a gap
    in front of the plane instead; that changes nothing for the group under test."""
    return max(0, j - (CFG.w_lower(0, j) + margin) + 1)


async def _wgt_group_case(dut, j, margin):
    """The lower bound of ONE group. Tile 1 (+1 weights) has one plane in clock X. Tile 2 has +1 weights everywhere except
    the PEs of group j (-1). The groups before j are written in the clocks just before group j and rewrite the weights
    they already hold, so issuing them early changes nothing; group j is issued at its bound plus `margin`; the groups after
    j follow one per clock (also rewriting what they hold). Only group j can corrupt the result, and only if its bound is too
    late by `margin` clocks."""
    n = CFG.n
    big = m.digit_range(1)
    p = m.make_prog(CFG)
    p.wrst()
    w1 = [[1] * n for _ in range(n)]
    x = [big] * n
    p.load_tile(w1)
    extra = early_groups(j, margin)
    t_gap = p.tc
    p.idle_until(t_gap + extra)
    last1 = p.vector(x, 1, clear=True)
    w2 = [[(-1 if m.wpos(n, r, c) // m.GS == j else 1) for c in range(n)] for r in range(n)]
    tj = CFG.w_lower(last1, j) + margin
    ts = [t_gap + g if g < extra else (tj - (j - g) if g <= j else tj + (g - j)) for g in range(CFG.ng)]
    assert extra >= j + 1 or ts[extra] > last1, "the groups before j would have to share a clock with the plane"
    for grp, t in zip(m.pack_tile(w2, n), ts):
        p.w8(grp, t)
    tp = ts[-1] + 1
    p.plane(m.plane_word([m.to_digits(v, 1)[0] for v in x]), m.CMD_XPF, tp)
    acc = m.block_result([(w1, x), (w2, x)], CFG)
    m.read_block(p, CFG, tp, acc, tau=None, label=f"group{j}_margin{margin}")
    return p, await run_prog(dut, p)


@cocotb.test()
async def test_wgt_overlap_each_group(dut):
    """The lower bound of every single group is tight: group j at `last_plane + delta + 1 + rcmax(j)` gives the exact
    result, one clock earlier it corrupts it. (test_wgt_overlap_bound shifts all groups together; the scheduler leaves
    some of them slack, so it does not show this per group.)"""
    await fresh(dut)
    for j in range(CFG.ng):
        p, out = await _wgt_group_case(dut, j, 0)
        check(p, out, f"(group {j} at its bound)")
        p, out = await _wgt_group_case(dut, j, -1)
        try:
            check(p, out)
        except AssertionError:
            continue
        raise AssertionError(f"group {j}: a W8 command one clock before its bound did NOT corrupt the result: rcmax[{j}] = {CFG.rcmax[j]} is not tight")


# ------------------------------------------------------------------------------------------------
# weights vs. the first plane of the same tile (upper bound of a W8 command)
# ------------------------------------------------------------------------------------------------
async def _wgt_late_case(dut, j, late):
    """Tile A (+1 weights), then tile B (-1 weights) written by hand: groups 0..j-1, the plane, group j at its latest
    allowed cycle (plus `late`), then the remaining groups. The previous wave is long gone."""
    n = CFG.n
    big = m.digit_range(1)
    p = m.make_prog(CFG)
    p.wrst()
    wa = [[1] * n for _ in range(n)]
    wb = [[-1] * n for _ in range(n)]
    x = [big] * n
    last_a, acc_a = m.plan_block(p, CFG, [(wa, x, 1)])
    m.read_block(p, CFG, last_a, acc_a, tau=None, label="A")
    s = max(p.tc, p.hazard_start + CFG.ng, last_a + 40)
    groups = m.pack_tile(wb, n)
    t = s
    for g in range(j):
        p.w8(groups[g], t)
        t += 1
    b0 = t
    p.plane(m.plane_word([m.to_digits(v, 1)[0] for v in x]), m.CMD_XPS, b0)
    aj = b0 + CFG.w_upper(0, j) + late                 # w_upper(first_plane=0, j) = delta + rcmin[j]
    assert aj > b0, "group j would have to share the cycle of the plane"
    t = aj
    for g in range(j, CFG.ng):
        p.w8(groups[g], t)
        t += 1
    acc_b = m.block_result([(wb, x)], CFG)
    m.read_block(p, CFG, b0, acc_b, tau=None, label=f"B(j={j},late={late})")
    return p, await run_prog(dut, p)


@cocotb.test()
async def test_wgt_upper_bound(dut):
    """The W8 command of group j must be issued by first_plane + delta + rcmin(group j): later, the first plane reaches
    the group's first PE with the old weight. Every group whose bound lies after the plane is tested at the bound and one clock late."""
    await fresh(dut)
    tested = 0
    for j in range(1, CFG.ng):
        if CFG.w_upper(0, j) <= 0:
            continue
        p, out = await _wgt_late_case(dut, j, 0)
        check(p, out, f"(group {j} at its latest cycle)")
        p, out = await _wgt_late_case(dut, j, 1)
        try:
            check(p, out)
        except AssertionError:
            tested += 1
            continue
        raise AssertionError(f"group {j} one clock later than the bound did NOT corrupt the result: the bound is not tight")
    assert tested >= 1


# ------------------------------------------------------------------------------------------------
# block boundary: the next block's clear vs. the raw bytes of the previous block
# ------------------------------------------------------------------------------------------------
async def _clear_margin_case(dut, margin):
    n = CFG.n
    p = m.make_prog(CFG)
    p.wrst()
    for blk in range(2):
        sgn = 1 if blk == 0 else -1                  # block 0 sums to +n, block 1 to -n: any late read differs
        w = [[1] * n for _ in range(n)]
        x = [sgn * m.digit_range(1)] * n
        last, acc = m.plan_block(p, CFG, [(w, x, 1)])
        m.read_block(p, CFG, last, acc, tau=None, label=f"clr{margin}.b{blk}", clear_margin=margin)
    return p, await run_prog(dut, p)


@cocotb.test()
async def test_block_clear_bound(dut):
    """The next block's XPS may clear column c only after column c was read; one clock earlier loses data."""
    await fresh(dut)
    p, out = await _clear_margin_case(dut, 0)
    check(p, out, "(at the bound)")
    p, out = await _clear_margin_case(dut, -1)
    try:
        check(p, out)
    except AssertionError:
        return
    raise AssertionError("starting the next block one clock early did NOT lose data: the bound is not tight")


# ------------------------------------------------------------------------------------------------
# READY and the settle time of the raw bytes
# ------------------------------------------------------------------------------------------------
@cocotb.test()
async def test_ready_timing(dut):
    """READY is low from the clock after the last plane until exactly t0 + 1 + TSETTLE (+4 read latency),
    for one-, two- and four-digit vectors."""
    await fresh(dut)
    r = rng(9)
    p = m.make_prog(CFG)
    p.wrst()
    for d in (1, 2, 4):
        p.idle_until(p.read_end + 5)
        last = p.vector(rand_xv(CFG.n, d, r), d, clear=True)
        for dt in range(1, CFG.tsettle + 6):
            want = m.ST_READY if dt >= 1 + CFG.tsettle else 0
            p.read_status(last + dt, want, f"ready@t0+{dt} (D={d})", mask=m.ST_READY)
    await play(dut, p)


@cocotb.test()
async def test_settle_margin(dut):
    """Scan one byte (low or high) of column 0 and of the last column every clock after the final accumulate: the carry-save pair
    must be binary no later than the documented raw time (high byte t0 + rawbase + c, low byte (accw - 8) / seg clocks earlier) and that
    time must be tight, for random and ripple-heavy data. READY (TSETTLE) comes later still."""
    await fresh(dut)
    n = CFG.n
    r = rng(10)
    trials = 36 if QUICK else 150
    dt0 = CFG.rawbase - 12 - CFG.lo_early
    span = CFG.tsettle + 4 - dt0
    assert dt0 > 0
    p = m.make_prog(CFG)
    p.wrst()
    scans = []
    for trial in range(trials):
        d = r.choice((1, 2, 3, 4))
        mode = r.choice(("rand", "ones", "alt"))
        tiles = []
        if trial < 4:
            # the longest possible ripple: the accumulator of the scanned column goes 0 -> -1 (all ones) -> 0, so that the last add
            # turns every bit and the carry has to run through the whole word
            col = (0, n - 1)[trial % 2]
            for sgn in (-1, 1):
                w = [[0] * n for _ in range(n)]
                w[0][col] = sgn
                tiles.append((w, [1] + [0] * (n - 1), 1))
            d = 1
        else:
            for _ in range(r.randint(1, 3)):
                if mode == "rand":
                    w = m.rand_tile(n, r)
                elif mode == "ones":
                    w = [[1] * n for _ in range(n)]
                else:
                    w = [[(1 if (i + j) % 2 else -1) for j in range(n)] for i in range(n)]
                tiles.append((w, rand_xv(n, d, r, extreme=0.5), d))
            col = (0, n - 1)[trial % 2]
        hi_byte = (trial // 2) % 2 == 0 if trial < 4 else (trial % 4 < 2)
        last, acc = m.plan_block(p, CFG, tiles)
        v = acc[col] & ((1 << CFG.accw) - 1)
        e0 = len(p.exp)
        start = max(dt0, p.tc - last)               # (the last commands of the tile can come after its last plane: no address in those clocks)
        for dt in range(start, dt0 + span):
            if trial < 4:                          # the deterministic trials read the same byte every clock: (low, high) x (column 0, last)
                lo_now = not hi_byte
            else:
                lo_now = dt % 2 == 0
            if lo_now:
                p.read(last + dt, m.ADDR_RAW0 + col, v & 0xFF, f"scan{trial}.lo@{dt}")
            else:
                p.read(last + dt, m.ADDR_RAW0 + n + col, v >> 8, f"scan{trial}.hi@{dt}")
        scans.append((e0, col, trial < 4, hi_byte, start, dt0 + span - start))
        # keep the next block's clear away from the scan window
        p.hazard_start = max(p.hazard_start, min(last + dt0 + span + 2 - 4, last + dt0 + span) - CFG.ng)
    out = await run_prog(dut, p)
    worst, hist = {}, {}
    for e0, col, det, hi_byte, start, nrd in scans:
        ok = [out[e.cycle] is not None and out[e.cycle] == e.value for e in p.exp[e0:e0 + nrd]]
        k = len(ok)
        while k > 0 and ok[k - 1]:
            k -= 1
        first_good = start + k        # first address offset from which every later read is correct
        key = (col, "hi" if hi_byte else "lo") if det else (col, "mixed")
        worst[key] = max(worst.get(key, 0), first_good)
        h = hist.setdefault(f"{key[0]}.{key[1]}", {})
        h[first_good] = h.get(first_good, 0) + 1
    ready_dt = 1 + CFG.tsettle
    lo_off = CFG.lo_early
    bound = {(c, "hi"): CFG.rawbase + CFG.cskew(c) for c in (0, n - 1)}
    bound.update({(c, "lo"): CFG.rawbase - lo_off + CFG.cskew(c) for c in (0, n - 1)})
    bound.update({(c, "mixed"): CFG.rawbase + CFG.cskew(c) for c in (0, n - 1)})
    dut._log.info(f"settle scan: worst first-good offset {worst}; documented raw bounds {bound}; READY at t0+{ready_dt}; hist {hist}")
    save_result("settle_hist", hist)
    save_result("settle_worst", {f"{k[0]}.{k[1]}": v for k, v in worst.items()})
    save_result("settle_ready_dt", ready_dt)
    save_result("settle_rawbase", CFG.rawbase)
    for key, w_ in worst.items():
        assert w_ <= bound[key], f"{key} settled at t0+{w_}, after the documented raw bound t0+{bound[key]}"
        if key[1] != "mixed":          # the deterministic worst-case trials show the bound is exact
            assert w_ >= bound[key], f"{key}: the raw bound t0+{bound[key]} is loose (worst observed t0+{w_})"
    assert max(worst.values()) <= ready_dt
