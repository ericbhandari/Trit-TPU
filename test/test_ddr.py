"""DDR lane tests (cocotb): the second command word per clock, sampled on the falling edge.

On a build without the DDR lane (TTPU_DDR=0) every test is skipped. Each test builds a program with ttpu_model (two command lanes
per clock, expected read data from the integer model), plays it with the DDR pin driver of harness.py (lane 0 changes a quarter
period after each rising edge, lane 1 a quarter period after each falling edge) and compares."""
import cocotb

import ttpu_model as m
from harness import CFG, check, fresh, play, rand_wmat, rand_xv, rng, run_prog
from test_timing import early_groups

DDR_TEST = cocotb.test(skip=not CFG.ddr)


def _gemv(p, r, o, k, d, tau=None, label="y"):
    wm = rand_wmat(o, k, r, r.choice((1.0, 0.6)))
    x = rand_xv(k, d, r, 0.1)
    return m.plan_gemv(CFG, wm, x, d, p=p, tau=tau, label=label)


def _junk_lane0(p, r, lo, hi, codes=range(8)):
    """Random words with the given command codes in every idle lane-0 slot of the cycles lo .. hi - 1."""
    p._ensure(hi)
    for t in range(lo, hi):
        if p.cmd0[t] == m.CMD_NOP:
            p.cmd0[t], p.ui0[t], p.fld0[t] = r.choice(list(codes)), r.getrandbits(8), r.getrandbits(5)
    p.ddr_stim = True


@DDR_TEST
async def test_ddr_gemv(dut):
    """Matrix-vector products in DDR mode for every digit count and every choice of plane lane (both, lane 0 only, lane 1
    only); lane 0 carries random words of every command code before the mode is switched on."""
    await fresh(dut)
    r = rng(31)
    for d in (1, 2, 3, 4):
        for xl in ((1, 0), (0,), (1,)):
            await fresh(dut)                              # a reset: the mode is off again
            tau = r.choice((0, 3, 9))
            p = m.make_prog(CFG)
            p.xlanes = xl
            p.wrst()
            p.cfg_tau(tau)
            p.nop(5)
            p.ddr_enable(True)
            _gemv(p, r, CFG.n * r.randint(1, 2), CFG.n * r.randint(1, 4), d, tau=tau, label=f"ddr_d{d}")
            p.scramble_lane0(r, 4)                        # junk on lane 0 while the chip is still in SDR mode
            await play(dut, p, f"(D={d}, plane lanes {xl})")


@DDR_TEST
async def test_ddr_sdr_ignores_lane0(dut):
    """Outside DDR mode lane 0 is ignored whatever it carries: a complete SDR program with random words on lane 0 in every
    clock (valid W8 and plane codes included) gives the SDR result, also after DDR mode was switched on and off again."""
    await fresh(dut)
    r = rng(32)
    p = m.make_prog(CFG)
    p.wrst()
    p.cfg_tau(1)
    for d in (1, 3):
        _gemv(p, r, CFG.n, 2 * CFG.n, d, tau=1, label=f"sdr_d{d}")
    p.scramble_lane0(r, len(p) + 4)
    await play(dut, p, "(SDR with junk on lane 0)")
    await fresh(dut)
    q = m.make_prog(CFG)
    q.wrst()
    q.cfg_tau(0)
    q.ddr_enable(True)
    _gemv(q, r, CFG.n, 2 * CFG.n, 2, tau=0, label="ddr")
    q.nop(60)
    q.ddr_enable(False)
    q.nop(10)
    n0 = len(q)
    _gemv(q, r, CFG.n, 2 * CFG.n, 3, tau=0, label="sdr_again")
    _junk_lane0(q, r, n0, len(q) + 4)                      # junk on lane 0 in the SDR phase after DDR mode was switched off
    await play(dut, q, "(DDR, then DDR off, then SDR with junk on lane 0)")


@DDR_TEST
async def test_ddr_mode_boundary(dut):
    """The mode WRST takes effect for the pairs from the second clock after it: a lane-0 W8 in the clock after the WRST is
    ignored (the old weights stay), one in the second clock after it is written."""
    await fresh(dut)
    r = rng(33)
    n = CFG.n
    for first in (1, 2):
        await fresh(dut)
        w0 = [[r.choice((-1, 1)) for _ in range(n)] for _ in range(n)]
        g_new = m.pack_group([1] * m.GS)                     # group 0 of the new content: all +1
        p = m.make_prog(CFG)
        p.wrst()
        p.load_tile(w0)
        p.nop(20)
        t = p.emit(m.CMD_WRST, m.DDR_ON)
        p.ddr_on = p.ddr_stim = True
        p.put0(t + first, m.CMD_W8, g_new[0], g_new[1])
        p.idle_until(t + 30)
        wexp = [row[:] for row in w0]
        if first == 2:
            for pos in range(m.GS):
                rr_, cc_ = m.stream_order(n)[pos]
                wexp[rr_][cc_] = 1
        x = [m.digit_range(1)] * n
        tp = p.emit(m.CMD_XPS, *m.plane_word([m.to_digits(v, 1)[0] for v in x]))
        m.read_block(p, CFG, tp, m.block_result([(wexp, x)], CFG), tau=None, label=f"boundary{first}")
        await play(dut, p, f"(lane-0 W8 in the clock {first} after the mode WRST)")


@DDR_TEST
async def test_ddr_lane0_other_commands_ignored(dut):
    """Lane 0 only does anything for W8 and the plane commands: CFG, WRST, RDS and NOP words (with any data and field) on
    lane 0 in DDR mode change nothing: no tau write, no pointer rewind, no read stream, no ERR."""
    await fresh(dut)
    r = rng(34)
    p = m.make_prog(CFG)
    p.wrst()
    p.cfg_tau(5)
    p.nop(5)
    p.ddr_enable(True)
    _gemv(p, r, CFG.n, 3 * CFG.n, 2, tau=5, label="y")
    _junk_lane0(p, r, 4, len(p) + 20, codes=(m.CMD_CFG, m.CMD_WRST, m.CMD_RDS, m.CMD_NOP))
    p.read(max(p.read_end, p.tc) + 3, m.ADDR_TAU, 5, "tau unchanged")
    p.read_status(max(p.read_end, p.tc) + 5, m.ST_DDR, "ddr on, no ERR", mask=m.ST_DDR | m.ST_ERR)
    await play(dut, p)


@DDR_TEST
async def test_ddr_plane_conflict(dut):
    """Two planes in one pair (one on each lane) set ERR; the plane on lane 1 is the one that counts."""
    await fresh(dut)
    n = CFG.n
    w = [[1] * n for _ in range(n)]
    p = m.make_prog(CFG)
    p.wrst()
    p.ddr_enable(True)
    t0 = p.tc
    last = _ddr_load_tile(p, w, t0)
    tp = last + 25
    x1, x0 = [m.digit_range(1)] * n, [-m.digit_range(1)] * n
    p.plane(m.plane_word([m.to_digits(v, 1)[0] for v in x1]), m.CMD_XPS, tp, 1)
    p.put0(tp, m.CMD_XPF, *m.plane_word([m.to_digits(v, 1)[0] for v in x0]))      # a second plane in the same pair
    m.read_block(p, CFG, tp, m.block_result([(w, x1)], CFG), tau=None, label="lane 1 plane counts")
    p.read_status(max(p.read_end, p.tc) + 3, m.ST_ERR | m.ST_DDR, "ERR after two planes in a pair", mask=m.ST_ERR | m.ST_DDR)
    await play(dut, p)


@DDR_TEST
async def test_ddr_lane0_illegal_values(dut):
    """An illegal packed value on lane 0 (byte > 242, field > 26) sets ERR for W8 and for planes, like on lane 1; WRST clears it
    and leaves the mode on."""
    await fresh(dut)
    for what, bd, fd in (("byte", 250, 0), ("field", 0, 27)):
        for cmd in (m.CMD_W8, m.CMD_XPS):
            if what == "field" and cmd == m.CMD_XPS and CFG.n == 4:
                continue                                      # N = 4: the field of a plane is not used
            p = m.make_prog(CFG)
            p.wrst()
            p.ddr_enable(True)
            t = p.tc
            p.read_status(t + 2, m.ST_DDR, "no ERR yet", mask=m.ST_ERR | m.ST_DDR)
            p.put0(t + 4, cmd, bd, fd)
            p.read_status(t + 24, m.ST_ERR | m.ST_DDR, f"ERR after lane-0 {what}", mask=m.ST_ERR | m.ST_DDR)
            p.idle_until(t + 26)
            p.wrst()
            p.read_status(p.tc + 20, m.ST_DDR, "ERR cleared by WRST, mode stays on", mask=m.ST_ERR | m.ST_DDR)
            await play(dut, p, f"(lane-0 {what}, command {cmd})")


@DDR_TEST
async def test_ddr_mode_flag(dut):
    """STATUS bit 3 follows the mode, the protocol byte is 3, any other WRST data leaves the mode alone, and a reset clears it."""
    await fresh(dut)
    p = m.make_prog(CFG)
    p.read(1, m.ADDR_PROTO, CFG.proto, "protocol byte")
    p.read_status(2, m.ST_READY, "SDR after reset")
    p.nop(6)
    for junk in (0x00, 0xD2, 0xD4, 0x5C, 0xFF, 0x53):              # near misses of the two mode bytes
        p.emit(m.CMD_WRST, junk)
        p.nop(3)
    p.read_status(p.tc, m.ST_READY, "still SDR after WRSTs with other data", mask=m.ST_READY | m.ST_DDR)
    p.nop(4)
    p.ddr_enable(True)
    p.nop(4)
    p.read_status(p.tc, m.ST_READY | m.ST_DDR, "DDR bit set", mask=m.ST_READY | m.ST_DDR)
    for junk in (0x00, 0x5E, 0xD2):
        p.emit(m.CMD_WRST, junk)
        p.nop(3)
    p.read_status(p.tc, m.ST_READY | m.ST_DDR, "mode unchanged by other WRST data", mask=m.ST_READY | m.ST_DDR)
    p.nop(4)
    p.ddr_enable(False)
    p.nop(4)
    p.read_status(p.tc, m.ST_READY, "DDR bit cleared", mask=m.ST_READY | m.ST_DDR)
    await play(dut, p)
    q = m.make_prog(CFG)
    q.wrst()
    q.ddr_enable(True)
    q.nop(8)
    q.read_status(q.tc, m.ST_DDR | m.ST_READY, "on", mask=m.ST_READY | m.ST_DDR)
    await play(dut, q)
    await fresh(dut)                                                # a reset: the mode is off again
    q = m.make_prog(CFG)
    q.read_status(1, m.ST_READY, "reset clears the mode", mask=m.ST_READY | m.ST_DDR)
    await play(dut, q)


# ------------------------------------------------------------------------------------------------
# timing bounds of the weight groups, per lane
# ------------------------------------------------------------------------------------------------
def _lane(g):
    return 0 if g < CFG.ng // 2 else 1


def _ddr_load_tile(p, w, t0):
    """Write a tile in DDR mode in the shortest way: the two halves in parallel, one group per lane per clock. Returns the last clock."""
    nh = CFG.ng // 2
    last = t0
    for g, grp in enumerate(m.pack_tile(w, CFG.n)):
        t = t0 + (g if g < nh else g - nh)
        p.w8(grp, t, _lane(g))
        last = max(last, t)
    return last


async def _ddr_lower_case(dut, j, margin):
    """The lower bound of ONE group in DDR mode (test_timing._wgt_group_case with the groups on their lanes): tile 1 (+1 weights)
    has one plane in clock `last1`; tile 2 has +1 weights except for the PEs of group j (-1); group j is issued at its bound
    plus `margin`, the groups before it in the clocks just before, the groups after it one per clock behind it."""
    n = CFG.n
    big = m.digit_range(1)
    p = m.make_prog(CFG)
    p.wrst()
    p.ddr_enable(True)
    w1 = [[1] * n for _ in range(n)]
    x = [big] * n
    extra = early_groups(j, margin)                  # (test_timing.early_groups: groups that would fall on or before the plane go in front of it)
    t_gap = _ddr_load_tile(p, w1, p.tc) + 1
    last1 = t_gap + extra
    p.plane(m.plane_word([m.to_digits(v, 1)[0] for v in x]), m.CMD_XPS, last1, 1)
    w2 = [[(-1 if m.wpos(n, r, c) // m.GS == j else 1) for c in range(n)] for r in range(n)]
    tj = CFG.w_lower(last1, j) + margin
    ts = [t_gap + g if g < extra else (tj - (j - g) if g <= j else tj + (g - j)) for g in range(CFG.ng)]
    assert extra >= j + 1 or ts[extra] > last1
    for g, (grp, tt) in enumerate(zip(m.pack_tile(w2, n), ts)):
        p.w8(grp, tt, _lane(g))
    tp = ts[-1] + 1
    p.plane(m.plane_word([m.to_digits(v, 1)[0] for v in x]), m.CMD_XPF, tp, 1)
    m.read_block(p, CFG, tp, m.block_result([(w1, x), (w2, x)], CFG), tau=None, label=f"ddr group{j} margin{margin}")
    return p, await run_prog(dut, p)


@DDR_TEST
async def test_ddr_wgt_lower_bound_each_group(dut):
    """The lower bound of every single group holds in DDR mode on both lanes: at its bound the result is exact, one clock
    earlier it is corrupted."""
    await fresh(dut)
    for j in range(CFG.ng):
        p, out = await _ddr_lower_case(dut, j, 0)
        check(p, out, f"(DDR group {j} at its bound)")
        p, out = await _ddr_lower_case(dut, j, -1)
        try:
            check(p, out)
        except AssertionError:
            continue
        raise AssertionError(f"DDR group {j}: a W8 one clock before its bound did not corrupt the result")


async def _ddr_upper_case(dut, j, late):
    """The upper bound of one group in DDR mode: tile A (+1 weights) is computed and read, then tile B (-1 weights) is written by
    hand: the groups before j one per clock ahead of the first plane, the plane in clock b0, group j at its latest cycle plus
    `late` and the others one per clock behind it. (In DDR mode group 0 may share the clock of the plane: it is on the other lane.)"""
    n = CFG.n
    big = m.digit_range(1)
    p = m.make_prog(CFG)
    p.wrst()
    p.ddr_enable(True)
    wa = [[1] * n for _ in range(n)]
    wb = [[-1] * n for _ in range(n)]
    x = [big] * n
    pa = _ddr_load_tile(p, wa, p.tc) + 1
    p.plane(m.plane_word([m.to_digits(v, 1)[0] for v in x]), m.CMD_XPS, pa, 1)
    m.read_block(p, CFG, pa, m.block_result([(wa, x)], CFG), tau=None, label="A")
    s = max(p.tc, p.hazard_start + CFG.ng, pa + 40)
    b0 = s + j
    aj = b0 + CFG.w_upper(0, j) + late
    ts = [b0 - (j - g) if g < j else aj + (g - j) for g in range(CFG.ng)]
    for g, grp in enumerate(m.pack_tile(wb, n)):
        p.w8(grp, ts[g], _lane(g))
    plane_lane = 1 - _lane(j) if aj == b0 else 1                 # never on the lane of a group in the same clock
    p.plane(m.plane_word([m.to_digits(v, 1)[0] for v in x]), m.CMD_XPS, b0, plane_lane)
    m.read_block(p, CFG, b0, m.block_result([(wb, x)], CFG), tau=None, label=f"B(j={j},late={late})")
    return p, await run_prog(dut, p)


@DDR_TEST
async def test_ddr_wgt_upper_bound(dut):
    """The upper bound of every group holds in DDR mode: at the bound the result is exact, one clock later the first plane reaches
    the group's first PE with the old weight."""
    await fresh(dut)
    for j in range(CFG.ng):
        p, out = await _ddr_upper_case(dut, j, 0)
        check(p, out, f"(DDR group {j} at its bound)")
        p, out = await _ddr_upper_case(dut, j, 1)
        try:
            check(p, out)
        except AssertionError:
            continue
        raise AssertionError(f"DDR group {j}: a W8 one clock after its bound did not corrupt the result")


@DDR_TEST
async def test_ddr_ready_follows_lane0_planes(dut):
    """READY falls with a plane on lane 0 exactly as with one on lane 1: low until 1 + TSETTLE clocks after the last plane (the effective
    clock), high from then on (test_timing.test_ready_timing for the DDR lane)."""
    await fresh(dut)
    r = rng(38)
    for xl in ((0,), (1,)):
        await fresh(dut)
        p = m.make_prog(CFG)
        p.xlanes = xl
        p.wrst()
        p.ddr_enable(True)
        la, acc = m.plan_block(p, CFG, [(m.rand_tile(CFG.n, r), m.rand_x(CFG.n, 2, r), 2)])
        for dt in (12, CFG.tsettle // 2, CFG.tsettle - 1, CFG.tsettle, CFG.tsettle + 1, CFG.tsettle + 4):      # (the last W8 of the tile is at most 11 clocks after its plane)
            p.read_status(la + dt, m.ST_READY if dt >= 1 + CFG.tsettle else 0, f"READY at t0+{dt}, planes on lane(s) {xl}", mask=m.ST_READY)
        await play(dut, p, f"(planes on lane(s) {xl})")


# ------------------------------------------------------------------------------------------------
# throughput
# ------------------------------------------------------------------------------------------------
@DDR_TEST
async def test_ddr_throughput_table(dut):
    """Clocks per tile in DDR mode: blocks of 12 and 20 tiles are simulated for every D (the results are checked against the
    model), and the difference of the last plane clocks divided by the 8 extra tiles is the steady-state period of the planner."""
    await fresh(dut)
    r = rng(37)
    got = {}
    for d in (1, 2, 3, 4):
        last = {}
        for tiles in (12, 20):
            p = m.make_prog(CFG)
            p.wrst()
            p.ddr_enable(True)
            tl = [(m.rand_tile(CFG.n, r), m.rand_x(CFG.n, d, r), d) for _ in range(tiles)]
            la, acc = m.plan_block(p, CFG, tl)
            m.read_block(p, CFG, la, acc, tau=None, label=f"ddr {tiles} tiles d{d}")
            await play(dut, p, f"(D={d}, {tiles} tiles)")
            last[tiles] = la
        got[d] = (last[20] - last[12]) / 8
        if CFG.n == 8:                                           # N = 4 has 2 groups per tile: DDR cannot beat SDR there (a host stays in SDR mode)
            assert got[d] < CFG.ng + d, f"D={d}: the DDR period {got[d]} is not shorter than SDR's {CFG.ng + d}"
    dut._log.info("DDR clocks per tile: %s", got)
