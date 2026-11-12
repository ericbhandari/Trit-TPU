# SPDX-License-Identifier: Apache-2.0
"""Status flags, protocol corner cases, reset behaviour and power-up from arbitrary state."""
import cocotb

import ttpu_model as m
import rtl_info
from harness import (CFG, QUICK, is_gl, check, fresh, plant, play, rand_xv, reset, rng, run_prog, save_result,
                     scramble, start_clock)


@cocotb.test()
async def test_status_err(dut):
    """ERR sets on illegal packed values (data byte 243..255, field 27..31) in W8 and plane commands, is sticky and is
    cleared by WRST; the same values in other commands are not errors; the reserved status bit stays 0."""
    await fresh(dut)
    n = CFG.n
    p = m.make_prog(CFG)
    p.wrst()
    p.nop(2)
    p.read_status(p.tc, 0, "no err initially", mask=m.ST_ERR | 0x02)
    p.nop(4)
    cases = [(m.CMD_W8, bad, 0) for bad in (243, 255, 250)] + [(m.CMD_W8, 0, bad) for bad in (27, 31, 28)] + [(m.CMD_W8, 243, 27)]
    cases += [(m.CMD_XP, bad, 0) for bad in (243, 255)] + [(m.CMD_XPF, 250, 0), (m.CMD_XPS, 244, 3)]
    if n > 4:                                        # for N = 4 the field of a plane is not used and not checked
        cases += [(m.CMD_XP, 0, 27), (m.CMD_XPF, 0, 31), (m.CMD_XPS, 0, 30)]
    for cmd, bd, fd in cases:
        p.emit(cmd, bd, fd)
        p.nop(7)
        p.read_status(p.tc, m.ST_ERR, f"err after cmd {cmd} byte {bd} field {fd}", mask=m.ST_ERR)
        p.nop(4)
        p.emit(m.CMD_W8, 100, 13)                    # a legal group does not clear it
        p.nop(7)
        p.read_status(p.tc, m.ST_ERR, "err sticky", mask=m.ST_ERR)
        p.nop(4)
        p.wrst()
        p.nop(7)
        p.read_status(p.tc, 0, "err cleared by WRST", mask=m.ST_ERR)
        p.nop(4)
    if n == 4:
        p.emit(m.CMD_XP, 0, 31)                      # (N = 4) a plane's field is don't-care
        p.nop(7)
        p.read_status(p.tc, 0, "N=4: field of a plane is not checked", mask=m.ST_ERR)
        p.nop(4)
    for cmd, bd, fd in ((m.CMD_W8, 242, 26), (m.CMD_XP, 242, 26)):   # the largest legal values
        p.emit(cmd, bd, fd)
        p.nop(7)
        p.read_status(p.tc, 0, "242 / 26 are legal", mask=m.ST_ERR)
        p.nop(4)
    p.wrst()
    p.nop(6)
    for cmd in (m.CMD_NOP, m.CMD_CFG):               # an illegal value on the pins is only an error in a W8 / plane cycle
        for bad in (243, 250, 255):
            p.emit(cmd, bad, 0)
    p.nop(8)
    p.read_status(p.tc, 0, "illegal values without W8/XP* do not set ERR", mask=m.ST_ERR | 0x02)
    await play(dut, p)


@cocotb.test()
async def test_illegal_parts_are_zero(dut):
    """An illegal data byte zeroes the five trits it carries, an illegal field zeroes the three of the field, in weights and in
    planes; the legal part of the same command is used. Checked on the computed result (also works on the gate-level netlist)."""
    await fresh(dut)
    n = CFG.n
    r = rng(19)
    for trial in range(2 if QUICK else 6):
        groups = [(r.randrange(243), r.randrange(27)) for _ in range(CFG.ng)]
        for g in r.sample(range(CFG.ng), min(3, CFG.ng)):
            b, f = groups[g]
            groups[g] = r.choice([(r.randrange(243, 256), f), (b, r.randrange(27, 32)), (r.randrange(243, 256), r.randrange(27, 32))])
        w = m.unpack_tile(groups, n)                 # the model zeroes the illegal parts
        d = r.choice((1, 2))
        x = rand_xv(n, d, r)
        planes = []
        for k in range(d):
            word = m.plane_word([m.to_digits(v, d)[k] for v in x])
            if k == 0 and trial % 2 == 0:            # damage plane 0: its byte part (or, for N = 8, its field) becomes illegal
                word = (r.randrange(243, 256), word[1]) if n == 4 or trial % 4 == 0 else (word[0], r.randrange(27, 32))
            planes.append(word)
        # the digits the chip really uses: planes decoded by the model
        used = [m.unpack_group(*wd)[:n] for wd in planes]
        xv = [sum(used[k][i] * 3 ** k for k in range(d)) for i in range(n)]
        p = m.make_prog(CFG)
        p.wrst()
        p.load_groups(groups)
        for k, wd in enumerate(planes):
            p.plane(wd, m.CMD_XPS if k == 0 else m.CMD_XP)
        acc = m.block_result([(w, xv)], CFG)
        m.read_block(p, CFG, p.last_plane, acc, tau=None, label=f"illegal{trial}")
        await play(dut, p, f"(trial {trial})")


@cocotb.test()
async def test_protocol_robustness(dut):
    """NOP junk, read streams between commands, pointer wrap, WRST mid-tile, planes separated by long gaps, plane wrap."""
    await fresh(dut)
    n = CFG.n
    r = rng(11)
    p = m.make_prog(CFG)
    p.wrst()

    # NOP+junk data and (harmless) read streams between W8 commands change nothing; gaps between planes are fine
    w = m.rand_tile(n, r)
    x = rand_xv(n, 2, r)
    for grp in m.pack_tile(w, n):
        p.w8(grp)
        if r.random() < 0.5:
            p.emit(m.CMD_RDS, r.randrange(0, 6))     # reads some addresses of the (idle) read port; nothing to check
        p.emit(m.CMD_NOP, r.getrandbits(8))
    last = p.vector(x, 2, clear=True, gaps={1: 9})
    m.read_block(p, CFG, last, m.block_result([(w, x)], CFG), tau=None, label="rds/nop/gaps")

    # pointer wrap: two tiles back to back without WRST -> the second one wins
    w2 = m.rand_tile(n, r)
    x2 = rand_xv(n, 1, r)
    p.idle_until(p.hazard_start)
    p.load_groups(m.pack_tile(w, n))
    p.load_groups(m.pack_tile(w2, n))
    last = p.vector(x2, 1, clear=True)
    m.read_block(p, CFG, last, m.block_result([(w2, x2)], CFG), tau=None, label="wrap")

    # WRST in the middle of a tile restarts at group 0
    p.idle_until(p.hazard_start)
    p.load_groups(m.pack_tile(w2, n)[:CFG.ng // 2 + 1])
    p.wrst()
    p.load_groups(m.pack_tile(w, n))
    last = p.vector(x2, 1, clear=True)
    m.read_block(p, CFG, last, m.block_result([(w, x2)], CFG), tau=None, label="wrst-mid-tile")

    # digit planes of one vector separated by long gaps
    p.idle_until(p.hazard_start)
    p.load_tile(w)
    xd = rand_xv(n, 3, r)
    t = p.vector(xd, 3, clear=True, gaps={1: 7, 2: 14})
    m.read_block(p, CFG, t, m.block_result([(w, xd)], CFG), tau=None, label="gapped planes")

    # RDS commands between the planes of a vector (the stream runs while the planes are sent) change nothing
    p.idle_until(max(p.hazard_start, p.read_end + 1))      # (the previous block's read stream must be over: one address register)
    p.load_tile(w)
    xr = rand_xv(n, 3, r)
    pr = [m.plane_word([m.to_digits(v, 3)[k] for v in xr]) for k in range(3)]
    for k in range(3):
        t = p.plane(pr[k], m.CMD_XPS if k == 0 else m.CMD_XP)
        p.emit(m.CMD_RDS, r.randrange(0, 4))
    m.read_block(p, CFG, t, m.block_result([(w, xr)], CFG), tau=None, label="rds inside a vector")

    # the plane counter wraps at 4: a fifth plane is scaled by 3^0 again
    p.idle_until(p.hazard_start)
    p.load_tile(w)
    xe = rand_xv(n, 4, r)
    pe = [m.to_digits(v, 4) for v in xe]
    extra = [r.choice((-1, 0, 1)) for _ in range(n)]
    for k in range(5):
        pl = [q[k] for q in pe] if k < 4 else extra
        t = p.plane(m.plane_word(pl), m.CMD_XPS if k == 0 else m.CMD_XP)
    exp = [m.s_wrap(sum(w[rr][c] * (xe[rr] + extra[rr]) for rr in range(n)), 16) for c in range(n)]
    m.read_block(p, CFG, t, exp, tau=None, label="5th plane wraps to scale 1")
    await play(dut, p)


@cocotb.test()
async def test_gaps_and_batches(dut):
    """Several vectors per tile (batch), back to back and with random gaps, accumulate correctly."""
    await fresh(dut)
    n = CFG.n
    r = rng(12)
    for trial in range(2 if QUICK else 6):
        p = m.make_prog(CFG)
        p.wrst()
        w = m.rand_tile(n, r)
        p.load_tile(w)
        vecs, last = [], None
        for i in range(r.randint(2, 5)):
            d = r.choice((1, 2, 3, 4))
            x = rand_xv(n, d, r, 0.1)
            gaps = {j: r.randint(1, 4) for j in range(d) if r.random() < 0.3}
            last = p.vector(x, d, clear=(i == 0), gaps=gaps)
            vecs.append((w, x))
            if r.random() < 0.4:
                p.nop(r.randint(1, 10))
        m.read_block(p, CFG, last, m.block_result(vecs, CFG), tau=None, label=f"batch{trial}")
        await play(dut, p, f"(trial {trial})")


@cocotb.test()
async def test_read_stream(dut):
    """RDS: one command reads count+1 consecutive addresses, one per clock, with the address pins ignored meanwhile (they carry random
    junk), the address wraps modulo 32, a new RDS restarts the stream, a count of 0 is a single read, and a stream runs while
    W8 / plane commands (whose field pins carry data) are sent."""
    await fresh(dut)
    n = CFG.n
    r = rng(20)
    w = m.rand_tile(n, r)
    x = rand_xv(n, 2, r)
    tau = 3
    p = m.make_prog(CFG)
    p.wrst()
    p.cfg_tau(tau)
    p.load_tile(w)
    last = p.vector(x, 2, clear=True)
    acc = m.block_result([(w, x)], CFG)
    mem = m.read_map(CFG, acc, tau)
    t = last + 1 + CFG.tsettle + 2
    p.idle_until(t)
    # (a) the whole address space from 0, wrapping a second time round
    p.read_stream(t, 0, [mem[a % 32] for a in range(0, 40)], "all addresses")
    t += 40
    # (b) a start in the middle, crossing the wrap
    p.idle_until(t)
    p.read_stream(t, 0x1C, [mem[(0x1C + i) % 32] for i in range(8)], "wrap 1c..03")
    t += 8
    # (c) count 0 is a single read; the next command may follow in the very next clock
    p.idle_until(t)
    p.read_stream(t, m.ADDR_TAU, [tau], "count 0")
    p.read_stream(t + 1, m.ADDR_INFO1, [CFG.accw, m.ST_READY], "count 1 right behind")
    t += 3
    # (d) a stream interrupted by a new RDS after 3 reads
    p.idle_until(t + 1)
    p.read_stream(t, 0x10, [mem[0x10 + i] for i in range(3)], "interrupted (first part)")
    p.read_stream(t + 3, 0x14, [mem[0x14 + i] for i in range(6)], "restarted")
    t += 9
    # (e0) lengths around the bits of the 6-bit count: 31, 32, 33 and 34 more reads (a count with only the top bit set, or with the
    #      top bit and nothing else above bit 0, is where a count that loses its top bit differs)
    for cnt in (31, 32, 33, 34):
        p.idle_until(t)
        start = (cnt * 7) % 32
        p.read_stream(t, start, [mem[(start + i) % 32] for i in range(cnt + 1)], f"{cnt} more reads")
        t += cnt + 1
    # (e) the longest stream (63 more reads)
    p.idle_until(t)
    p.read_stream(t, 5, [mem[(5 + i) % 32] for i in range(64)], "64 reads")
    await play(dut, p)

    # (f) streams while a tile is being loaded and a vector is sent (field pins carry data meanwhile)
    p = m.make_prog(CFG)
    p.wrst()
    p.cfg_tau(tau)
    p.load_tile(w)
    last = p.vector(x, 2, clear=True)
    t = last + 1 + CFG.tsettle + 2
    p.idle_until(t)
    p.read_tern(t, acc, tau, "tern")                  # RDS(0x08, G-1)
    p.read_raw(t + CFG.g, acc, "raw")                 # RDS(0x10, 15) right behind it
    w2 = m.rand_tile(n, r)
    x2 = rand_xv(n, 3, r)
    p.hazard_start = max(p.hazard_start, t + CFG.g - CFG.ng)
    last2, acc2 = m.plan_block(p, CFG, [(w2, x2, 3)])  # its W8 / XPS commands overlap the raw stream
    m.read_block(p, CFG, last2, acc2, tau=None, label="second block")
    await play(dut, p, "(streams under traffic)")


def _free_read_cycle(p: m.Prog, ln: int = 1) -> int:
    """First cycle at or after the next free command cycle where `ln` consecutive cycles are free for a read (no command, no address, no stream)."""
    t = p.tc
    while any((t + i) in p._cover or (t + i) in p._addr_set or (t + i < len(p.cmd) and p.cmd[t + i] != m.CMD_NOP) for i in range(ln)):
        t += 1
    return t


@cocotb.test()
async def test_output_hold(dut):
    """Output hold: in a clock whose field pins carry data (W8 / XP* outside a read stream) the read of that clock must not
    reach uo_out. Every test checks this on its own program (harness.check); this one is made for it. Random traffic: W8 and plane words with
    random legal data (so that the data clocks would address every byte of the read port, nearly all of them unknown or changing),
    single reads and read streams of the constant bytes of the port (ID, info, accumulator width, tau, protocol) in any order, a read in the
    very next clock after a data clock and the clock before one, streams that run while the commands go on, and idle polling. The reads must
    return their bytes (the expectations) and uo_out must not move in the clock that follows a data slot. Run in SDR mode with junk on lane 0
    and in DDR mode with words on lane 0 (lane 0 is never an address)."""
    total = 0
    await fresh(dut)
    for variant in (("sdr",) + (("ddr",) if CFG.ddr else ())):
        if variant != "sdr":
            await reset(dut, 8)                                       # (the clock is already running)
        r = rng(41 if variant == "sdr" else 42)
        tau = r.randrange(1, 200)
        p = m.make_prog(CFG)
        p.wrst()
        if variant == "ddr":
            p.ddr_enable(True)
        p.cfg_tau(tau)
        p.nop(8)
        mem = m.read_map(CFG, None, tau)
        mem[m.ADDR_STATUS] = None                                     # (READY / ERR move with the traffic: not part of the check)
        steps = 700 if not QUICK else 250
        for _ in range(steps):
            k = r.random()
            if k < 0.34:
                p.emit(m.CMD_W8, r.randrange(243), r.randrange(27))
            elif k < 0.47:
                p.emit(r.choice((m.CMD_XP, m.CMD_XPF, m.CMD_XPS)), r.randrange(243), r.randrange(27))
            elif k < 0.68:                                            # a single read of one of the constant bytes
                a = r.choice((m.ADDR_ID, m.ADDR_INFO0, m.ADDR_INFO1, m.ADDR_TAU, m.ADDR_PROTO))
                t = _free_read_cycle(p)
                p.read(t, a, mem[a], f"read {a:#04x}")
                p.tc = max(p.tc, t + 1)
            elif k < 0.80:                                            # a stream over addresses 0..5 (status is not checked)
                ln = r.randrange(1, 7)
                a = r.randrange(0, 7 - ln)
                t = _free_read_cycle(p, ln)
                p.read_stream(t, a, mem[a:a + ln], f"stream {a}+{ln}")
            elif k < 0.87:                                     # a read in the clock of a CFG (same tau) or WRST (data 0): their field pins are an address
                a = r.choice((m.ADDR_ID, m.ADDR_INFO0, m.ADDR_INFO1, m.ADDR_TAU, m.ADDR_PROTO))
                cmd_ = m.CMD_CFG if r.random() < 0.5 else m.CMD_WRST
                t = _free_read_cycle(p)                               # (a free cycle: no command, no address, not inside a stream)
                p.put(t, cmd_, tau if cmd_ == m.CMD_CFG else 0)
                p.tc = max(p.tc, t + 1)
                p.read(t, a, mem[a], f"read {a:#04x} in a {'CFG' if cmd_ == m.CMD_CFG else 'WRST'} clock")
            elif k < 0.94 and variant == "ddr":                       # lane 0 words: data of W8 / planes, never an address
                t = r.randrange(max(p.tc - 3, 0), p.tc + 3)
                if not p.busy0(t):
                    p.put0(t, r.choice((m.CMD_W8, m.CMD_W8, m.CMD_XP, m.CMD_XPF)), r.randrange(243), r.randrange(27))
            else:
                p.emit(m.CMD_NOP, r.getrandbits(8))
        p.nop(10)
        if variant == "sdr":
            p.scramble_lane0(r, len(p) + 2)
        frozen = p.frozen_cycles()
        n_reads = len(p.exp)
        assert len(frozen) >= (80 if QUICK else 200) and n_reads >= 100, f"the random program is too tame: {len(frozen)} data clocks, {n_reads} read bytes"
        await play(dut, p, f"(random traffic, {variant})")
        total += len(frozen)
        dut._log.info(f"output hold, {variant}: {len(frozen)} data clocks held, {n_reads} bytes read, {len(p)} clocks")
    save_result("output_hold_data_clocks", total)


@cocotb.test()
async def test_threshold_boundaries(dut):
    """Ternarizer: sums engineered to sit at tau-1, tau, tau+1 and the negatives, tau = 0..255."""
    await fresh(dut)
    n = CFG.n
    taus = [0, 1, 8, 128, 255] if QUICK else [0, 1, 7, 8, 9, 31, 100, 127, 128, 255]
    for tau in taus:
        targets = ([tau - 1, tau, tau + 1, -(tau - 1), -tau, -(tau + 1), 0, (tau * 5) // 3] * 2)[:n]
        ntile = max(abs(t) for t in targets) // n + 2
        tiles = []
        for ti in range(ntile):
            w = [[0] * n for _ in range(n)]
            for c, t in enumerate(targets):
                cnt = max(0, min(n, abs(t) - ti * n))
                for rr in range(cnt):
                    w[rr][c] = 1 if t >= 0 else -1
            tiles.append((w, [1] * n, 1))
        p = m.make_prog(CFG)
        p.wrst()
        p.cfg_tau(tau)
        last, acc = m.plan_block(p, CFG, tiles)
        assert [abs(a) for a in acc] == [abs(t) for t in targets], (acc, targets)
        m.read_block(p, CFG, last, acc, tau=tau, label=f"tau{tau}")
        await play(dut, p, f"(tau {tau})")


@cocotb.test()
async def test_reset_midstream(dut):
    """Reset in the middle of a vector and of a weight load: control state returns to idle, a new block works."""
    await fresh(dut)
    n = CFG.n
    r = rng(13)
    for width in (3, 6):
        p = m.make_prog(CFG)                         # leave junk: half a tile, half a vector, a config change
        p.cfg_tau(77)
        p.load_groups(m.pack_tile(m.rand_tile(n, r), n)[:CFG.ng - 1])
        p.emit(m.CMD_XPS, 0xAA, 0x15)
        p.nop(3)
        await run_prog(dut, p)
        await reset(dut, width)
        q = m.make_prog(CFG)
        q.read(0, m.ADDR_TAU, 0, "tau cleared by reset")
        q.read_status(1, m.ST_READY, "status idle after reset", mask=0x07)
        w = m.rand_tile(n, r)
        x = rand_xv(n, 2, r)
        q.nop(6)
        q.load_tile(w)                               # no WRST: reset must have rewound the pointer
        last = q.vector(x, 2, clear=True)
        # tau = 0 after reset must also hold inside the ternarizer (not only in the readback copy)
        m.read_block(q, CFG, last, m.block_result([(w, x)], CFG), tau=0, label=f"after reset({width})")
        await play(dut, q, f"(reset width {width})")


@cocotb.test()
async def test_powerup_state(dut):
    """RTL only: scramble every flip-flop to random values, reset, and run a block. Repeats for many
    seeds and short reset pulses; a no-reset control shows the scramble really perturbs the design."""
    if is_gl(dut):
        dut._log.info("skipped in gate-level run (state scrambling needs RTL access)")
        return
    await start_clock(dut)
    n = CFG.n
    r = rng(14)
    n_regs = 0
    widths = (3,) if QUICK else (3, 4, 8)
    for sd in range(6 if QUICK else 24):
        width = widths[sd % len(widths)]
        n_regs = scramble(dut, r)
        await reset(dut, width, release_wait=0)
        p = m.make_prog(CFG)
        if sd % 2:                                   # odd seeds: look at the flags first (junk in the plane / weight flag pipelines must not
            p.nop(10)                                # have set ERR or kept READY low), then run the block
            p.read_status(p.tc, m.ST_READY, "READY and no ERR after a scrambled reset", mask=m.ST_READY | m.ST_ERR)
            p.nop(8)
        d = r.choice((1, 2, 3, 4))
        tiles = [(m.rand_tile(n, r), rand_xv(n, d, r, 0.1), d) for _ in range(2)]
        last, acc = m.plan_block(p, CFG, tiles)      # (even seeds:) first W8 in the very clock rst_n rises
        m.read_block(p, CFG, last, acc, tau=None, label=f"powerup{sd}")
        await play(dut, p, f"(seed {sd}, reset width {width})", immediate=True)
    dut._log.info(f"scrambled {n_regs} state elements per iteration")
    differs = 0                                      # negative control: scramble without reset is visible
    for _ in range(8):
        scramble(dut, r)
        q = m.make_prog(CFG)
        q.read(0, m.ADDR_TAU, 0, "tau")
        q.read_status(1, m.ST_READY, "status", mask=0x07)
        out = await run_prog(dut, q)
        try:
            check(q, out)
        except AssertionError:
            differs += 1
    assert differs > 0, "scrambling had no visible effect: the power-up test would be vacuous"
    save_result("scramble_regs", n_regs)


@cocotb.test()
async def test_powerup_planted_plane_junk(dut):
    """RTL only: the one power-up state that can still set ERR after a minimal (3-clock) reset is a plane command in the decode
    stage with an illegal data byte: the flags travel down the plane pipeline and reach the ERR logic two clocks after the
    reset has been released unless the pipeline's last-but-two stage is gated by the registered reset (`pk_gate`). A random
    scramble only hits it by luck (a plane flag, a decode gate that is open and a byte of 243..255 together), so plant it:
    every plane flavour, every stage of the pipeline that a 3-clock reset has not flushed yet, and the same with a legal
    byte as the control (the planted flag alone must not matter)."""
    if is_gl(dut):
        dut._log.info("skipped in gate-level run (state planting needs RTL access)")
        return
    await start_clock(dut)
    n = CFG.n
    r = rng(31)
    trials = 0
    for flag in ("xc1r", "xf1r", "xs1r"):
        for byte in (0xFF, 0xF3, 0x00):
            scramble(dut, r)
            for nm in ("xc1r", "xf1r", "xs1r"):
                plant(dut, nm, 1 if nm == flag else 0)
            plant(dut, "d1w", byte)
            plant(dut, "g_rq[2].q", 1)       # the decode gate that lets the planted flag into the pipeline
            for nm in ("wgt1r", "wrst1r", "cfg1r", "rds1r"):
                plant(dut, nm, 0)
            await reset(dut, 3, release_wait=0)
            p = m.make_prog(CFG)
            p.nop(10)
            p.read_status(p.tc, m.ST_READY, f"READY and no ERR ({flag}, byte {byte:#04x}, planted before a 3-clock reset)",
                          mask=m.ST_READY | m.ST_ERR)
            p.nop(8)
            tiles = [(m.rand_tile(n, r), rand_xv(n, 2, r, 0.1), 2)]
            last, acc = m.plan_block(p, CFG, tiles)
            m.read_block(p, CFG, last, acc, tau=None, label=f"planted{trials}")
            await play(dut, p, f"({flag}, byte {byte:#04x})", immediate=True)
            trials += 1
    dut._log.info(f"{trials} planted power-up states, no ERR after any")


@cocotb.test()
async def test_min_reset_width(dut):
    """RTL only: how short may the reset pulse be and still give a clean start from a scrambled state?"""
    if is_gl(dut):
        return
    await start_clock(dut)
    n = CFG.n
    r = rng(15)
    results = {}
    starts = 12 if QUICK else 40
    for width in (1, 2, 3, 4):
        fails = 0
        for _ in range(starts):
            scramble(dut, r)
            await reset(dut, width, release_wait=0)
            p = m.make_prog(CFG)
            tiles = [(m.rand_tile(n, r), rand_xv(n, 2, r), 2)]
            last, acc = m.plan_block(p, CFG, tiles)
            m.read_block(p, CFG, last, acc, tau=None, label="rw")
            out = await run_prog(dut, p, immediate=True)
            try:
                check(p, out)
            except AssertionError:
                fails += 1
        results[width] = fails
    dut._log.info(f"failures out of {starts} scrambled starts per reset width: {results}")
    save_result("reset_width_failures", results)
    save_result("reset_width_starts", starts)
    assert results[3] == 0 and results[4] == 0, f"a 3-clock reset must be sufficient: {results}"


@cocotb.test()
async def test_commands_during_reset(dut):
    """Commands applied while rst_n is low (including the last reset clock) are ignored; the first clock
    with rst_n high already accepts commands."""
    await fresh(dut)
    n = CFG.n
    r = rng(18)
    # leave tau = 77 and a half written tile behind so that "ignored" can be told from "reset to default"
    p0 = m.make_prog(CFG)
    p0.cfg_tau(77)
    p0.load_groups(m.pack_tile(m.rand_tile(n, r), n)[:CFG.ng - 2])
    p0.nop(4)
    await run_prog(dut, p0)

    release = 9
    p = m.make_prog(CFG)
    junk = [(m.CMD_W8, 250, 3), (m.CMD_W8, 255, 31), (m.CMD_XPS, 0xAA, 5), (m.CMD_XP, 0x55, 7), (m.CMD_CFG, 200, 0),
            (m.CMD_XPF, 0x0F, 9), (m.CMD_RDS, 63, 0), (m.CMD_XPS, 0xFF, 31), (m.CMD_W8, 251, 30)]
    for c, d, f in junk:                             # cycles 0..8: all with rst_n low, the last one is the
        p.emit(c, d, f)                              # final reset clock
    assert p.tc == release
    # cycle 9: rst_n is high. Status first (no command), then a clean block right away.
    p.read_status(release, m.ST_READY, "ready after reset", mask=m.ST_READY | m.ST_ERR | 0x02)
    p.read(release + 1, m.ADDR_TAU, 0, "tau reset")
    w = m.rand_tile(n, r)
    x = rand_xv(n, 3, r)
    p.idle_until(release + 6)
    p.load_tile(w)                                   # pointer must be at group 0 without a WRST
    last = p.vector(x, 3, clear=True)
    m.read_block(p, CFG, last, m.block_result([(w, x)], CFG), tau=0, label="clean block after junk in reset")
    await play(dut, p, rst_release=release)

    # one command type at a time in the very last reset clock, followed by a tile load in the clock rst_n rises: a command
    # that was wrongly accepted in reset would reach the weight pointer, the sequencer, tau or the error flag a few clocks
    # later (each would be a ghost in the decode pipeline)
    names = {m.CMD_W8: "W8", m.CMD_WRST: "WRST", m.CMD_CFG: "CFG", m.CMD_XP: "XP", m.CMD_XPF: "XPF", m.CMD_XPS: "XPS", m.CMD_RDS: "RDS"}
    for cmd, data in ((m.CMD_W8, 250), (m.CMD_WRST, 0), (m.CMD_CFG, 200), (m.CMD_XP, 0x55), (m.CMD_XPF, 0x0F), (m.CMD_XPS, 0xAA), (m.CMD_RDS, 63),
                      (m.CMD_XP, 250), (m.CMD_XPF, 243), (m.CMD_XPS, 255)):      # (illegal data in a plane would set ERR if the plane were accepted)
        pg = m.make_prog(CFG)
        pg.nop(release - 1)
        pg.emit(cmd, data)                           # the last clock with rst_n low
        assert pg.tc == release
        w = m.rand_tile(n, r)
        x = rand_xv(n, 3, r)
        pg.load_tile(w)                              # first W8 command in the clock rst_n rises
        ts = pg.tc + 1                               # status and tau are read after the tile, before the first plane
        pg.read_status(ts, m.ST_READY, f"status after {names[cmd]} in the last reset clock",
                       mask=m.ST_READY | m.ST_ERR | 0x02)
        pg.read(ts + 2, m.ADDR_TAU, 0, f"tau after {names[cmd]} in the last reset clock")
        pg.idle_until(ts + 6)
        last = pg.vector(x, 3, clear=True)
        m.read_block(pg, CFG, last, m.block_result([(w, x)], CFG), tau=0, label=f"block after {names[cmd]} in the last reset clock")
        await play(dut, pg, f"({names[cmd]} in the last reset clock)", rst_release=release)

    # a read stream accepted in the last reset clock would take over the address register: the very first read after the reset shows it
    pr = m.make_prog(CFG)
    pr.nop(release - 1)
    pr.emit(m.CMD_RDS, 63)                           # the last clock with rst_n low
    assert pr.tc == release
    pr.read(release, m.ADDR_ID, m.ID_BYTE, "id in the first clock after reset")
    pr.read(release + 1, m.ADDR_INFO0, (4 << 5) | CFG.n, "info0")
    pr.read(release + 2, m.ADDR_INFO1, CFG.accw, "accw")
    pr.read(release + 3, m.ADDR_PROTO, CFG.proto, "protocol version")
    pr.nop(8)
    await play(dut, pr, "(RDS in the last reset clock)", rst_release=release)

    # the first clock with rst_n high accepts a command
    p2 = m.make_prog(CFG)
    p2.emit(m.CMD_WRST)                              # cycle 0: rst_n rises with this command
    p2.emit(m.CMD_W8, 250, 0)                        # an illegal byte, must set ERR
    p2.nop(8)
    p2.read_status(p2.tc, m.ST_ERR, "command in the first clock after reset is accepted", mask=m.ST_ERR)
    await reset(dut, 3, release_wait=0)
    await play(dut, p2, immediate=True)


def _tiles_for(targets, n, d=4):
    """Tiles (weights, activations, digits) whose accumulation gives exactly `targets` (one signed sum per column) at
    d = 4 digits. Bulk tiles carry the largest digit value on every row (n * 40 per tile and column; a column stops
    using them when it has enough), one more tile adds 40 per selected row, and the remaining 0..39 is a subset sum of
    the row values 1, 2, 4, 8, 16, 32 spread over ceil(6 / n) tiles."""
    big = m.digit_range(d)
    plan = []
    for t in targets:
        a = abs(t)
        full, a = divmod(a, n * big)
        k, rest = divmod(a, big)
        plan.append((1 if t >= 0 else -1, full, k, rest))
    tiles = []
    for i in range(max(p[1] for p in plan)):
        tiles.append(([[(sg if i < full else 0) for (sg, full, k, rest) in plan] for _ in range(n)], [big] * n, d))
    tiles.append(([[(sg if r < k else 0) for (sg, full, k, rest) in plan] for r in range(n)], [big] * n, d))
    for lo in range(0, 6, n):
        rows = list(range(lo, min(lo + n, 6)))
        xs = [1 << b for b in rows] + [0] * (n - len(rows))
        tiles.append(([[(sg if r < len(rows) and (rest >> rows[r]) & 1 else 0) for (sg, full, k, rest) in plan] for r in range(n)], xs, d))
    return tiles


@cocotb.test()
async def test_ternarizer_large_sums(dut):
    """Ternarizer and raw read-out on large accumulator values. The magnitudes 4099, 8195, ... 28675 have exactly
    one combination of the bits 14:12 set (bits 11:8 are zero, the low byte is 3), so a slice error in the
    upper-bit test cannot hide behind a neighbouring bit; 32765 and 32771 ... 57347 cross the 16-bit wrap. Both signs,
    tau = 255 (only the upper bits can make the result non-zero) and tau = 0. The threshold-boundary test only
    reaches a few hundred."""
    await fresh(dut)
    n = CFG.n
    mags = [4099, 8195, 12291, 16387, 20483, 24579, 28675, 32765, 32771, 40963, 49155, 57347]
    for flip in ((0,) if QUICK else (0, 1)):
        for start in range(0, len(mags), n):
            chunk = (mags[start:start + n] * 2)[:n]
            targets = [(1 if (c + flip) % 2 == 0 else -1) * v for c, v in enumerate(chunk)]
            for tau in ((255,) if QUICK else (255, 0)):
                p = m.make_prog(CFG)
                p.wrst()
                p.cfg_tau(tau)
                last, acc = m.plan_block(p, CFG, _tiles_for(targets, n))
                assert acc == [m.s_wrap(t, CFG.accw) for t in targets], (acc, targets)
                m.read_block(p, CFG, last, acc, tau=tau, label=f"large{start}.f{flip}.tau{tau}")
                await play(dut, p, f"(targets {targets[:3]}..., tau {tau})")


# `reg` variables that are combinational (assigned with blocking assignments in always @(*) blocks). Every other
# `reg` of the RTL must be a flip-flop: it is then found by test/rtl_info.py and scrambled by the power-up test.
COMBINATIONAL_REGS = {"ax", "hi_c", "lo_c", "po_a", "po_b", "po_c"}


@cocotb.test()
async def test_state_list_complete(dut):
    """The power-up test must scramble every flip-flop: every `reg` of the RTL is either the target of a
    non-blocking assignment (and so on the scrambler's list, taken from the RTL itself) or listed as combinational
    here; a register that is neither fails this test. The scrambler must also reach most of the names in this build."""
    declared = rtl_info.reg_decls()
    flops = rtl_info.flop_names()
    unclassified = declared - flops - COMBINATIONAL_REGS
    assert not unclassified, f"registers that are neither flip-flops nor listed as combinational: {sorted(unclassified)}"
    stale = COMBINATIONAL_REGS - declared
    assert not stale, f"COMBINATIONAL_REGS lists names that are not declared any more: {sorted(stale)}"
    if is_gl(dut):
        return
    await start_clock(dut)
    count = scramble(dut, rng(21))
    dut._log.info(f"{len(flops)} flip-flop names in the RTL, {count} state elements scrambled in this build")
    assert count >= 25 * CFG.n, f"the scrambler reached only {count} state elements"
