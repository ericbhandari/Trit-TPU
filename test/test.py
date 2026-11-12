# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0
"""TTPU regression (cocotb).

Every test builds a cycle-level program with ttpu_model (pin values per clock plus the expected read
data from an independent integer model), plays it on the DUT and compares. Tests are independent:
each one starts with a reset but deliberately keeps whatever junk the previous test left in the
unreset datapath, which is itself part of what is being verified.

Environment: TTPU_N (array size, default 8), TTPU_GL=1 for gate-level runs (white-box checks and
state scrambling are skipped), TTPU_QUICK=1 for a reduced regression.
"""
import cocotb

import ttpu_model as m
from harness import (CFG, QUICK, is_gl, check, fresh, peek, play, rand_wmat, rand_xv, rng, run_prog,
                     save_result)

# ------------------------------------------------------------------------------------------------
# identification, status, configuration
# ------------------------------------------------------------------------------------------------
@cocotb.test()
async def test_id_info(dut):
    """ID/INFO bytes, protocol version, reset values of STATUS and tau; the same bytes through one read stream."""
    await fresh(dut)
    p = m.make_prog(CFG)
    p.read(0, m.ADDR_ID, m.ID_BYTE, "id")
    p.read(1, m.ADDR_INFO0, (4 << 5) | CFG.n, "info0 (DMAX, N)")
    p.read(2, m.ADDR_INFO1, CFG.accw, "info1 (ACCW)")
    p.read(3, m.ADDR_STATUS, m.ST_READY, "status after reset")
    p.read(4, m.ADDR_TAU, 0, "tau reset value")
    p.read(5, m.ADDR_PROTO, CFG.proto, "protocol version")
    for i, a in enumerate((0x06, 0x07, 0x0A, 0x0F)):
        p.read(6 + i, a, 0x00, f"reserved address 0x{a:02x} reads zero")
    p.nop(14)
    p.read_stream(p.tc, m.ADDR_ID, [m.ID_BYTE, (4 << 5) | CFG.n, CFG.accw, m.ST_READY, 0, CFG.proto], "id stream")
    await play(dut, p)


@cocotb.test()
async def test_cfg_tau_roundtrip(dut):
    """CFG writes tau; it reads back; WRST/NOP/other commands do not disturb it."""
    await fresh(dut)
    r = rng(2)
    p = m.make_prog(CFG)
    t = 0
    for tau in [0, 1, 7, 8, 127, 128, 254, 255] + [r.randrange(256) for _ in range(8)]:
        p.cfg_tau(tau)
        p.nop(2)
        p.wrst()
        p.nop(1)
        t = p.tc
        p.read(t, m.ADDR_TAU, tau, f"tau={tau}")
        p.nop(5)
    await play(dut, p)


# ------------------------------------------------------------------------------------------------
# weight loading
# ------------------------------------------------------------------------------------------------
@cocotb.test()
async def test_unpack_exhaustive(dut):
    """White-box: every data byte 0..255 and every field value 0..31 in every weight-group position decodes like the model."""
    if is_gl(dut):
        dut._log.info("skipped in gate-level run (white-box)")
        return
    await fresh(dut)
    ng = CFG.ng
    seen_b, seen_f = set(), set()
    for base in range(0, 256, ng):
        for rot in range(2):                               # two rotations: every (value, group) pair over the passes
            pairs = []
            for g in range(ng):
                b = (base + g + rot) % 256
                f = (base // ng * 3 + g * 5 + rot) % 32
                pairs.append((b, f))
            p = m.make_prog(CFG)
            p.wrst()
            p.load_groups(pairs)
            p.nop(10)
            await run_prog(dut, p)
            flat = [x for b, f in pairs for x in m.unpack_group(b, f)]
            got = peek(dut, "w_flat").value
            assert got.is_resolvable, f"w_flat has X after groups {pairs}"
            g = int(got)
            for pos, (r, c) in enumerate(m.stream_order(CFG.n)):
                code = (g >> (2 * (CFG.n * r + c))) & 3
                assert code == m.trit_code(flat[pos]), f"stream position {pos} (PE {r},{c}): code {code} expected {m.trit_code(flat[pos])} (groups {pairs})"
            seen_b.update(b for b, _ in pairs)
            seen_f.update(f for _, f in pairs)
    # every byte value and every field value was seen
    assert len(seen_b) == 256, len(seen_b)
    assert len(seen_f) == 32, len(seen_f)


@cocotb.test()
async def test_weight_mapping_blackbox(dut):
    """Black-box: read the whole tile back with N basis vectors; works on the gate-level netlist too."""
    await fresh(dut)
    n = CFG.n
    r = rng(4)
    for trial in range(1 if QUICK else 3):
        w = m.rand_tile(n, r, density=0.7)
        wmat = [[w[k][o] for k in range(n)] for o in range(n)]      # O x K
        p = m.make_prog(CFG)
        p.wrst()
        for k in range(n):
            e = [1 if i == k else 0 for i in range(n)]
            m.plan_gemv(CFG, wmat, e, 1, p=p, label=f"basis{k}", clear_margin=0)
        await play(dut, p, f"(trial {trial})")


# ------------------------------------------------------------------------------------------------
# fast mode (D = 1) and effort mode (D = 2..4)
# ------------------------------------------------------------------------------------------------
@cocotb.test()
async def test_fast_mode_random(dut):
    """D = 1 matrix-vector products with K accumulated over several tiles; raw and ternary readout."""
    await fresh(dut)
    n = CFG.n
    r = rng(5)
    p = m.make_prog(CFG)
    p.wrst()
    for trial in range(2 if QUICK else 6):
        o, k = n * r.randint(1, 3), n * r.randint(1, 5)
        wmat = rand_wmat(o, k, r, density=r.choice((1.0, 0.6)))
        x = rand_xv(k, 1, r)
        tau = r.choice((0, 1, 2, 3, 5, 8, 12))
        p.cfg_tau(tau)
        m.plan_gemv(CFG, wmat, x, 1, p=p, tau=tau, label=f"fast{trial}")
    await play(dut, p)


@cocotb.test()
async def test_effort_mode_random(dut):
    """D = 2, 3, 4 digit-serial activations (up to 81 levels) against the exact integer product."""
    await fresh(dut)
    n = CFG.n
    r = rng(6)
    p = m.make_prog(CFG)
    p.wrst()
    for d in (2, 3, 4) * (1 if QUICK else 2):
        o, k = n * r.randint(1, 2), n * r.randint(1, 4)
        wmat = rand_wmat(o, k, r)
        x = rand_xv(k, d, r, extreme=0.2)
        m.plan_gemv(CFG, wmat, x, d, p=p, label=f"effort_d{d}")
    await play(dut, p)


@cocotb.test()
async def test_accumulator_extremes(dut):
    """Largest legal sums: K = 768 with 4 digits (+-30720), signed accumulator edge, then wrap-around."""
    await fresh(dut)
    n = CFG.n
    p = m.make_prog(CFG)
    p.wrst()
    kmax = CFG.maxk(4)                                   # 819 -> use 768 (96 tiles)
    k = min(768, (kmax // n) * n)
    for sign_w, sign_x in ((+1, +1), (-1, +1), (+1, -1), (-1, -1)):
        wmat = [[sign_w] * k for _ in range(n)]
        x = [sign_x * m.digit_range(4)] * k
        m.plan_gemv(CFG, wmat, x, 4, p=p, label=f"extreme{sign_w}{sign_x}")
    if not QUICK:
        # deliberately overflow the signed 16-bit range: results wrap exactly like the model
        k2 = 1024
        wmat = [[1] * k2 for _ in range(n)]
        x = [m.digit_range(4)] * k2
        m.plan_gemv(CFG, wmat, x, 4, p=p, label="wrap")
    await play(dut, p)


@cocotb.test()
async def test_mixed_digit_counts(dut):
    """The digit count is chosen per vector: D = 1, 4, 2, 3 tiles mixed inside one accumulation block."""
    await fresh(dut)
    n = CFG.n
    r = rng(7)
    for trial in range(2 if QUICK else 5):
        tiles = []
        for d in r.sample([1, 2, 3, 4] * 2, 6):
            tiles.append((m.rand_tile(n, r), rand_xv(n, d, r, 0.15), d))
        p = m.make_prog(CFG)
        p.wrst()
        tau = r.choice((0, 3, 9))
        p.cfg_tau(tau)
        last, acc = m.plan_block(p, CFG, tiles)
        m.read_block(p, CFG, last, acc, tau=tau, label=f"mix{trial}")
        await play(dut, p, f"(trial {trial})")
