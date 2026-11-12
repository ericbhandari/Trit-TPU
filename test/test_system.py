# SPDX-License-Identifier: Apache-2.0
"""System-level tests: a full layer of the target model on the tile engine, and the clocks per tile that
the verified schedule achieves."""
import cocotb

import ttpu_model as m
from harness import (CFG, QUICK, check, fresh, rand_wmat, rand_xv, rng, run_prog, save_result)


@cocotb.test()
async def test_ffn_layer(dut):
    """The 192 -> 768 FFN matrix of the target model (2304 tiles) with 3-digit activations, end to end."""
    await fresh(dut)
    r = rng(16)
    o_dim, k_dim = (64, 64) if QUICK else (768, 192)
    d = 3
    wmat = rand_wmat(o_dim, k_dim, r, density=0.85)
    x = rand_xv(k_dim, d, r)
    p = m.make_prog(CFG)
    p.wrst()
    p, y, stats = m.plan_gemv(CFG, wmat, x, d, p=p, label="ffn")
    ref = [m.s_wrap(sum(wmat[o][k] * x[k] for k in range(k_dim)), 16) for o in range(o_dim)]
    assert ref == y, "planner expectation differs from the plain-python reference"
    out = await run_prog(dut, p)
    nchk = check(p, out, "(FFN layer)")
    cyc_per_tile = stats["cycles"] / stats["tiles"]
    dut._log.info(f"{stats['tiles']} tiles, {stats['cycles']} clocks, {cyc_per_tile:.2f} clocks/tile, {nchk} bytes checked")
    save_result(f"ffn_{o_dim}x{k_dim}_d{d}", dict(stats, clocks_per_tile=cyc_per_tile, bytes_checked=nchk))
    if CFG.n == 8:
        # the verified schedule is bus bound: ng W8 commands + d planes per tile, plus one clock between two output blocks
        blocks = o_dim // CFG.n
        expect = stats["tiles"] * (CFG.ng + d) + (blocks - 1)
        assert stats["cycles"] == expect, f"{stats['cycles']} clocks for {stats['tiles']} tiles, expected {expect} (tiles x (8 + D) + block gaps)"
        if not QUICK:
            assert stats["cycles"] == 25439, "the 2304-tile FFN layer must take 25 439 clocks"


@cocotb.test()
async def test_throughput_table(dut):
    """Clocks per tile of the verified schedule for D = 1..4. The steady-state period is the difference of the last plane of
    a 12-tile and of a 20-tile accumulation block, divided by the 8 extra tiles; it must be the bus bound, 8 + D clocks
    (9, 10, 11, 12)."""
    await fresh(dut)
    n = CFG.n
    r = rng(17)
    table = {}
    for d in (1, 2, 3, 4):
        last = {}
        for ntiles in (12, 20):
            wmat = rand_wmat(n, n * ntiles, r)
            x = rand_xv(n * ntiles, d, r)
            p = m.make_prog(CFG)
            p.wrst()
            p, y, stats = m.plan_gemv(CFG, wmat, x, d, p=p)
            out = await run_prog(dut, p)
            check(p, out, f"(throughput d={d}, {ntiles} tiles)")
            last[ntiles] = p.marks["block0_last_chunk"]
        period = (last[20] - last[12]) / 8
        table[f"d{d}"] = dict(tile_clocks=period, tiles=20, stream_clocks=last[20], first_tile_clocks=last[12] / 12)
        if CFG.n == 8:
            assert period == CFG.ng + d, f"steady-state period {period} clocks per tile at D={d}, expected {CFG.ng + d}"
    save_result("tile_clocks", table)
    dut._log.info(f"clocks per tile: {table}")
