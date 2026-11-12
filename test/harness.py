"""cocotb helpers shared by the TTPU tests: clock/reset, program player, result checking and a
power-up state scrambler."""
from __future__ import annotations

import json
import os
import random
import sys
import zlib

import cocotb
from cocotb.clock import Clock
from cocotb.handle import ArrayObject, HierarchyArrayObject, HierarchyObject
from cocotb.triggers import FallingEdge, RisingEdge, Timer

import ttpu_model as m
import rtl_info

GL = os.environ.get("TTPU_GL", "0") == "1"          # gate-level run: no white-box access
N = int(os.environ.get("TTPU_N", "8"))
TREG = os.environ.get("TTPU_TREG", "0") == "1"
RPIPE = os.environ.get("TTPU_RPIPE", "0") == "1"
ACCW = int(os.environ.get("TTPU_ACCW", "16"))
DDR = os.environ.get("TTPU_DDR", "1") == "1"
FREEZE = os.environ.get("TTPU_FREEZE", "1") == "1"   # the released build has the output hold
CFG =m.Cfg(n=N, accw=ACCW, treg=TREG, rpipe=RPIPE, ddr=DDR)

# Names of the state elements (flip-flops) in src/project.v: every identifier that is the target of a non-blocking
# assignment, found by test/rtl_info.py. The scrambler writes random values to exactly these, i.e. it emulates the
# arbitrary state of a chip that has just powered up; a new register is picked up automatically.
STATE_NAMES = rtl_info.flop_names()


def is_gl(dut) -> bool:
    """True when the DUT is the gate-level netlist (no RTL signals to look at). Decided from the design itself, so that
    `make GATES=yes` works without any environment variable: the RTL has the internal wire `w_flat`, the netlist has not."""
    if GL or os.environ.get("GATES", "") == "yes":
        return True
    try:
        dut.user_project.w_flat
        return False
    except AttributeError:
        return True


async def start_clock(dut, period_ns: int = 20):
    cocotb.start_soon(Clock(dut.clk, period_ns, unit="ns").start())


def _checks(dut, on: bool):
    """Switch the RTL's simulation-only assertions (present when built with -DTTPU_ASSERT) on or off."""
    flag = getattr(dut.user_project, "chk_en", None)
    if flag is not None:
        flag.value = 1 if on else 0


async def reset(dut, cycles: int = 8, release_wait: int = 1):
    """Hold rst_n low for `cycles` clocks with idle pins, then release at a falling edge.

    release_wait = 0 returns at the release edge itself, so that a program started with
    run_prog(..., immediate=True) drives its first command in the very clock rst_n goes high."""
    dut.ena.value = 1
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.rst_n.value = 0
    for _ in range(cycles):
        await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    if cycles >= 3:                      # a documented reset: the state is reachable again, so the assertions apply
        _checks(dut, True)
    for _ in range(release_wait):
        await FallingEdge(dut.clk)


def _val(handle):
    v = handle.value
    return int(v) if v.is_resolvable else None


CLOCK_NS = 20


async def run_prog_ddr(dut, prog: m.Prog, tail: int = 6):
    """Play a program that has a lane-0 word in every clock (DDR driver). The chip samples lane 0 on the falling edge and
    lane 1 on the next rising edge; the pins change a quarter period after the edge before, so both are stable around
    their sampling edge. uo_out is sampled at the falling edge as in run_prog (cycle t = the clock of pair t)."""
    q = CLOCK_NS // 4
    last_exp = max((e.cycle for e in prog.exp), default=0)
    total = max(len(prog) + tail, last_exp + 2)
    prog._ensure(total)
    out = []
    await RisingEdge(dut.clk)                     # lane 0 of pair 0 must be on the pins before the first falling edge
    await Timer(q, unit="ns")
    dut.ui_in.value, dut.uio_in.value = prog.pins0(0)
    for t in range(total):
        await FallingEdge(dut.clk)
        out.append(_val(dut.uo_out))
        await Timer(q, unit="ns")
        dut.ui_in.value, dut.uio_in.value = prog.pins(t)
        await RisingEdge(dut.clk)
        await Timer(q, unit="ns")
        dut.ui_in.value, dut.uio_in.value = prog.pins0(t + 1)
    flag = getattr(dut.user_project, "assert_fail", None)
    if flag is not None:
        assert int(flag.value) == 0, "a simulation-only assertion of the RTL failed (see the 'TTPU ASSERTION' lines above)"
    return out


async def run_prog(dut, prog: m.Prog, tail: int = 6, immediate: bool = False, rst_release=None):
    """Play a program: at each falling edge sample uo_out (the value visible in this clock) and
    then drive the pins for this clock. Returns the list of sampled uo_out values (None = X/Z).
    immediate=True starts at the current falling edge instead of waiting for the next one.
    rst_release=k drives rst_n low from the start and raises it together with the pins of cycle k, so
    that commands can be applied while the chip is in reset and in the very clock reset is released."""
    if prog.ddr_stim and rst_release is None and not immediate:
        return await run_prog_ddr(dut, prog, tail)
    last_exp = max((e.cycle for e in prog.exp), default=0)
    total = max(len(prog) + tail, last_exp + 2)
    prog._ensure(total - 1)
    out = []
    if rst_release is not None:
        dut.rst_n.value = 0
    for t in range(total):
        if not (immediate and t == 0):
            await FallingEdge(dut.clk)
        if rst_release is not None and t == rst_release:
            dut.rst_n.value = 1
        out.append(_val(dut.uo_out))
        u, io = prog.pins(t)
        dut.ui_in.value = u
        dut.uio_in.value = io
    flag = getattr(dut.user_project, "assert_fail", None)      # present when the RTL is built with -DTTPU_ASSERT
    if flag is not None:
        assert int(flag.value) == 0, "a simulation-only assertion of the RTL failed (see the 'TTPU ASSERTION' lines above)"
    return out


def check_hold(prog: m.Prog, out, what: str = "", max_report: int = 12) -> int:
    """Output hold: in a clock whose field pins carried data (a W8 / XP* command outside a read stream) the read of
    that clock must not reach uo_out, so uo_out in cycle c + READ_LATENCY equals uo_out in the clock before. Two unknown values
    (nothing has been read since power-up) count as equal. Returns the number of clocks checked; raises AssertionError otherwise."""
    bad, n = [], 0
    for c in prog.frozen_cycles():
        t = c + m.READ_LATENCY
        if t >= len(out):
            continue
        n += 1
        if out[t] != out[t - 1]:
            bad.append((c, out[t - 1], out[t]))
    if bad:
        fmt = lambda v: "X" if v is None else f"0x{v:02x}"      # noqa: E731
        lines = [f"output hold violated in {len(bad)}/{n} data clocks {what}"]
        for c, a, b in bad[:max_report]:
            lines.append(f"  address slot cycle {c:6d} ({m.CMD_NAMES.get(prog.cmd[c], prog.cmd[c])}): uo_out went {fmt(a)} -> {fmt(b)} in cycle {c + m.READ_LATENCY}")
        raise AssertionError("\n".join(lines))
    return n


def check(prog: m.Prog, out, what: str = "", max_report: int = 12, hold: bool = True):
    """Compare sampled outputs with the program's expectations (and, for the release build, the output hold). Raises AssertionError on mismatch."""
    bad = []
    for e in prog.exp:
        got = out[e.cycle]
        if e.mask == 0:                  # (a read whose value is not part of the check: it may be unknown, e.g. the raw bytes of a datapath that was never written)
            continue
        if got is None or (got & e.mask) != (e.value & e.mask):
            bad.append((e, got))
    if bad:
        lines = [f"{len(bad)}/{len(prog.exp)} expectations failed {what}"]
        for e, got in bad[:max_report]:
            g = "X" if got is None else f"0x{got:02x}"
            lines.append(f"  cycle {e.cycle:6d} {e.label:28s} got {g} expected 0x{e.value & 0xFF:02x}")
        raise AssertionError("\n".join(lines))
    if hold and FREEZE:
        check_hold(prog, out, what, max_report)
    return len(prog.exp)


async def play(dut, prog: m.Prog, what: str = "", immediate: bool = False, rst_release=None):
    out = await run_prog(dut, prog, immediate=immediate, rst_release=rst_release)
    n = check(prog, out, what)
    return out, n


def scramble(dut, rng: random.Random) -> int:
    """Write random values into every state element of the DUT (RTL simulation only)."""
    count = 0

    def put(h):
        nonlocal count
        try:
            width = len(h)
            h.value = rng.getrandbits(width) if width > 1 else rng.getrandbits(1)
            count += 1
        except Exception:
            pass

    def rec(h):
        for c in h:
            if isinstance(c, (HierarchyObject, HierarchyArrayObject)):
                rec(c)
            elif c._name in STATE_NAMES:
                if isinstance(c, ArrayObject):
                    for e in c:
                        put(e)
                else:
                    put(c)

    rec(dut.user_project)
    _checks(dut, False)                  # the forced state is unreachable: the RTL's assertions describe reachable states
    return count


def peek(dut, path: str):
    """White-box read of a signal under tb.user_project (RTL only), e.g. 'u_array.p_w'."""
    h = dut.user_project
    for part in path.split("."):
        if "[" in part:
            name, idx = part[:-1].split("[")
            h = getattr(h, name)[int(idx)]
        else:
            h = getattr(h, part)
    return h


def plant(dut, path: str, value: int):
    """White-box write of a state element under tb.user_project (RTL only), same path syntax as peek; used after scramble()
    to put one chosen state into an otherwise random power-up state."""
    peek(dut, path).value = value


# ------------------------------------------------------------------------------------------------
# helpers shared by the test modules
# ------------------------------------------------------------------------------------------------
QUICK = os.environ.get("TTPU_QUICK", "0") == "1"
SEED = int(os.environ.get("TTPU_SEED", "1"))
RESULTS = os.environ.get("TTPU_RESULTS", "")      # optional json path for measured numbers


def rng(tag: int) -> random.Random:
    """Per-test reproducible random source (also seeds the junk bytes put on NOP cycles)."""
    r = random.Random(SEED * 1000 + tag)
    random.seed(SEED * 1000 + tag + 500)
    return r


async def fresh(dut, cycles: int = 8):
    # cocotb seeds `random` from the clock: without this the junk bytes that the programs put on NOP cycles would differ
    # from run to run, and so would which mutants a test happens to kill. A test that calls rng() later overrides this.
    random.seed(SEED * 1000 + zlib.crc32(sys._getframe(1).f_code.co_name.encode()) % 1000 + 700)
    await start_clock(dut)
    await reset(dut, cycles)


def rand_wmat(o, k, r, density=1.0):
    return [[(r.choice((-1, 1)) if r.random() < density else 0) for _ in range(k)] for _ in range(o)]


def rand_xv(k, d, r, extreme=0.0):
    return m.rand_x(k, d, r, extreme)


def save_result(key, value):
    """Merge one measured number into the results json."""
    if not RESULTS:
        return
    data = {}
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            data = json.load(f)
    data[key] = value
    with open(RESULTS, "w") as f:
        json.dump(data, f, indent=1)
