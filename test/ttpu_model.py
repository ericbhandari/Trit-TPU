"""Golden model and host-side driver library for the TTPU (ternary tile processing unit), protocol 3 (protocol 2 plus the DDR lane).

Everything the chip does is defined here twice: once as plain integer math (the golden model that
the tests compare against) and once as a cycle-level *program builder* that turns a high-level
request (load a tile, stream a vector, read results) into per-clock pin values.  The same program
can be played by cocotb (test.py) and against the gate-level netlist.

Pin protocol (see src/project.v for the full description):
    ui_in[7:0]   data byte
    uio_in[2:0]  command      uio_in[7:3]  field: 5 more data bits in W8 / XP* clocks, else the read address
    uo_out       registered read data, four clocks after the address; unchanged in the clock of a W8 / XP* address slot (output hold)

The chip takes eight ternary values per clock: a weight group (W8) or an activation plane (XP*) is 13 pin bits,
5 trits packed base-3 in the data byte and 3 trits packed base-3 in the 5-bit field.  The tile scheduler places the
commands of a tile so that the new weights follow right behind the wave of the previous tile (`schedule_tile`).
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from functools import lru_cache

# --------------------------------------------------------------------------------------------
# protocol constants
# --------------------------------------------------------------------------------------------
CMD_NOP, CMD_W8, CMD_WRST, CMD_CFG, CMD_XP, CMD_XPF, CMD_XPS, CMD_RDS = range(8)
T8_CMDS = (CMD_W8, CMD_XP, CMD_XPF, CMD_XPS)        # commands whose field pins carry data, not a read address
CMD_NAMES = {CMD_NOP: "NOP", CMD_W8: "W8", CMD_WRST: "WRST", CMD_CFG: "CFG", CMD_XP: "XP", CMD_XPF: "XPF", CMD_XPS: "XPS", CMD_RDS: "RDS"}

ADDR_ID, ADDR_INFO0, ADDR_INFO1, ADDR_STATUS, ADDR_TAU, ADDR_PROTO = 0x00, 0x01, 0x02, 0x03, 0x04, 0x05
ADDR_TERN0 = 0x08
ADDR_RAW0 = 0x10
ID_BYTE = 0x3E
PROTO_VERSION = 2

ST_READY, ST_ERR = 0x01, 0x04                       # status bit 1 is reserved (reads 0)
ST_DDR = 0x08                                       # status bit 3: DDR mode is on (DDR builds only)
DDR_ON, DDR_OFF = 0xD3, 0x5D                        # data byte of the WRST command that switches DDR mode on / off

READ_LATENCY = 4          # clocks from the read address on the pins to the data on uo_out
DMAX = 4                  # digit planes supported by the digit-scale tables
GS = 8                    # trits per W8 / XP* command
BIG = 10 ** 9


# --------------------------------------------------------------------------------------------
# number formats
# --------------------------------------------------------------------------------------------
def s_wrap(v: int, bits: int) -> int:
    """Two's complement wrap of an integer to a signed value of `bits` bits."""
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


def trit_code(t: int) -> int:
    """Ternary trit -> 2-bit pin code (00 = 0, 01 = +1, 11 = -1)."""
    return {0: 0b00, 1: 0b01, -1: 0b11}[t]


def pack_byte(trits) -> int:
    """Up to five trits (t0 first) -> base-3 byte, digit values 0,1,2 for trit 0,+1,-1."""
    assert len(trits) <= 5
    b = 0
    for i, t in enumerate(trits):
        b += {0: 0, 1: 1, -1: 2}[t] * 3 ** i
    return b


def unpack_byte(b: int):
    """Model of the on-chip unpacker. Illegal bytes (>242) give five zero trits."""
    if b > 242:
        return [0] * 5
    out = []
    for _ in range(5):
        d = b % 3
        b //= 3
        out.append({0: 0, 1: 1, 2: -1}[d])
    return out


def unpack_field(f: int):
    """The 5-bit field carries three trits (t5, t6, t7) packed base-3; illegal values (>26) give zeros."""
    if f > 26:
        return [0] * 3
    out = []
    for _ in range(3):
        d = f % 3
        f //= 3
        out.append({0: 0, 1: 1, 2: -1}[d])
    return out


def pack_group(trits):
    """Up to eight trits (t0 first) -> (data byte, field) of one W8 / XP* command."""
    assert len(trits) <= GS
    t = list(trits) + [0] * (GS - len(trits))
    return pack_byte(t[:5]), pack_byte(t[5:])


def unpack_group(b: int, f: int):
    return unpack_byte(b) + unpack_field(f)


def bad_group(b: int, f: int) -> bool:
    return b > 242 or f > 26


def wpos(n: int, r: int, c: int) -> int:
    """Position of PE (r, c) in the weight stream: diagonal order (r + c), then r. Same function as in the RTL."""
    d = r + c
    cnt = sum((k + 1) if k < n else (2 * n - 1 - k) for k in range(d))
    rlo = d - (n - 1) if d > n - 1 else 0
    return cnt + (r - rlo)


@lru_cache(maxsize=None)
def stream_order(n: int):
    """PE (r, c) of every stream position."""
    order = [None] * (n * n)
    for r in range(n):
        for c in range(n):
            order[wpos(n, r, c)] = (r, c)
    assert None not in order, "wpos is not a bijection"
    return tuple(order)


def groups_per_tile(n: int) -> int:
    return (n * n) // GS


def pack_tile(w, n: int | None = None):
    """Tile W[r][c] (row r = input index, column c = output index) -> list of N*N/8 (byte, field) pairs in stream order."""
    n = n or len(w)
    flat = [int(w[r][c]) for (r, c) in stream_order(n)]
    return [pack_group(flat[GS * g: GS * g + GS]) for g in range(groups_per_tile(n))]


def unpack_tile(groups, n: int):
    flat = []
    for b, f in groups:
        flat += unpack_group(b, f)
    w = [[0] * n for _ in range(n)]
    for pos, (r, c) in enumerate(stream_order(n)):
        w[r][c] = flat[pos]
    return w


def to_digits(x: int, d: int):
    """Integer -> d balanced-ternary digits, least significant first. Raises if out of range."""
    ds = []
    for _ in range(d):
        r = ((x + 1) % 3) - 1
        ds.append(r)
        x = (x - r) // 3
    if x != 0:
        raise ValueError("value does not fit in the requested number of digits")
    return ds


def from_digits(ds) -> int:
    return sum(dg * 3 ** i for i, dg in enumerate(ds))


def digit_range(d: int) -> int:
    """Largest magnitude representable with d balanced-ternary digits."""
    return (3 ** d - 1) // 2


def plane_word(plane):
    """One digit plane (N trits, row 0 first) -> (data byte, field) of an XP* command."""
    return pack_group(plane)


# --------------------------------------------------------------------------------------------
# chip parameters and timing rules derived from the architecture
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Cfg:
    n: int = 8
    accw: int = 16
    treg: bool = False              # single-clock PE (default) or PE with a registered product term (faster, larger)
    rpipe: bool = False             # digit-scale table pipelined in two stages (default: one stage)
    lx: int = 7                     # clocks from the pins of a plane command until the plane is in the array register
    lw: int = 7                     # clocks from the pins of a W8 command until the PE weights change
    ey: int = 1                     # column output register stage between the array and the accumulators
    seg: int = 4                    # bits per carry segment of the accumulators
    ddr: bool = True                # the build has the double-data-rate lane (parameter DDR of the RTL)

    @property
    def proto(self):                # protocol byte (read address 5)
        return 3 if self.ddr else PROTO_VERSION

    @property
    def pel(self):                  # clocks per PE hop of the partial sum
        return 2 if self.treg else 1

    @property
    def g(self):                    # ternary result bytes (4 columns each)
        return self.n // 4

    @property
    def ng(self):                   # 8-trit weight groups per tile
        return groups_per_tile(self.n)

    @property
    def delta(self):                # latency difference plane - weight
        return self.lx - self.lw

    @property
    def nseg(self):                 # carry segments per accumulator
        return self.accw // self.seg

    def cskew(self, c: int) -> int:
        """Clocks by which column c is later than column 0 (the column skew of the systolic array)."""
        return c

    @property
    def tsettle(self):              # clocks after the last plane until READY
        return 3 * self.n + self.nseg + self.pel + 8 + int(self.rpipe) + (self.lx - 3) + self.ey

    @property
    def rawbase(self):
        """High byte of column c is binary when its address is on the pins at t0 + rawbase + c (t0 = last plane):
        the column's last add is at the end of clock t0 + lx + n + pel + rpipe + ey + c, the carry then needs nseg - 1 more
        clocks to pass the segments, and the read samples the register two clocks after the address."""
        return self.lx + self.n + self.pel + int(self.rpipe) + self.ey + self.nseg - 2

    @property
    def raw_start(self):
        """First address time (relative to the last plane) of a read stream over the 2N raw bytes: byte c of the low bank
        needs t0 + rawbase - lo_early + c, byte c of the high bank (read n clocks later) t0 + rawbase + c."""
        return self.rawbase - min(self.lo_early, self.n)

    @property
    def lo_early(self):
        """Clocks by which the low byte of a column is binary before its high byte (carry segments between them)."""
        return (self.accw - 8) // self.seg

    @property
    def rcmax(self):
        o = stream_order(self.n)
        return tuple(max(r + self.cskew(c) for r, c in o[GS * g: GS * g + GS]) for g in range(self.ng))

    @property
    def rcmin(self):
        o = stream_order(self.n)
        return tuple(min(r + self.cskew(c) for r, c in o[GS * g: GS * g + GS]) for g in range(self.ng))

    def w_lower(self, last_plane: int, j: int) -> int:
        """Earliest clock for the W8 command of group j so that the new weights do not disturb the previous tile's wave
        (whose last plane command was in clock `last_plane`): the group lands lw clocks later and PE (r, c) reads its
        old weight until last_plane + lx + r + c."""
        return last_plane + self.delta + 1 + self.rcmax[j]

    def w_upper(self, first_plane: int, j: int) -> int:
        """Latest clock for the W8 command of group j so that the weights are in place when the tile's first plane
        (command in clock `first_plane`) reaches the first PE of the group."""
        return first_plane + self.delta + self.rcmin[j]

    def maxk(self, d: int) -> int:
        """Largest K for which a d-digit dot product cannot overflow the signed accumulator."""
        return ((1 << (self.accw - 1)) - 1) // digit_range(d)


def tern_out(acc: int, tau: int) -> int:
    """Ternarizer: +1 if acc > tau, -1 if acc < -tau, else 0."""
    return 1 if acc > tau else (-1 if acc < -tau else 0)


def read_map(cfg: Cfg, acc=None, tau: int = 0, ready: bool = True, err: bool = False, ddr_mode: bool = False):
    """The 32 bytes of the read port as the chip returns them when idle (`acc` = settled accumulators, or None when the
    raw and ternary bytes are not defined)."""
    mem = [0] * 32
    mem[ADDR_ID], mem[ADDR_INFO0], mem[ADDR_INFO1] = ID_BYTE, (4 << 5) | cfg.n, cfg.accw
    mem[ADDR_STATUS] = (ST_READY if ready else 0) | (ST_ERR if err else 0) | (ST_DDR if (ddr_mode and cfg.ddr) else 0)
    mem[ADDR_TAU], mem[ADDR_PROTO] = tau, cfg.proto
    if acc is not None:
        for q in range(cfg.g):
            byte = 0
            for j in range(4):
                byte |= trit_code(tern_out(acc[4 * q + j], tau)) << (2 * j)
            mem[ADDR_TERN0 + q] = byte
        for c in range(cfg.n):
            v = acc[c] & ((1 << cfg.accw) - 1)
            mem[ADDR_RAW0 + c], mem[ADDR_RAW0 + cfg.n + c] = v & 0xFF, (v >> 8) & 0xFF
    return mem


def block_result(tiles, cfg: Cfg):
    """Exact accumulator contents after streaming `tiles` = [(W, x_vector_ints)] into one block.

    The accumulator is 2^accw modular, so overflow wraps exactly as in hardware.
    """
    n = cfg.n
    acc = [0] * n
    for w, x in tiles:
        for c in range(n):
            acc[c] += sum(int(w[r][c]) * int(x[r]) for r in range(n))
    return [s_wrap(a, cfg.accw) for a in acc]


# --------------------------------------------------------------------------------------------
# program builder
# --------------------------------------------------------------------------------------------
@dataclass
class Expect:
    cycle: int           # cycle in which uo_out is sampled
    value: int
    label: str
    mask: int = 0xFF


@dataclass
class Prog:
    """Per-clock pin values plus expected read data.  Cycle t is the t-th rising clock edge."""
    cfg: Cfg = field(default_factory=Cfg)
    ui: list = field(default_factory=list)
    cmd: list = field(default_factory=list)
    addr: list = field(default_factory=list)      # read address on the field pins (in cycles that are not W8 / XP*)
    fld: list = field(default_factory=list)       # field value of W8 / XP* cycles
    exp: list = field(default_factory=list)
    marks: dict = field(default_factory=dict)
    hazard_start: int = 0          # earliest first-W8 cycle of the next block for a back-to-back tile load (see read_block)
    tc: int = 0                    # next free *command* cycle
    tern_hold: int = 0             # last cycle in which a ternary read still needs the current tau
    read_end: int = -1             # last cycle in which a read address is presented
    last_plane: int | None = None  # cycle of the last plane command (the wave of that plane is still in the array)
    first_plane: int | None = None  # cycle of the first plane (XPF / XPS) of the latest vector
    _addr_set: set = field(default_factory=set)
    _cover: set = field(default_factory=set)    # cycles whose field pins are ignored (inside a read stream)
    # DDR lane (lane 0: the word sampled on the falling edge before the rising edge of cycle t; lane 1 is the ordinary word)
    cmd0: list = field(default_factory=list)
    ui0: list = field(default_factory=list)
    fld0: list = field(default_factory=list)
    ddr_stim: bool = False         # the pin player drives a lane-0 word in every clock
    _ddr_floor: int | None = None  # earliest cycle the DDR scheduler may use (set by plan_block for the tiles of one block)
    ddr_on: bool = False           # DDR mode is on: the planner uses both lanes
    xlanes: tuple = (1, 0)         # lanes a plane may use in DDR mode (order = preference when both are free)
    lane_last: list = field(default_factory=lambda: [-1, -1])    # cycle of the last W8 command on lane 0 / lane 1

    # ---- low level ----
    def __len__(self):
        return len(self.ui)

    def _ensure(self, t: int):
        while len(self.ui) <= t:
            self.cmd0.append(CMD_NOP)
            self.ui0.append(0)
            self.fld0.append(0)
            self.ui.append(0)
            self.cmd.append(CMD_NOP)
            self.addr.append(self.addr[-1] if self.addr else 0)
            self.fld.append(0)

    def busy(self, t: int, cmd: int = CMD_NOP) -> bool:
        """Is cycle t unusable for `cmd`?  A command needs a free command slot; W8 / XP* also need the field pins, so
        they cannot share a cycle with a read address."""
        if t < len(self.cmd) and self.cmd[t] != CMD_NOP:
            return True
        if cmd == CMD_RDS:                  # a new stream would take over the address of a running one, or of a single read
            return t in self._cover or t in self._addr_set
        return cmd in T8_CMDS and t in self._addr_set

    def busy0(self, t: int) -> bool:
        """Is the lane-0 word of cycle t taken?"""
        return t < len(self.cmd0) and self.cmd0[t] != CMD_NOP

    def put0(self, t: int, cmd: int, data: int = 0, fld: int = 0):
        """A lane-0 word (DDR). Only W8 and the plane commands do anything there; anything else is ignored by the chip
        (the tests send such words on purpose). The field pins of a lane-0 word are always data, never a read address."""
        self._ensure(t)
        assert self.cmd0[t] == CMD_NOP, f"lane-0 collision at cycle {t}"
        self.cmd0[t], self.ui0[t], self.fld0[t] = cmd, data, fld
        self.ddr_stim = True
        self.tc = max(self.tc, t + 1)

    def pins0(self, t: int):
        """(ui_in, uio_in) of the lane-0 word sampled on the falling edge before rising edge t. The chip decodes lane 0 one
        clock after lane 1, so this word acts like a lane-1 word of cycle t + 1: the lane-0 arrays (cmd0, ui0, fld0, `put0`,
        the scheduler) are indexed by that effective cycle, and the pins carry them one slot earlier."""
        self._ensure(t + 1)
        return self.ui0[t + 1], (self.fld0[t + 1] << 3) | self.cmd0[t + 1]

    def scramble_lane0(self, rng, upto: int):
        """Random words (any command code) in every idle lane-0 slot before cycle `upto`: the chip must ignore lane 0 outside
        DDR mode."""
        self._ensure(upto)
        for t in range(upto):
            if self.cmd0[t] == CMD_NOP:
                self.cmd0[t], self.ui0[t], self.fld0[t] = rng.randrange(8), rng.getrandbits(8), rng.getrandbits(5)
        self.ddr_stim = True

    def ddr_enable(self, on: bool = True) -> int:
        """WRST with the mode byte; the mode applies to the pairs from the second clock after the command. Returns its cycle."""
        t = self.emit(CMD_WRST, DDR_ON if on else DDR_OFF)
        self.idle_until(t + 2)
        self.ddr_on = on
        self.ddr_stim = True
        self.lane_last = [self.tc - 1, self.tc - 1]
        return t

    def put(self, t: int, cmd: int, data: int = 0, fld: int = 0):
        self._ensure(t)
        assert self.cmd[t] == CMD_NOP, f"command collision at cycle {t}"
        assert not (cmd in T8_CMDS and t in self._addr_set), f"command {cmd} at cycle {t} collides with a read address"
        assert not (cmd == CMD_RDS and (t in self._cover or t in self._addr_set)), f"RDS at cycle {t} collides with a running read stream"
        self.cmd[t], self.ui[t], self.fld[t] = cmd, data, fld
        self.tc = max(self.tc, t + 1) if cmd != CMD_RDS else self.tc

    def emit(self, cmd: int, data: int = 0, fld: int = 0) -> int:
        """Append one command at the next free command cycle (skipping cycles that are already taken); returns it."""
        t = self.tc
        while self.busy(t, cmd):
            t += 1
        self.put(t, cmd, data, fld)
        self.tc = t + 1
        return t

    def nop(self, n: int = 1):
        for _ in range(n):
            self.emit(CMD_NOP, random.getrandbits(8))   # NOPs carry junk data on purpose

    def idle_until(self, t: int):
        while self.tc < t:
            self.emit(CMD_NOP, random.getrandbits(8))

    def set_addr(self, t: int, a: int):
        self._ensure(t)
        self.addr[t] = a
        # keep holding that address until another read (or a stream) overrides it
        for u in range(t + 1, len(self)):
            if u in self._addr_set or u in self._cover:
                break
            self.addr[u] = a
        self._addr_set.add(t)

    def read(self, ta: int, a: int, value: int, label: str, mask: int = 0xFF):
        """Present address `a` in cycle ta, expect `value` in cycle ta + READ_LATENCY.
        Two different reads in the same clock are a program error (one address bus); so is a read in a clock whose field
        pins carry W8 / XP* data or that lies inside a read stream."""
        self._ensure(ta)
        if self.cmd[ta] in T8_CMDS:
            raise ValueError(f"read at cycle {ta} ({label}) collides with a W8/XP* command: the field pins carry data")
        if ta in self._cover:
            raise ValueError(f"read at cycle {ta} ({label}) lies inside a read stream")
        if self.cmd[ta] == CMD_RDS:
            raise ValueError(f"read at cycle {ta} ({label}) collides with the RDS command of a read stream")
        if ta in self._addr_set and self.addr[ta] != a:
            raise ValueError(f"read address collision at cycle {ta}: {self.addr[ta]:#x} vs {a:#x} ({label})")
        self.read_end = max(self.read_end, ta)
        self.set_addr(ta, a)
        if value is None:
            value, mask = 0, 0
        self.exp.append(Expect(ta + READ_LATENCY, value, label, mask))

    def read_stream(self, ta: int, a: int, values, label: str, mask: int = 0xFF):
        """Read len(values) consecutive addresses a, a+1, ... with one RDS command in cycle ta: the first address is
        read as usual, the following ones one per clock without anything on the field pins.  Expect values[i] in cycle
        ta + i + READ_LATENCY."""
        count = len(values) - 1
        assert 0 <= count <= 63
        self._ensure(ta + count)
        if self.cmd[ta] != CMD_NOP or ta in self._addr_set or ta in self._cover:
            raise ValueError(f"RDS at cycle {ta} ({label}): the cycle is taken")
        for u in range(ta + 1, ta + count + 1):
            if u in self._addr_set or u in self._cover or self.cmd[u] == CMD_RDS:
                raise ValueError(f"read stream {label} overlaps another read at cycle {u}")
        self.put(ta, CMD_RDS, count)
        self.addr[ta] = a
        self._addr_set.add(ta)
        for u in range(ta + 1, ta + count + 1):
            self._cover.add(u)
            self.addr[u] = random.getrandbits(5)        # the pins must be ignored while the stream runs
        self.read_end = max(self.read_end, ta + count)
        for i, v in enumerate(values):
            self.exp.append(Expect(ta + i + READ_LATENCY, 0 if v is None else v, f"{label}[{i}]", 0 if v is None else mask))

    # ---- commands ----
    def wrst(self):
        return self.emit(CMD_WRST)

    def cfg_tau(self, tau: int):
        """Write tau. The ternarizer follows tau continuously (tau reaches uo_out 3 clocks after the
        command), so the write must not precede the address of a pending ternary read."""
        self.idle_until(self.tern_hold)
        return self.emit(CMD_CFG, tau)

    def w8(self, group, t: int | None = None, lane: int = 1) -> int:
        """One W8 command: group = (data byte, field). lane 0 is the DDR lane."""
        b, f = group
        if t is not None:
            self.lane_last[lane] = max(self.lane_last[lane], t)
        if lane == 0:
            self.put0(t, CMD_W8, b, f)
            return t
        if t is None:
            t = self.emit(CMD_W8, b, f)
            self.lane_last[1] = max(self.lane_last[1], t)
            return t
        self.put(t, CMD_W8, b, f)
        return t

    def load_tile(self, w, t0: int | None = None, margin: int = 0):
        """Write a packed tile back to back (N*N/8 W8 commands), starting no earlier than t0 and no earlier than the
        previous tile's wave allows (`margin` shifts that bound: negative values violate it). Returns (first, last)."""
        if t0 is not None:
            self.idle_until(t0)
        if self.wref() is not None:
            cfg = self.cfg
            need = max(cfg.w_lower(self.wref(), j) - j for j in range(cfg.ng)) + margin
            self.idle_until(need)
        return self.load_groups(pack_tile(w, self.cfg.n))

    def load_groups(self, groups, t0: int | None = None):
        if t0 is not None:
            self.idle_until(t0)
        first = last = None
        for grp in groups:
            last = self.w8(grp)
            first = last if first is None else first
        return first, last

    def plane(self, word, kind: int, t: int | None = None, lane: int = 1) -> int:
        """One XP / XPF / XPS command; remembers its cycle as the last plane."""
        assert kind in (CMD_XP, CMD_XPF, CMD_XPS)
        b, f = word
        if lane == 0:
            self.put0(t, kind, b, f)
        elif t is None:
            t = self.emit(kind, b, f)
        else:
            self.put(t, kind, b, f)
        self.last_plane = t
        if kind != CMD_XP:
            self.first_plane = t
        return t

    def wref(self):
        """Reference clock of the lower bound of the next tile's W8 commands: the last plane of the previous tile."""
        return self.last_plane

    def vector(self, x, d: int, clear: bool, t0: int | None = None, gaps=None):
        """Stream one activation vector of d digit planes (least significant digit first), back to back.
        Returns the cycle of the last plane.

        clear -> first plane is XPS (accumulators restart), else XPF. `gaps` optionally inserts NOP cycles
        before plane number i (dict i -> count).
        """
        if t0 is not None:
            self.idle_until(t0)
        planes = [[to_digits(int(v), d)[k] for v in x] for k in range(d)]
        last = None
        for k in range(d):
            if gaps and k in gaps:
                self.nop(gaps[k])
            kind = CMD_XPS if (k == 0 and clear) else (CMD_XPF if k == 0 else CMD_XP)
            last = self.plane(plane_word(planes[k]), kind)
        return last

    # ---- result readout ----
    def raw_values(self, expect_acc):
        """The 2N raw bytes in address order: the low bytes of columns 0..N-1, then the high bytes."""
        m_ = (1 << self.cfg.accw) - 1
        lo = [(expect_acc[c] & m_) & 0xFF for c in range(self.cfg.n)]
        hi = [((expect_acc[c] & m_) >> 8) & 0xFF for c in range(self.cfg.n)]
        return lo + hi

    def tern_values(self, expect_acc, tau: int):
        vals = []
        for q in range(self.cfg.g):
            byte = 0
            for j in range(4):
                byte |= trit_code(tern_out(expect_acc[4 * q + j], tau)) << (2 * j)
            vals.append(byte)
        return vals

    def read_raw(self, ta: int, expect_acc, label: str = "raw", stream: bool = True):
        """Read all columns (low byte then high byte). One RDS command at ta (default) or one address per clock."""
        vals = self.raw_values(expect_acc)
        if stream:
            self.read_stream(ta, ADDR_RAW0, vals, label)
        else:
            for i, v in enumerate(vals):
                self.read(ta + i, ADDR_RAW0 + i, v, f"{label}[{i % self.cfg.n}].{'lo' if i < self.cfg.n else 'hi'}")
        return ta + len(vals)

    def read_tern(self, ta: int, expect_acc, tau: int, label: str = "tern", stream: bool = True):
        vals = self.tern_values(expect_acc, tau)
        if stream:
            self.read_stream(ta, ADDR_TERN0, vals, label)
        else:
            for q, v in enumerate(vals):
                self.read(ta + q, ADDR_TERN0 + q, v, f"{label}[{q}]")
        self.tern_hold = max(self.tern_hold, ta + len(vals) - 1)
        return ta + len(vals)

    def read_status(self, ta: int, value: int, label: str = "status", mask: int = 0xFF):
        self.read(ta, ADDR_STATUS, value, label, mask)

    def frozen_cycles(self, first: int = 3):
        """Cycles whose field pins carry data, not a read address: a W8 / XP* command outside a read stream (inside a stream the
        address is the running one and the byte is wanted whatever the pins carry). The chip does not let the read of such a clock
        reach uo_out: uo_out in cycle c + READ_LATENCY must equal uo_out in the clock before (the output hold).
        Cycles before `first` are left out: the read pipeline and the stream counter there may still hold the state of whatever ran
        before the program (a power-up state that was not reset).
        The streams are taken from the RDS commands in the program (DATA[5:0] = n: the next n clocks are inside the stream, and a new
        RDS takes the stream over), not from the bookkeeping of read_stream(), so that a bare emit(CMD_RDS, n) counts as well. An RDS
        that the chip ignores (in reset) only makes this list shorter."""
        out, remaining = [], 0
        for c in range(len(self.cmd)):
            if self.cmd[c] == CMD_RDS:
                remaining = self.ui[c] & 0x3F
                continue
            in_stream = remaining > 0
            if in_stream:
                remaining -= 1
            if c >= first and self.cmd[c] in T8_CMDS and not in_stream:
                out.append(c)
        return out

    # ---- pin values per clock ----
    def pins(self, t: int):
        """(ui_in, uio_in) of cycle t."""
        c = self.cmd[t]
        a = self.fld[t] if c in T8_CMDS else self.addr[t]
        return self.ui[t], (a << 3) | c


def make_prog(cfg: Cfg) -> Prog:
    return Prog(cfg=cfg)


# --------------------------------------------------------------------------------------------
# tile scheduler
# --------------------------------------------------------------------------------------------
_SCHED_CACHE: dict = {}
DDR_OBJ = (0, 1, 2)       # order in which the DDR scheduler compares (end of the tile, last plane, sum of the command clocks)


def schedule_tile(p: Prog, cfg: Cfg, d: int, wmargin: int = 0, xmargin: int = 0, xmin: int | None = None,
                  tstart: int | None = None):
    """Place the N*N/8 W8 commands and the d plane commands of one tile on free command cycles.

    Rules: group j's W8 may be issued from  last_plane + delta + 1 + rcmax[j]  (the previous tile's wave has
    left its PEs) and must be issued by  first_plane + delta + rcmin[j]  (the weights are there when the new wave arrives);
    the W8 commands keep their order (write pointer), planes keep theirs; one command per cycle. The schedule that
    finishes earliest wins (ties: the one whose last plane is earlier, which relaxes the next tile's bounds).
    wmargin / xmargin shift the two bounds (negative wmargin / positive xmargin place commands where the rule forbids
    them: used by the tests that show both bounds are tight).
    Returns (w_times, plane_times)."""
    ng = cfg.ng
    t0 = p.tc if tstart is None else max(p.tc, tstart)
    last = p.wref()
    lo = [(-BIG if last is None else cfg.w_lower(last, j) + wmargin) for j in range(ng)]
    horizon = max(t0, max(lo), xmin or 0) + ng + d + 10
    busy = tuple(p.busy(t, CMD_W8) for t in range(t0, horizon + 1))
    key = (cfg.n, cfg.delta, d, wmargin, xmargin, tuple(max(l - t0, -1) for l in lo), busy,
           None if xmin is None else max(xmin - t0, 0))
    plan = _SCHED_CACHE.get(key)
    if plan is None:
        rcmin = cfg.rcmin
        lo_r = [max(l - t0, 0) for l in lo]
        xmin_r = 0 if xmin is None else max(xmin - t0, 0)
        hz = horizon - t0
        memo = {}

        def f(s, j, q, b0):
            if j == ng and q == d:
                return (s, 0), None
            if s > hz:
                return (BIG, BIG), None
            k = (s, j, q, b0)
            if k in memo:
                return memo[k]
            best, act = (BIG, BIG), None
            if busy[s]:
                best, _ = f(s + 1, j, q, b0)
                memo[k] = (best, "I")
                return memo[k]
            if j < ng and s >= lo_r[j] and (b0 is None or s <= b0 + cfg.delta + rcmin[j] + xmargin):
                c, _ = f(s + 1, j + 1, q, b0)
                if c < best:
                    best, act = c, "W"
            if q < d and (q > 0 or s >= xmin_r):
                c, _ = f(s + 1, j, q + 1, s if q == 0 else b0)
                if q == d - 1 and c[0] < BIG:
                    c = (c[0], s)
                if c < best:
                    best, act = c, "X"
            c, _ = f(s + 1, j, q, b0)
            if c < best:
                best, act = c, "I"
            memo[k] = (best, act)
            return memo[k]

        total, _ = f(0, 0, 0, None)
        assert total[0] < BIG, "no feasible schedule for the tile"
        ws, xs = [], []
        s, j, q, b0 = 0, 0, 0, None
        while not (j == ng and q == d):
            act = f(s, j, q, b0)[1]
            if act == "W":
                ws.append(s)
                j += 1
            elif act == "X":
                xs.append(s)
                if q == 0:
                    b0 = s
                q += 1
            s += 1
        plan = (tuple(ws), tuple(xs))
        _SCHED_CACHE[key] = plan
    return [t0 + a for a in plan[0]], [t0 + b for b in plan[1]]


def schedule_tile_ddr(p: Prog, cfg: Cfg, d: int, wmargin: int = 0, xmargin: int = 0, xmin: int | None = None,
                      tstart: int | None = None):
    """The same as `schedule_tile` for DDR mode: every clock has two command slots. The first half of the weight groups goes
    through lane 0 in order, the second half through lane 1 in order (each lane has its own pointer ring, so the two
    sequences are independent); a plane may use either lane but at most one plane per clock (the array takes one plane per
    clock). The window rules for every group are those of `schedule_tile`; a tile may start while the other lane is still
    busy with the previous tile. Returns (ws, xs): [(clock, lane)] per group and per plane."""
    ng = cfg.ng
    nge, ngo = ng // 2, ng - ng // 2          # groups on lane 0, groups on lane 1
    base = p._ddr_floor if p._ddr_floor is not None else (p.tc if tstart is None else max(p.tc, tstart))
    t0 = max(base, min(p.lane_last) + 1)
    if p.last_plane is not None:
        t0 = max(t0, p.last_plane + 1)
    lane_lo = [max(p.lane_last[ln] + 1, t0) - t0 for ln in (0, 1)]
    last = p.wref()
    lo = [(-BIG if last is None else cfg.w_lower(last, j) + wmargin) for j in range(ng)]
    horizon = max(t0, max(lo), xmin or 0) + ng + d + 10
    b1 = tuple(p.busy(t, CMD_W8) for t in range(t0, horizon + 1))
    b0 = tuple(p.busy0(t) for t in range(t0, horizon + 1))
    key = ("ddr", cfg.n, cfg.delta, d, wmargin, xmargin, tuple(max(l - t0, -1) for l in lo), b0, b1,
           None if xmin is None else max(xmin - t0, 0), p.xlanes, tuple(lane_lo), DDR_OBJ)
    plan = _SCHED_CACHE.get(key)
    if plan is None:
        rcmin = cfg.rcmin
        lo_r = [max(l - t0, 0) for l in lo]
        xmin_r = 0 if xmin is None else max(xmin - t0, 0)
        hz = horizon - t0
        memo = {}
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 4000))

        def f(s, je, jo, q, bb):
            if je == nge and jo == ngo and q == d:
                return (s, 0, 0), None
            if s > hz:
                return (BIG, BIG, BIG), None
            k = (s, je, jo, q, bb)
            if k in memo:
                return memo[k]
            best, act = (BIG, BIG, BIG), None
            o0 = ["I"] + ([] if b0[s] else ["W"] + (["X"] if 0 in p.xlanes else []))
            o1 = ["I"] + ([] if b1[s] else ["W"] + (["X"] if 1 in p.xlanes else []))
            for a0 in o0:
                for a1 in o1:
                    nx = (a0 == "X") + (a1 == "X")
                    if nx > 1 or (nx and not (q < d and (q > 0 or s >= xmin_r))):
                        continue
                    qn = q + nx
                    bn = (s if q == 0 else bb) if nx else bb
                    jen, jon = je, jo
                    ok = True
                    for act_, j, ln in ((a0, je, 0), (a1, nge + jo, 1)):
                        if act_ == "W":
                            if (ln == 0 and je >= nge) or (ln == 1 and jo >= ngo) or s < lane_lo[ln] or s < lo_r[j] or \
                                    (bn is not None and s > bn + cfg.delta + rcmin[j] + xmargin):
                                ok = False
                    if not ok:
                        continue
                    if a0 == "W":
                        jen = je + 1
                    if a1 == "W":
                        jon = jo + 1
                    c, _ = f(s + 1, jen, jon, qn, bn)
                    if c[0] >= BIG:
                        continue
                    c = (c[0], s if (nx and qn == d) else c[1], c[2] + s * ((a0 != "I") + (a1 != "I")))     # third: earlier is better
                    if tuple(c[i] for i in DDR_OBJ) < tuple(best[i] for i in DDR_OBJ):
                        best, act = c, (a0, a1)
            memo[k] = (best, act)
            return memo[k]

        total, _ = f(0, 0, 0, 0, None)
        assert total[0] < BIG, "no feasible schedule for the tile"
        ws, xs = [None] * ng, []
        s, je, jo, q, bb = 0, 0, 0, 0, None
        while not (je == nge and jo == ngo and q == d):
            a0, a1 = f(s, je, jo, q, bb)[1]
            nx = (a0 == "X") + (a1 == "X")
            if nx:
                xs.append((s, 0 if a0 == "X" else 1))
                if q == 0:
                    bb = s
                q += 1
            if a0 == "W":
                ws[je] = (s, 0)
                je += 1
            if a1 == "W":
                ws[nge + jo] = (s, 1)
                jo += 1
            s += 1
        plan = (tuple(ws), tuple(xs))
        _SCHED_CACHE[key] = plan
    return [(t0 + a, ln) for a, ln in plan[0]], [(t0 + a, ln) for a, ln in plan[1]]


# --------------------------------------------------------------------------------------------
# GEMV planner: y = Wmat @ x on the NxN tile engine
# --------------------------------------------------------------------------------------------
def tile_of(wmat, ob: int, kb: int, n: int):
    """Tile for output block ob, input block kb: PE(r,c) holds Wmat[ob*n + c][kb*n + r]."""
    return [[int(wmat[ob * n + c][kb * n + r]) for c in range(n)] for r in range(n)]


def min_next_first_plane(cfg: Cfg, ta_raw: int) -> int:
    """Earliest cycle for the first plane (XPS) of the next output block so that its clear reaches column c only after the
    raw bytes of column c were read.

    Column c is sampled by the read mux at ta_raw + 2c + 3 at the latest (high byte); the clear of column c takes effect at
    the end of clock X + lx + n + pel + rpipe + ey + c for an XPS in clock X. The binding column is c = n - 1:
        X >= ta_raw + 2 - lx - pel - rpipe - ey.
    """
    return ta_raw + 2 - cfg.lx - cfg.pel - int(cfg.rpipe) - cfg.ey + (cfg.n - 1 - cfg.cskew(cfg.n - 1))


def read_block(p: Prog, cfg: Cfg, last_plane: int, acc, tau: int | None = None, label: str = "y",
               ready_delay: int = 0, clear_margin: int = 0, early: bool | None = None):
    """Schedule the readout of a finished output block (reads go through RDS streams).

    Raw only (tau None): the first byte is read as soon as the first column is binary (`early`, t0 + raw_start);
    with a ternary read first the READY time t0 + 1 + tsettle is used for the ternary bytes and the raw bytes follow.
    Also records, in p.hazard_start, how early the next block may start (clear_margin shifts that bound; negative values are
    used by the boundary tests). Returns the first raw read cycle.
    """
    if early is None:
        early = tau is None
    ta = last_plane + (cfg.raw_start if early else 1 + cfg.tsettle) + ready_delay
    while p.busy(ta, CMD_RDS) or ta in p._cover or ta in p._addr_set:
        ta += 1
    if tau is not None:
        ta2 = p.read_tern(ta, acc, tau, f"{label}.tern")
        ta = ta2
        while p.busy(ta, CMD_RDS) or ta in p._cover or ta in p._addr_set:
            ta += 1
    p.read_raw(ta, acc, f"{label}.raw")
    p.hazard_start = max(p.hazard_start, min_next_first_plane(cfg, ta) + clear_margin - cfg.ng)
    return ta


def plan_block(p: Prog, cfg: Cfg, tiles, start: int | None = None, wgt_margin: int = 0, xp_margin: int = 0):
    """Stream one output block. tiles = [(W, x, d)]. Returns (last_plane_cycle, expected accumulators).

    Each tile's commands go where `schedule_tile` puts them: weights right behind the previous wave, the first plane of
    the block no earlier than the clear rule allows (p.hazard_start + ng)."""
    last = None
    for i, (w, x, d) in enumerate(tiles):
        groups = pack_tile(w, cfg.n)
        planes = [plane_word([to_digits(int(v), d)[k] for v in x]) for k in range(d)]
        xmin = (p.hazard_start + cfg.ng) if (i == 0 and p.hazard_start > 0) else None
        if p.ddr_on:
            if i == 0:
                p._ddr_floor = p.tc if start is None else max(p.tc, start)
            ws, xs = schedule_tile_ddr(p, cfg, d, wgt_margin, xp_margin, xmin=xmin, tstart=start if i == 0 else None)
            for grp, (t, ln) in zip(groups, ws):
                p.w8(grp, t, ln)
            for k, (word, (t, ln)) in enumerate(zip(planes, xs)):
                kind = (CMD_XPS if i == 0 else CMD_XPF) if k == 0 else CMD_XP
                p.plane(word, kind, t, ln)
            p.tc = max(p.tc, max(t for t, _ in ws + xs) + 1)
            last = xs[-1][0]
        else:
            ws, xs = schedule_tile(p, cfg, d, wgt_margin, xp_margin, xmin=xmin, tstart=start if i == 0 else None)
            for grp, t in zip(groups, ws):
                p.w8(grp, t)
            for k, (word, t) in enumerate(zip(planes, xs)):
                kind = (CMD_XPS if i == 0 else CMD_XPF) if k == 0 else CMD_XP
                p.plane(word, kind, t)
            p.tc = max(p.tc, max(ws[-1], xs[-1]) + 1)
            last = xs[-1]
    return last, block_result([(w, x) for w, x, _ in tiles], cfg)


def plan_gemv(cfg: Cfg, wmat, x, d: int, p: Prog | None = None, tau: int | None = None,
              label: str = "y", clear_margin: int = 0, wgt_margin: int = 0):
    """Append a full matrix-vector product y = Wmat @ x to the program and register the expected reads.

    wmat is O x K (ternary), x is a length-K list of ints that fit in d balanced-ternary digits.
    Returns (prog, expected y, stats dict).
    """
    n = cfg.n
    o_dim, k_dim = len(wmat), len(wmat[0])
    assert o_dim % n == 0 and k_dim % n == 0
    p = p or make_prog(cfg)
    y_all, n_tiles, t_start = [], 0, p.tc
    for ob in range(o_dim // n):
        tiles = [(tile_of(wmat, ob, kb, n), [int(v) for v in x[kb * n:(kb + 1) * n]], d)
                 for kb in range(k_dim // n)]
        last, acc = plan_block(p, cfg, tiles, wgt_margin=wgt_margin)
        y_all += acc
        n_tiles += len(tiles)
        read_block(p, cfg, last, acc, tau, f"{label}[ob{ob}]", clear_margin=clear_margin)
        p.marks[f"block{ob}_last_chunk"] = last
    stats = dict(tiles=n_tiles, cycles=p.tc - t_start, d=d)
    return p, y_all, stats


# --------------------------------------------------------------------------------------------
# random helpers
# --------------------------------------------------------------------------------------------
def rand_tile(n: int, rng: random.Random, density: float = 1.0):
    return [[(rng.choice((-1, 1)) if rng.random() < density else 0) for _ in range(n)] for _ in range(n)]


def rand_x(n: int, d: int, rng: random.Random, extreme: float = 0.0):
    m = digit_range(d)
    return [(rng.choice((-m, m)) if rng.random() < extreme else rng.randint(-m, m)) for _ in range(n)]
