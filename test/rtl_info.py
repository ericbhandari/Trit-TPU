"""Static look at src/project.v: which registers are flip-flops and which `reg` variables are combinational.

A flip-flop is any identifier that is the target of a non-blocking assignment (`x <= ...;`): in this design those
only occur in clocked blocks. Everything else declared `reg` is assigned with blocking assignments in
combinational blocks. The power-up scrambler (test/harness.py)
takes its register list from here, so a flip-flop added to the RTL is picked up automatically, and
test_robust.test_state_list_complete fails if a `reg` can be classified neither way.
"""
import os
import re
from pathlib import Path

SRC = Path(os.environ.get("TTPU_SRC") or Path(__file__).resolve().parent.parent / "src" / "project.v")
# registers that exist only in the `ifdef TTPU_ASSERT` block: no hardware, never scrambled
SIM_ONLY = {"assert_fail", "chk_armed", "chk_en"}
_STOP = {"input", "output", "inout", "wire", "reg", "parameter", "localparam", "integer", "genvar", "assign",
         "always", "initial", "function", "task", "generate", "endgenerate", "begin", "end", "if", "else", "for"}


def _clean(path):
    text = Path(path).read_text()
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def flop_names(path=SRC):
    """Identifiers that are targets of a non-blocking assignment statement."""
    text = _clean(path)
    pat = re.compile(r"(?:^|[;{}]|\bbegin\b|\bend\b|\belse\b|\))\s*([A-Za-z_]\w*)\s*(?:\[[^;=]*?\]\s*)*<=(?!=)")
    return {m.group(1) for m in pat.finditer(text)} - _STOP - SIM_ONLY


def reg_decls(path=SRC):
    """Names of all variables declared `reg` (module-level, generate blocks, `output reg` ports)."""
    text = _clean(path)
    names = set()
    for m in re.finditer(r"\breg\b", text):
        i = m.end()
        # optional signed, optional packed range
        mm = re.match(r"\s*(?:signed\s*)?(?:\[[^\]]*\]\s*)?", text[i:])
        i += mm.end()
        while True:
            mi = re.match(r"\s*([A-Za-z_]\w*)", text[i:])
            if not mi or mi.group(1) in _STOP:
                break
            names.add(mi.group(1))
            i += mi.end()
            ma = re.match(r"(?:\s*\[[^\]]*\])*(?:\s*=\s*[^,;)]*)?", text[i:])      # array dimensions, initial value
            i += ma.end()
            mc = re.match(r"\s*,", text[i:])
            if not mc:
                break
            i += mc.end()
    return names - SIM_ONLY


def comb_reg_names(path=SRC):
    """`reg` variables that are not flip-flops (assigned with blocking assignments in `always @(*)` blocks)."""
    return reg_decls(path) - flop_names(path)


if __name__ == "__main__":
    f, d = flop_names(), reg_decls()
    print(f"{len(f)} flip-flop names, {len(d - f)} combinational regs: {sorted(d - f)}")
    print("flip-flops:", sorted(f))
