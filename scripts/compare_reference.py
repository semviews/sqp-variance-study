#!/usr/bin/env python3
"""Compare the regenerated tables and macros in outputs/tables/ with reference/tables/.

reference/tables/ holds the tables and macros used for the submitted paper.
Tables are compared as files. For macros.tex each macro is compared by name,
and every macro whose value differs, or that exists on one side only, is
printed. Exits with status 1 if anything differs.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "tables"
CURRENT = ROOT / "outputs" / "tables"
MACRO = re.compile(r"^\\newcommand\{\\([A-Za-z@]+)\}\{(.*)\}\s*$")
CELL = re.compile(r"^\\expandafter\\def\\csname (cell@[^\\]+)\\endcsname\{(.*)\}\s*$")


def macros(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = MACRO.match(line) or CELL.match(line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def main() -> int:
    differing = 0
    tables = sorted(path.name for path in REFERENCE.glob("*.tex") if path.name != "macros.tex")
    for name in tables:
        current = CURRENT / name
        if not current.exists():
            print(f"missing   {name}")
            differing += 1
        elif current.read_bytes() != (REFERENCE / name).read_bytes():
            print(f"differs   {name}")
            differing += 1
    print(f"tables: {len(tables) - differing} of {len(tables)} identical")

    reference, current = macros(REFERENCE / "macros.tex"), macros(CURRENT / "macros.tex")
    changed = [name for name in sorted(reference.keys() | current.keys()) if reference.get(name) != current.get(name)]
    for name in changed:
        print(f"macro {name}: reference {reference.get(name, '(absent)')!r}, now {current.get(name, '(absent)')!r}")
    print(f"macros: {len(reference) - sum(name in reference for name in changed)} of {len(reference)} identical")
    return 1 if differing or changed else 0


if __name__ == "__main__":
    sys.exit(main())
