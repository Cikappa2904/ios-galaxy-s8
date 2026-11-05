#!/usr/bin/env python3
"""Add the Galaxy S8+ (dream2lte) STMicro FTS8CD56 touch driver to an xnu
tree that already has 0124-dreamlte-s8.patch (prep-xnu-tree.sh output).

dream2lte_fts.c (next to this script) goes in front of
dreamlte_touch_start(), which falls back to it when no S6SY761 answers at
0x48. Export the result as 0140-dream2lte-fts-touch.patch.
Usage: apply-dream2lte-fts.py XNU_TREE
"""
import sys
from pathlib import Path

tree = Path(sys.argv[1])
path = tree / 'bsd/dev/a5xelte_touch.c'
text = path.read_text()
fts = (Path(__file__).parent / 'dream2lte_fts.c').read_text()

edits = [
    ('static void\ndreamlte_touch_start(void)\n{\n',
     fts + 'static void\ndreamlte_touch_start(void)\n{\n'),
    ('\t\tprintf("S8 TOUCH: no S6SY761 in application mode (status %#x)\\n", b[0]);\n\t\treturn;\n',
     '\t\tprintf("S8 TOUCH: no S6SY761 in application mode (status %#x)\\n", b[0]);\n'
     '\t\t(void)s8fts_start();      /* Galaxy S8+ (G955F) */\n\t\treturn;\n'),
]
for old, new in edits:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f'expected 1x {old!r}, found {n}')
    text = text.replace(old, new)
path.write_text(text)
print(f'edited {path}')
