#!/usr/bin/env python3
"""Rebuild the GPU memory dumped by s8mtl_draw.c (s8_dump) from a kernel log.

The driver logs, once, after the first failed vertex/tiler chain:
  S8 MALI| MTL dump begin: chain 0x<jc> slot <n>, <k> regions
  S8 MALI| MTL dump region 0x<gpu> size 0x<size>[ (skipped)]
  S8 MALI| MTL D <gpu hex> <base64 of 72 bytes>        (non-zero blocks only)
  S8 MALI| MTL dump end: <lines> lines

  undump.py LOG OUT.bin BASE SIZE     -> flat image of [BASE, BASE+SIZE), zeros elsewhere
  undump.py LOG                       -> summary only
Prints the chain head, the regions and whether the line count matches.
"""
import base64, re, sys

BLOCK = 72


def parse(path):
    text = open(path, 'rb').read().decode('latin-1')
    parse.text = text
    begin = re.search(r'MTL dump begin: chain 0x([0-9a-f]+) slot (\d+), (\d+) regions', text)
    end = re.search(r'MTL dump end: (\d+) lines', text)
    regions = [(int(a, 16), int(s, 16), bool(k)) for a, s, k in
               re.findall(r'MTL dump region 0x([0-9a-f]+) size 0x([0-9a-f]+)( \(skipped\))?', text)]
    blocks, bad = {}, 0
    for addr, data in re.findall(r'MTL D ([0-9a-f]+) ([A-Za-z0-9+/=]+)', text):
        try:
            b = base64.b64decode(data, validate=True)
        except Exception:
            bad += 1
            continue
        if len(b) != BLOCK:
            bad += 1
            continue
        blocks[int(addr, 16)] = b
    return begin, end, regions, blocks, bad


def main():
    if len(sys.argv) not in (2, 5):
        sys.exit(__doc__)
    begin, end, regions, blocks, bad = parse(sys.argv[1])
    if not begin:
        sys.exit('no "MTL dump begin" in the log')
    print(f'chain 0x{begin.group(1)} slot {begin.group(2)}, {begin.group(3)} regions')
    vt = re.search(r'MTL dump vt chain 0x([0-9a-f]+) fault 0x([0-9a-f]+)', parse.text)
    if vt:
        print(f'vertex/tiler chain 0x{vt.group(1)}, fault pointer 0x{vt.group(2)}')
    for a, s, skipped in regions:
        print(f'  region 0x{a:x} size 0x{s:x}{" skipped" if skipped else ""}')
    want = int(end.group(1)) if end else None
    print(f'{len(blocks)} blocks decoded, {bad} bad lines, end line says {want}'
          f'{"" if want == len(blocks) else "  <-- MISMATCH (lines lost?)"}')
    if len(sys.argv) == 5:
        base, size = int(sys.argv[3], 0), int(sys.argv[4], 0)
        img = bytearray(size)
        n = 0
        for a, b in blocks.items():
            if base <= a and a + BLOCK <= base + size:
                img[a - base:a - base + BLOCK] = b
                n += 1
        open(sys.argv[2], 'wb').write(img)
        print(f'{sys.argv[2]}: {n} blocks placed in 0x{base:x}+0x{size:x}')
    return 0 if want == len(blocks) and bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
